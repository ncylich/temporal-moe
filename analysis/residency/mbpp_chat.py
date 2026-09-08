#!/usr/bin/env python3
"""MBPP as a standard surface for every released model in Sections 6 and 7, one producer.

Same prompt, sampling protocol and scorer for every family, with the family-specific parts
handled explicitly rather than by lm_eval's task defaults:
  prompt      the MBPP convention (task text + the asserts, ask for one ```python block)
  thinking    stripped before extraction: gemma4 <|channel>...<channel|> spans, qwen3.5 and
              LFM2.5 <think>...</think>, gpt-oss the text after the last final-channel marker;
              a response that hit the cap without closing its thinking submitted nothing
  extraction  the LAST fenced code block (mbpp_gemma / humaneval_gemma rule), scored by
              exec against the problem's asserts in a subprocess (heg_scorer.py)
  sampling    the model's shipped generation_config (never greedy), the runner's no-recipe
              fallback 0.7/0.95 otherwise, seed 1234; presence_penalty from the config goes
              through vllm_glue (TEMPORAL_FAST_PP=1 -> the fast processor, 0 -> vLLM native),
              which is how the two are compared on the same sample
  residency   arms free and R<k>, all in one engine boot (batch-fair protocol)
Rows append to instruct_genbench_vllm.csv with task mbpp_chat and the dump goes to
results/ablations/genbench_samples/<tag>_<arm>_mbpp_chat.json (prompt ids, raw text, extracted
code, pass, thinking tokens, unfinished).

    mbpp_chat.py --model gemma4_instruct --arms free,R8,R16 --limit 40 --tag gemma4_instruct_mbpp40
"""
import argparse
import csv
import json as _json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402
from instruct_selfce import MODELS                                   # noqa: E402

FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
GEMMA_CHANNEL = re.compile(r"<\|channel>.*?(?:<channel\|>|\Z)", re.S)
THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
GPTOSS_FINAL = "<|channel|>final<|message|>"


def strip_thinking(arch, text):
    """Returns (visible_text, unfinished_thinking)."""
    if arch == "gemma4":
        unfinished = "<|channel>" in text and "<channel|>" not in text
        return GEMMA_CHANNEL.sub("", text), unfinished
    if arch in ("qwen3_5", "lfm"):
        unfinished = "<think>" in text and "</think>" not in text
        return THINK.sub("", text), unfinished
    if arch == "gptoss":
        if GPTOSS_FINAL in text:
            return text.rsplit(GPTOSS_FINAL, 1)[1], False
        # no final channel: the analysis ran to the cap (or the model skipped channels)
        return ("", True) if "<|channel|>analysis" in text else (text, False)
    return text, False


MAIN_GUARD = re.compile(r"^if\s+__name__.*$", re.M)


def function_only(code):
    """Diagnostic only (mbpp_rescore.py): drop the model's own self-test scaffold, everything from a
    top-level `if __name__` guard on and trailing top-level assert/print statements, to measure
    how many failures are typos in throwaway boilerplate around a correct function (2026-09-04:
    29 of 500 for base gemma at R8). The scored rule keeps the scaffold."""
    m = MAIN_GUARD.search(code)
    if m:
        code = code[: m.start()]
    lines = code.rstrip().split("\n")
    while lines and re.match(r"^(assert\b|print\s*\()", lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def extract(arch, text, hit_cap):
    vis, unfinished = strip_thinking(arch, text)
    if unfinished or (hit_cap and not FENCE.search(vis)):
        return "", True
    blocks = FENCE.findall(vis)
    code = blocks[-1] if blocks else vis
    for tok in ("<|return|>", "<|end|>", "<end_of_turn>", "<|im_end|>"):
        code = code.replace(tok, "")
    # Scored as written: the whole last block, the model's own self-test scaffold included (the
    # recorded mbpp_gemma rule). A model that breaks that scaffold under the constraint is showing
    # real damage; function_only() above is kept for the diagnostic re-score (mbpp_rescore.py).
    return code, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--path", default=None, help="weights directory (default: the registry path)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--arms", default="free")
    ap.add_argument("--tag", required=True, help="record label (must carry the effort/think variant)")
    ap.add_argument("--task-name", default="mbpp_chat")
    ap.add_argument("--csv-name", default="instruct_genbench_vllm.csv")
    ap.add_argument("--think", choices=("default", "on", "off"), default="off")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="override the shipped value (0 disables); default: generation_config")
    ap.add_argument("--temperature", type=float, default=None, help="override generation_config")
    ap.add_argument("--top-p", type=float, default=None, help="override generation_config")
    A = ap.parse_args()
    M = dict(MODELS[A.model])
    if A.path:
        M["path"] = A.path
    arch = M["arch"]
    if A.reasoning_effort:
        assert A.reasoning_effort in A.tag, "--reasoning-effort must appear in --tag (dumps are keyed by record)"

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import genprotocol
    genprotocol.check_dump_dir()
    from datasets import load_dataset
    probs = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    if A.limit:
        probs = probs[: A.limit]

    # Clamp to the model's context window (OLMoE: 4096): engine length to the window, budget to
    # what fits after the longest MBPP prompt. The row's max_gen_toks column records the budget.
    try:
        _cfg = _json.load(open(os.path.join(M["path"], "config.json")))
        _mpe = int(_cfg.get("max_position_embeddings") or (_cfg.get("text_config") or {}).get("max_position_embeddings") or 0)
    except (FileNotFoundError, ValueError):
        _mpe = 0
    if _mpe and A.max_model_len > _mpe:
        A.max_model_len = _mpe
        A.max_tokens = min(A.max_tokens, _mpe - 768)
        print(f"[mbpp_chat] context window {_mpe}: max_model_len={A.max_model_len} max_tokens={A.max_tokens}", flush=True)

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    arms = A.arms.split(",")
    assert arms and all(a == "free" or re.fullmatch(r"R\d+", a) for a in arms), "bad --arms"
    kw = dict(vllm_glue.llm_kwargs(), gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len)
    if A.max_num_seqs:
        kw["max_num_seqs"] = A.max_num_seqs
    if arch == "gemma4":
        kw["hf_overrides"] = {"allow_global_per_layer_attribute_access": True}
    if arch == "gptoss":
        kw["dtype"] = "auto"
    llm = LLM(model=M["path"], **kw)
    if A.adapter:
        from apply_adapter import apply_adapter
        apply_adapter(llm, A.adapter, M["path"])

    msgs = [[{"role": "user", "content":
              "You are an expert Python programmer. Write a Python function for this "
              "task:\n\n" + p["text"] + "\n\nYour code must pass these tests:\n\n"
              + "\n".join(p["test_list"])
              + "\n\nProvide the complete function in a single ```python code block."}]
            for p in probs]
    tests = [(p.get("test_setup_code") or "") + "\n" + "\n".join(p["test_list"]) for p in probs]

    try:
        _gc = _json.load(open(os.path.join(M["path"], "generation_config.json")))
    except (FileNotFoundError, ValueError):
        _gc = {}
    has_recipe = any(k in _gc for k in ("temperature", "top_p", "top_k"))
    temp = A.temperature if A.temperature is not None else _gc.get("temperature", 1.0 if has_recipe else 0.7)
    top_p = A.top_p if A.top_p is not None else _gc.get("top_p", 1.0 if has_recipe else 0.95)
    top_k = _gc.get("top_k") or -1
    pp = A.presence_penalty if A.presence_penalty is not None else _gc.get("presence_penalty", 0.0)
    min_p = _gc.get("min_p")
    sp_kw = dict(temperature=temp, top_p=top_p, top_k=top_k, seed=1234, max_tokens=A.max_tokens)
    if pp:
        sp_kw["presence_penalty"] = float(pp)
    if min_p:
        sp_kw["min_p"] = float(min_p)
    if arch == "gptoss":
        sp_kw["skip_special_tokens"] = False
    ck = {}
    if A.think != "default":
        ck["enable_thinking"] = A.think == "on"
    if A.reasoning_effort:
        ck["reasoning_effort"] = A.reasoning_effort
    print(f"[mbpp_chat] {A.tag} arch={arch} sampling temp={temp} top_p={top_p} top_k={top_k} pp={pp} "
          f"min_p={min_p} think={A.think} effort={A.reasoning_effort} fast_pp={os.environ.get('TEMPORAL_FAST_PP', '1')} "
          f"n={len(probs)} cap={A.max_tokens}", flush=True)

    tk = llm.get_tokenizer()
    preds_path = f"/tmp/mbpp_chat_preds_{os.getpid()}.json"
    for arm in arms:
        R = None if arm == "free" else int(arm.lstrip("R"))
        DEC.update(on=R is not None, R=R or 0, swaps=1)
        DEC["state"].clear()
        t0 = time.time()
        outs = llm.chat(msgs, SamplingParams(**sp_kw), chat_template_kwargs=ck) if ck else \
            llm.chat(msgs, SamplingParams(**sp_kw))
        hit_cap = [len(o.outputs[0].token_ids) >= A.max_tokens - 8 for o in outs]
        ext = [extract(arch, o.outputs[0].text, h) for o, h in zip(outs, hit_cap)]
        preds = [[c] for c, _ in ext]
        unfin = [u for _, u in ext]
        _json.dump({"preds": preds, "tests": tests}, open(preds_path, "w"))
        out = subprocess.run(["/workspace/venv_fla/bin/python",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "heg_scorer.py"),
                              preds_path], capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("PASS1")]
        assert line, f"scorer failed: {out.stderr[-400:]}"
        p1 = float(line[0].split()[1])
        iline = [l for l in out.stdout.splitlines() if l.startswith("ITEMS")]
        passed = [c == "1" for c in iline[0].split()[1]] if iline else [None] * len(outs)
        secs = time.time() - t0

        def _ntok(t):
            return len(tk(t, add_special_tokens=False).input_ids)
        items = []
        for p, o, (code, u), ps, h in zip(probs, outs, ext, passed, hit_cap):
            raw = o.outputs[0].text
            vis, _ = strip_thinking(arch, raw)
            items.append({"doc": f"mbpp/{p['task_id']}", "raw": raw, "extracted": code,
                          "gen_toks": len(o.outputs[0].token_ids), "gen_ids": list(o.outputs[0].token_ids),
                          "prompt_ids": list(o.prompt_token_ids or []),
                          "think_toks": max(0, _ntok(raw) - _ntok(vis)), "pass": ps,
                          "unfinished": u, "hit_cap": h})
        genprotocol.write_dump(A.tag, arm, A.task_name, items, len(probs))
        n_unf = sum(unfin); n_cap = sum(hit_cap)
        print(f"[mbpp_chat] {A.tag} {arm}: pass@1 = {p1:.4f} ({len(probs)} problems, {n_unf} unfinished, "
              f"{n_cap} at cap, {secs:.0f}s)", flush=True)
        with open(os.path.join(ABLATIONS, A.csv_name), "a", newline="") as fh:
            csv.writer(fh).writerow([A.tag, M["E"], M["k"], arm, R or "", A.task_name,
                                     "pass@1,channel-aware", f"{p1:.6f}", A.limit or "full",
                                     A.max_tokens, f"{secs:.0f}"])
    os.path.exists(preds_path) and os.remove(preds_path)


if __name__ == "__main__":
    main()
