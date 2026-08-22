#!/usr/bin/env python3
"""MMLU for gpt-oss with harmony-tolerant answer extraction.

The stock mmlu_flan_cot_fewshot get-answer filter only accepts "The answer is (X)";
gpt-oss answers in its final channel as e.g. "**Answer: (B) 4**", flooring every cell
regardless of correctness (probe: reasoning and letter correct, extraction 0). Here the
suite path is unchanged (chat template, greedy, final-channel filter, stateful rule on
generated tokens) and only the extraction differs: last "(A)-(D)"-style letter, with
"The answer is (X)" still matched first. Rows append to instruct_genbench_vllm.csv with
task mmlu_gptoss_relaxed, applied identically to every arm.

    mmlu_gptoss.py --model gptoss_120b --arms free,R4,R16
"""
import argparse
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
from instruct_selfce import MODELS                                   # noqa: E402
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402

STRICT = re.compile(r"[Tt]he answer is \(?([A-D])\)?")
RELAXED = re.compile(r"\(([A-D])\)|\*\*?\s*[Aa]nswer\s*[:\-]?\s*\(?([A-D])\)?")


FINAL_LINE = re.compile(r"^\W*A(?:nswer)?\s*[:=]\s*\(?([A-D])\)?", re.I | re.M)


def extract(text):
    m = STRICT.search(text)
    if m:
        return m.group(1)
    fin = None                       # explicit final-answer lines beat trailing
    for m2 in FINAL_LINE.finditer(text[-400:]):   # option-analysis mentions
        fin = m2.group(1)
    if fin:
        return fin.upper()
    hits = RELAXED.findall(text)
    if hits:
        a, b = hits[-1]
        return a or b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--limit", type=int, default=4, help="items per subject")

    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--gen-cap", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=5632,
                    help="thinking arms need prompt + 4096: use 8192")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=("low", "medium", "high"))
    ap.add_argument("--think", choices=("default", "on", "off"), default="default",
                    help="chat-template thinking toggle (enable_thinking kwarg)")
    ap.add_argument("--record-as", default=None)
    ap.add_argument("--path", default=None, help="override checkpoint dir (merged adapters)")
    ap.add_argument("--csv-name", default="instruct_genbench_vllm.csv",
                    help="diagnostics use screening_genbench.csv")
    A = ap.parse_args()
    M = MODELS[A.model]
    if A.path:
        M = dict(M, path=A.path)

    import genprotocol
    genprotocol.check_dump_dir()      # dumps are default-on: fail before engine boot

    vllm_glue.install()
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    kw = {}
    if M["arch"] == "gemma4":
        # transformers 5.15 marks head_dim per-layer on gemma4; vLLM reads it
        # globally (same accommodation as instruct_genbench_vllm)
        kw["hf_overrides"] = {"allow_global_per_layer_attribute_access": True}
    lm = VLLM(pretrained=M["path"], batch_size="auto", max_gen_toks=A.gen_cap,
              max_model_len=A.max_model_len, gpu_memory_utilization=A.gpu_mem,
              enforce_eager=True, enable_prefix_caching=False, dtype="auto", **kw)

    if A.reasoning_effort or A.think != "default":
        _tk = lm.tokenizer
        _orig_act = _tk.apply_chat_template
        _extra = {}
        if A.reasoning_effort:
            _extra["reasoning_effort"] = A.reasoning_effort
        if A.think != "default":
            _extra["enable_thinking"] = A.think == "on"
        _tk.apply_chat_template = lambda *aa, **kk: _orig_act(
            *aa, **{**kk, **_extra})

    # arch-appropriate think marker (this harness now serves every model's
    # relaxed-extraction MMLU, not only gpt-oss)
    THINK_MARK = {"qwen3_5": "</think>", "lfm": "</think>",
                  "gemma4": "<channel|>",
                  "gptoss": "<|channel|>final<|message|>"}.get(M["arch"])
    if M["arch"] == "qwen3_5" and A.think == "off":
        THINK_MARK = None
    import genprotocol
    genprotocol.install(lm, cap=A.gen_cap, think_marker=THINK_MARK)

    # card-recipe sampling, as instruct_genbench_vllm (greedy never; 1.0/1.0 only
    # when the model ships that): recipe from the shipped generation_config
    import json as _json
    try:
        _gc = _json.load(open(os.path.join(M["path"], "generation_config.json")))
    except (FileNotFoundError, ValueError):
        _gc = {}
    _has_recipe = any(k in _gc for k in ("temperature", "top_p", "top_k"))
    _dt, _dp = (1.0, 1.0) if _has_recipe else (0.7, 0.95)
    _t, _p = _gc.get("temperature", _dt), _gc.get("top_p", _dp)
    _k = _gc.get("top_k") or -1
    gen_kwargs = f"do_sample=True,temperature={_t},top_p={_p},top_k={_k},seed=1234"
    if _gc.get("presence_penalty"):
        gen_kwargs += f",presence_penalty={_gc['presence_penalty']}"
    if _gc.get("min_p"):
        gen_kwargs += f",min_p={_gc['min_p']}"
    if M["arch"] == "gptoss":
        gen_kwargs += ",skip_special_tokens=False"   # channel markers must survive
    print(f"[mmlu] sampling: temp={_t} top_p={_p} top_k={_k} (model config)",
          flush=True)

    out = os.path.join(ABLATIONS, A.csv_name)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    for arm in A.arms.split(","):
        R = None if arm == "free" else int(arm.lstrip("R"))
        DEC.update(on=R is not None, R=R or 0, swaps=1)
        DEC["state"].clear()
        t0 = time.time()
        res = simple_evaluate(model=lm, tasks=["mmlu_flan_cot_fewshot"], limit=A.limit,
                              apply_chat_template=True,
                              gen_kwargs=gen_kwargs,
                              log_samples=True)
        n = hit = miss_extract = n_eval = 0
        dump = []          # per ARM, across all subjects (was per-subject: dumps
        hit_strict = 0     # held only the final subject's items; scores unaffected)
        for task, samp in res["samples"].items():
            for x in samp:
                n_eval += 1
                resp = x["resps"][0][0] if x.get("resps") else ""
                # raw = pre-strip (channels/think intact); resp = post-strip scored
                raw = genprotocol.FINALS.get((task, x.get("doc_id")), resp)
                gold = re.search(r"\(([A-D])\)", str(x.get("target", "")))
                pred = extract(resp)
                sm = STRICT.search(resp)
                it = {"doc": f"{task}:{x.get('doc_id')}", "raw": raw,
                      "gen_toks": len(lm.tokenizer(
                          raw, add_special_tokens=False).input_ids),
                      "think_toks": (len(lm.tokenizer(
                          raw.rsplit(THINK_MARK, 1)[0] if THINK_MARK in raw else raw,
                          add_special_tokens=False).input_ids)
                          if THINK_MARK else 0),
                      "gold": gold.group(1) if gold else None,
                      "pred_relaxed": pred,
                      "pred_strict": sm.group(1) if sm else None,
                      "text": resp}
                dump.append(it)
                if gold is None:
                    continue           # unscoreable target: dumped, not scored
                n += 1
                miss_extract += pred is None
                hit += pred == gold.group(1)
                hit_strict += bool(sm) and sm.group(1) == gold.group(1)
        acc = hit / max(1, n)
        acc_s = hit_strict / max(1, n)
        secs = time.time() - t0
        print(f"  [{A.model}] {arm} mmlu relaxed={acc:.4f} strict={acc_s:.4f} "
              f"({n} items, {miss_extract} unextracted, {secs:.0f}s)", flush=True)
        # dual rows from the SAME generations + per-item dump: extraction questions
        # become re-analysis, never regeneration (2026-08-16 format-drift finding)
        genprotocol.write_dump(A.record_as or A.model, arm, "mmlu_dual",
                               dump, n_eval)
        for met, val in (("acc,relaxed-extract", acc), ("acc,strict-flan", acc_s)):
            w.writerow([A.record_as or A.model, M["E"], M["k"], arm, R or "",
                        "mmlu_gptoss_relaxed", met, f"{val:.6f}", A.limit,
                        A.gen_cap, f"{secs:.0f}"])
        fh.flush()
    fh.close()
    print(f"MMLU-GPTOSS {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
