#!/usr/bin/env python3
"""MBPP for gemma4-IT with its reasoning-channel format handled properly.

Why this exists: HumanEval has only 164 problems, which puts the paired standard error on
a residency gap at 2.9 points -- so it cannot resolve anything smaller than ~5.7. The
adapter's measured effect on math is +3.1. HumanEval therefore CANNOT distinguish "the
adapter does nothing for code" from "the adapter helps code exactly as much as it helps
math"; both read as ~0. MBPP's 500 test problems, pooled with HumanEval's 164, bring the
code surface to n=664 and SE ~1.5, which resolves a math-sized effect.

The stock lm_eval mbpp_instruct task scores this model at 0.28 pass@1 while it scores
0.98 on HumanEval -- the same channel-marker extraction failure humaneval_gemma.py exists
to fix. Same treatment here: strip <|channel>...<channel|> spans, take the LAST fenced
code block, exec against the problem's asserts in a subprocess.

The D7 training pool was screened against MBPP test before this was adopted
(screen_mbpp.py): 12/8471 rows carry any MBPP 8-gram, all 1-2 grams of generic
boilerplate, no problem or solution content.

Rows append to instruct_genbench_vllm.csv with task mbpp_gemma.

    mbpp_gemma.py --path /dev/shm/gemma4-26b-it --arms free,R8,R16 --tag gemma4_instruct
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

CHANNEL = re.compile(r"<\|channel>.*?(?:<channel\|>|\Z)", re.S)
FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract(text, unfinished=False):
    """Last fenced block outside the reasoning channels. Identical rule to
    humaneval_gemma.extract so the two code surfaces are scored the same way: a response
    that hit the cap without closing its channel submitted nothing."""
    if unfinished:
        return ""
    text = CHANNEL.sub("", text)
    blocks = FENCE.findall(text)
    return blocks[-1] if blocks else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--arms", default="free,R8,R16",
                    help="comma list run in ONE engine boot (batch-fair protocol)")
    ap.add_argument("--csv-name", default="instruct_genbench_vllm.csv")
    ap.add_argument("--tag", default=None, help="model column label; default from path")
    ap.add_argument("--think", choices=("on", "off"), default="off")
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    A = ap.parse_args()
    tag = A.tag or ("gemma4_adapted" if "merged" in A.path else "gemma4_instruct")
    assert A.think == "off" or A.tag, "--think on requires an explicit --tag"

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import genprotocol
    genprotocol.check_dump_dir()
    from datasets import load_dataset
    probs = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    if A.limit:
        probs = probs[: A.limit]

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    arms = A.arms.split(",")
    assert all(a == "free" or re.fullmatch(r"R\d+", a) for a in arms) and arms, "bad --arms"
    llm = LLM(model=A.path, enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_model_len, enable_prefix_caching=False)
    # MBPP's convention: the asserts are shown so the model knows the required signature.
    msgs = [[{"role": "user", "content":
              "You are an expert Python programmer. Write a Python function for this "
              "task:\n\n" + p["text"] + "\n\nYour code must pass these tests:\n\n"
              + "\n".join(p["test_list"])
              + "\n\nProvide the complete function in a single ```python code block."}]
            for p in probs]
    tests = [(p.get("test_setup_code") or "") + "\n" + "\n".join(p["test_list"])
             for p in probs]

    _gc = _json.load(open(os.path.join(A.path, "generation_config.json")))
    tk = llm.get_tokenizer()
    preds_path = f"/tmp/mbpp_preds_{os.getpid()}.json"
    for arm in arms:                       # all arms share ONE engine boot
        R = None if arm == "free" else int(arm.lstrip("R"))
        DEC.update(on=R is not None, R=R or 0, swaps=1)
        DEC["state"].clear()
        t0 = time.time()
        outs = llm.chat(msgs, SamplingParams(
            temperature=_gc.get("temperature", 1.0), top_p=_gc.get("top_p", 1.0),
            top_k=_gc.get("top_k") or -1, seed=1234, max_tokens=A.max_tokens),
            chat_template_kwargs={"enable_thinking": A.think == "on"})

        unfin = [len(o.outputs[0].token_ids) >= A.max_tokens - 8
                 and "<channel|>" not in o.outputs[0].text for o in outs]
        preds = [[extract(o.outputs[0].text, u)] for o, u in zip(outs, unfin)]
        _json.dump({"preds": preds, "tests": tests}, open(preds_path, "w"))
        out = subprocess.run(["/workspace/venv_fla/bin/python",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "heg_scorer.py"), preds_path],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("PASS1")]
        assert line, f"scorer failed: {out.stderr[-400:]}"
        p1 = float(line[0].split()[1])
        iline = [l for l in out.stdout.splitlines() if l.startswith("ITEMS")]
        passed = [c == "1" for c in iline[0].split()[1]] if iline else [None] * len(outs)
        secs = time.time() - t0
        print(f"[mbppg] {tag} {arm}: pass@1 = {p1:.4f} "
              f"({len(probs)} problems, {secs:.0f}s)", flush=True)

        def _ntok(t):
            return len(tk(t, add_special_tokens=False).input_ids)
        genprotocol.write_dump(tag, arm, "mbpp_gemma", [
            {"doc": f"mbpp/{p['task_id']}", "raw": o.outputs[0].text,
             "gen_toks": len(o.outputs[0].token_ids),
             "gen_ids": list(o.outputs[0].token_ids),
             "prompt_ids": list(o.prompt_token_ids or []),
             "think_toks": _ntok(o.outputs[0].text)
             - _ntok(CHANNEL.sub("", o.outputs[0].text)),
             "pass": ps, "unfinished": u}
            for p, o, ps, u in zip(probs, outs, passed, unfin)], len(probs))

        with open(os.path.join(ABLATIONS, A.csv_name), "a", newline="") as fh:
            csv.writer(fh).writerow(
                [tag, 128, 8, arm, R or "", "mbpp_gemma", "pass@1,channel-aware",
                 f"{p1:.6f}", A.limit or "full", A.max_tokens, f"{secs:.0f}"])
    os.path.exists(preds_path) and os.remove(preds_path)


if __name__ == "__main__":
    main()
