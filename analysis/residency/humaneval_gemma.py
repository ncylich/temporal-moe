#!/usr/bin/env python3
"""HumanEval for gemma4-IT with its reasoning-channel format handled properly.

gemma4's chat template emits intermediate reasoning wrapped in <channel|>...<|channel>
spans; lm_eval's stock filter extracted the first code-looking text (the truncated
thought) at a 640-token cap, flooring every cell. Here: 1536-token budget, strip the
channel spans, take the LAST fenced code block, score with the code_eval metric
(pass@1, greedy). Rows append to instruct_genbench_vllm.csv with task
humaneval_gemma_fixed.

    humaneval_gemma.py --path /dev/shm/gemma4-26b-it --arm free
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
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402

CHANNEL = re.compile(r"<\|channel>.*?(?:<channel\|>|\Z)", re.S)
FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract(text):
    text = CHANNEL.sub("", text)
    blocks = FENCE.findall(text)
    return blocks[-1] if blocks else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--arm", default=None, choices=("free", "R8", "R16"),
                    help="single arm (legacy)")
    ap.add_argument("--arms", default=None,
                    help="comma list run in ONE engine boot (e.g. free,R8,R16)")
    ap.add_argument("--csv-name", default="instruct_genbench_vllm.csv",
                    help="screening runs use screening_genbench.csv")
    ap.add_argument("--tag", default=None, help="model column label; default from path")
    ap.add_argument("--think", choices=("on", "off"), default="off",
                    help="enable_thinking template kwarg (off pre-closes the channel)")
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke)")
    ap.add_argument("--max-tokens", type=int, default=1536,
                    help="generation budget (think-on needs ~3072: thinking alone can "
                         "run past 1800 tokens)")
    ap.add_argument("--max-model-len", type=int, default=2560)
    A = ap.parse_args()
    tag = A.tag or ("gemma4_adapted" if "merged" in A.path else "gemma4_instruct")
    # dump files are keyed (record, arm, task): think-on needs its own record name
    # or it overwrites the think-off dumps
    assert A.think == "off" or A.tag, "--think on requires an explicit --tag"

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import genprotocol
    genprotocol.check_dump_dir()      # dumps are default-on: fail before engine boot
    from datasets import load_dataset
    probs = list(load_dataset("openai/openai_humaneval", split="test"))
    if A.limit:
        probs = probs[: A.limit]

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    arms = A.arms.split(",") if A.arms else [A.arm]
    assert all(a == "free" or re.fullmatch(r"R\d+", a) for a in arms) and arms, "bad --arm(s)"
    llm = LLM(model=A.path, enforce_eager=True, gpu_memory_utilization=0.85,
              max_model_len=A.max_model_len, enable_prefix_caching=False)
    msgs = [[{"role": "user", "content":
              "Complete the following Python function. Provide the complete function "
              "in a single ```python code block.\n\n" + p["prompt"]}] for p in probs]
    # sampling per gemma's shipped generation_config (greedy loops on long reasoning)
    import json as _json
    import subprocess
    _gc = _json.load(open(os.path.join(A.path, "generation_config.json")))
    tk = llm.get_tokenizer()
    for arm in arms:                       # all arms share ONE engine boot
        R = None if arm == "free" else int(arm.lstrip("R"))
        DEC.update(on=R is not None, R=R or 0, swaps=1)
        DEC["state"].clear()
        t0 = time.time()
        outs = llm.chat(msgs, SamplingParams(
            temperature=_gc.get("temperature", 1.0), top_p=_gc.get("top_p", 1.0),
            top_k=_gc.get("top_k") or -1, seed=1234, max_tokens=A.max_tokens),
            chat_template_kwargs={"enable_thinking": A.think == "on"})

        preds = [[extract(o.outputs[0].text)] for o in outs]
        tests = [p["test"] + f"\ncheck({p['entry_point']})" for p in probs]
        # code_eval forks sandbox workers, which is unsafe inside the live vLLM
        # process: score in a clean subprocess instead.
        _json.dump({"preds": preds, "tests": tests}, open("/tmp/heg_preds.json", "w"))
        out = subprocess.run(["/workspace/venv_fla/bin/python",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "heg_scorer.py")],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("PASS1")]
        assert line, f"scorer failed: {out.stderr[-400:]}"
        res = {"pass@1": float(line[0].split()[1])}
        iline = [l for l in out.stdout.splitlines() if l.startswith("ITEMS")]
        passed = [c == "1" for c in iline[0].split()[1]] if iline else [None] * len(outs)
        secs = time.time() - t0
        print(f"[heg] {tag} {arm}: pass@1 = {res['pass@1']:.4f} "
              f"({len(probs)} problems, {secs:.0f}s)", flush=True)

        def _ntok(t):
            return len(tk(t, add_special_tokens=False).input_ids)
        # think_toks: token mass of the channel spans (raw minus channel-stripped)
        genprotocol.write_dump(tag, arm, "humaneval_gemma_fixed", [
            {"doc": p["task_id"], "raw": o.outputs[0].text,
             "gen_toks": len(o.outputs[0].token_ids),
             "think_toks": _ntok(o.outputs[0].text)
             - _ntok(CHANNEL.sub("", o.outputs[0].text)),
             "pass": ps}
            for p, o, ps in zip(probs, outs, passed)], len(probs))

        with open(os.path.join(ABLATIONS, A.csv_name), "a", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([tag, 128, 8, arm, R or "",
                        "humaneval_gemma_fixed", "pass@1,channel-aware",
                        f"{res['pass@1']:.6f}", A.limit or "full", A.max_tokens,
                        f"{secs:.0f}"])


if __name__ == "__main__":
    main()
