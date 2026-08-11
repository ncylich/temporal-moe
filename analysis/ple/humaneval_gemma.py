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

CHANNEL = re.compile(r"<channel\|>.*?<\|channel>", re.S)
FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract(text):
    text = CHANNEL.sub("", text)
    blocks = FENCE.findall(text)
    return blocks[-1] if blocks else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--arm", required=True, choices=("free", "R8", "R16"))
    ap.add_argument("--tag", default=None, help="model column label; default from path")
    A = ap.parse_args()
    tag = A.tag or ("gemma4_adapted" if "merged" in A.path else "gemma4_instruct")

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    from datasets import load_dataset
    probs = list(load_dataset("openai/openai_humaneval", split="test"))

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    llm = LLM(model=A.path, enforce_eager=True, gpu_memory_utilization=0.85,
              max_model_len=2560, enable_prefix_caching=False)
    DEC.update(on=A.arm != "free", R=16 if A.arm == "R16" else 8, swaps=1)
    DEC["state"].clear()
    msgs = [[{"role": "user", "content":
              "Complete the following Python function. Provide the complete function "
              "in a single ```python code block.\n\n" + p["prompt"]}] for p in probs]
    t0 = time.time()
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1536))

    preds = [[extract(o.outputs[0].text)] for o in outs]
    tests = [p["test"] + f"\ncheck({p['entry_point']})" for p in probs]
    # code_eval forks sandbox workers, which is unsafe inside the live vLLM process:
    # score in a clean subprocess instead.
    import json as _json
    import subprocess
    _json.dump({"preds": preds, "tests": tests}, open("/tmp/heg_preds.json", "w"))
    out = subprocess.run(["/workspace/venv_fla/bin/python",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "heg_scorer.py")],
                         capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("PASS1")]
    assert line, f"scorer failed: {out.stderr[-400:]}"
    res = {"pass@1": float(line[0].split()[1])}
    secs = time.time() - t0
    print(f"[heg] {tag} {A.arm}: pass@1 = {res['pass@1']:.4f} "
          f"({len(probs)} problems, {secs:.0f}s)", flush=True)

    with open(os.path.join(ABLATIONS, "instruct_genbench_vllm.csv"), "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([tag, 128, 8, A.arm, (16 if A.arm == "R16" else 8) if A.arm != "free" else "",
                    "humaneval_gemma_fixed", "pass@1,channel-aware",
                    f"{res['pass@1']:.6f}", "full", 1536, f"{secs:.0f}"])


if __name__ == "__main__":
    main()
