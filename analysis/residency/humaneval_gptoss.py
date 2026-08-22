#!/usr/bin/env python3
"""HumanEval for gpt-oss with harmony-native prompting.

lm_eval's humaneval_instruct pre-fills the assistant turn with an open ```python fence
(gen_prefix), which violates harmony's channel grammar: gpt-oss-120b degenerates into
token salad on such turns (20b merely tolerates them). Here the model is prompted
plainly, generates its natural analysis+final channels, and the LAST fenced block of
the FINAL channel is scored (subprocess code_eval, as humaneval_gemma). Rows append to
instruct_genbench_vllm.csv with task humaneval_gptoss.

    humaneval_gptoss.py --model gptoss_120b --arm R4 [--reasoning-effort low]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
from instruct_selfce import MODELS                                   # noqa: E402
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402

FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def final_channel(text):
    if "<|channel|>final<|message|>" in text:
        return text.rsplit("<|channel|>final<|message|>", 1)[1] \
                   .split("<|return|>")[0].split("<|end|>")[0]
    if "<|channel|>" in text:
        return ""                                # truncated inside analysis
    return text.split("<|return|>")[0].split("<|end|>")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("gptoss_20b", "gptoss_120b"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=("low", "medium", "high"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="high effort needs 4096+: 35% of high-effort analyses exceed 2048")
    A = ap.parse_args()
    M = MODELS[A.model]
    tag = A.tag or A.model
    # dump files are keyed (record, arm, task): the effort must live in the record
    # or two efforts silently overwrite each other's dumps
    assert not A.reasoning_effort or A.reasoning_effort in tag, \
        f"--reasoning-effort {A.reasoning_effort} requires a --tag containing it"

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    import genprotocol
    genprotocol.check_dump_dir()      # dumps are default-on: fail before engine boot
    from datasets import load_dataset
    probs = list(load_dataset("openai/openai_humaneval", split="test"))
    if A.limit:
        probs = probs[: A.limit]

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    llm = LLM(model=M["path"], enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
              max_model_len=max(4096, A.max_tokens + 512),
              enable_prefix_caching=False)
    R = None if A.arm == "free" else int(A.arm.lstrip("R"))
    if R is not None:
        assert R >= M["k"]
    DEC.update(on=R is not None, R=R or 0, swaps=1)
    DEC["state"].clear()

    try:
        gc = json.load(open(os.path.join(M["path"], "generation_config.json")))
    except (FileNotFoundError, ValueError):
        gc = {}
    ck = {"chat_template_kwargs": {"reasoning_effort": A.reasoning_effort}} \
        if A.reasoning_effort else {}
    msgs = [[{"role": "user", "content":
              "Complete the following Python function. Provide the complete function "
              "in a single ```python code block.\n\n" + p["prompt"]}] for p in probs]
    t0 = time.time()
    outs = llm.chat(msgs, SamplingParams(
        temperature=gc.get("temperature", 1.0), top_p=gc.get("top_p", 1.0),
        seed=1234, max_tokens=A.max_tokens, skip_special_tokens=False), **ck)

    raws = [o.outputs[0].text for o in outs]
    finals = [final_channel(t) for t in raws]
    preds = [[(FENCE.findall(f) or [f])[-1]] for f in finals]
    tests = [p["test"] + f"\ncheck({p['entry_point']})" for p in probs]
    json.dump({"preds": preds, "tests": tests}, open("/tmp/heg_preds.json", "w"))
    out = subprocess.run(["/workspace/venv_fla/bin/python",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "heg_scorer.py")],
                         capture_output=True, text=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("PASS1")]
    assert line, f"scorer failed: {out.stderr[-400:]}"
    p1 = float(line[0].split()[1])
    iline = [ln for ln in out.stdout.splitlines() if ln.startswith("ITEMS")]
    passed = [c == "1" for c in iline[0].split()[1]] if iline else [None] * len(raws)
    secs = time.time() - t0
    print(f"[hgo] {tag} {A.arm} effort={A.reasoning_effort or 'default'}: "
          f"pass@1 = {p1:.4f} ({len(probs)} problems, {secs:.0f}s)", flush=True)

    import torch
    td = "/workspace/instruct-traj/genbench_tokens"
    os.makedirs(td, exist_ok=True)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(M["path"])

    def _ntok(t):
        return len(tk(t, add_special_tokens=False).input_ids)
    # think_toks: everything before the final channel opens (analysis channel);
    # marker absent with <|channel|> present => truncated inside analysis: all think
    MARK = "<|channel|>final<|message|>"
    genprotocol.write_dump(tag, A.arm, "humaneval_gptoss", [
        {"doc": p["task_id"], "raw": r, "gen_toks": len(o.outputs[0].token_ids),
         "think_toks": _ntok(r.rsplit(MARK, 1)[0]) if MARK in r
         else (_ntok(r) if "<|channel|>" in r else 0),
         "pass": ps}
        for p, r, o, ps in zip(probs, raws, outs, passed)], len(probs))
    torch.save({"items": [{"doc_id": i, "ids": tk(r, add_special_tokens=False).input_ids}
                          for i, r in enumerate(raws)]},
               os.path.join(td, f"{tag}_{A.arm}_humaneval_gptoss.pt"))

    with open(os.path.join(ABLATIONS, "instruct_genbench_vllm.csv"), "a",
              newline="") as fh:
        csv.writer(fh).writerow(
            [tag, M["E"], M["k"], A.arm, R or "", "humaneval_gptoss",
             "pass@1,channel-aware", f"{p1:.6f}",
             A.limit or "full", A.max_tokens, f"{secs:.0f}"])


if __name__ == "__main__":
    main()
