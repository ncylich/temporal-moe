#!/usr/bin/env python3
"""HumanEval for think-in-text models with a plain (unprimed) chat prompt.

lm_eval's humaneval_instruct pre-fills the assistant turn with an open ```python fence
(gen_prefix). For models whose template pre-opens a think block (qwen3.5) the primer
lands INSIDE thinking and derails generation (qwen: 0.93 raw-scored confusion ->
0.16 under answer-only scoring). Same class as gpt-oss's harmony clash; same cure as
humaneval_gptoss: plain instruction, natural think+answer, LAST fenced block of the
post-think text scored via the subprocess code_eval scorer.

    humaneval_think.py --model qwen35_instruct --arm R8 [--think off]
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


def answer_part(text):
    return text.split("</think>", 1)[1] if "</think>" in text else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=("qwen35_instruct", "lfm25_instruct"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--think", choices=("default", "off"), default="default")
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--path", default=None)
    A = ap.parse_args()
    M = MODELS[A.model]
    if A.path:
        M = dict(M, path=A.path)
    tag = A.tag or A.model

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    from datasets import load_dataset
    probs = list(load_dataset("openai/openai_humaneval", split="test"))
    if A.limit:
        probs = probs[: A.limit]

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    llm = LLM(model=M["path"], enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_tokens + 1024, enable_prefix_caching=False)
    R = None if A.arm == "free" else int(A.arm.lstrip("R"))
    if R is not None:
        assert R >= M["k"]
    DEC.update(on=R is not None, R=R or 0, swaps=1)
    DEC["state"].clear()

    try:
        gc = json.load(open(os.path.join(M["path"], "generation_config.json")))
    except (FileNotFoundError, ValueError):
        gc = {}
    sp = SamplingParams(
        temperature=A.temperature if A.temperature is not None
        else gc.get("temperature", 1.0),
        top_p=A.top_p if A.top_p is not None else gc.get("top_p", 1.0),
        top_k=gc.get("top_k") or -1,
        presence_penalty=A.presence_penalty
        if A.presence_penalty is not None else gc.get("presence_penalty", 0.0),
        seed=1234, max_tokens=A.max_tokens)
    ck = {}
    if A.think == "off":
        ck["chat_template_kwargs"] = {"enable_thinking": False}
    msgs = [[{"role": "user", "content":
              "Complete the following Python function. Provide the complete function "
              "in a single ```python code block.\n\n" + p["prompt"]}] for p in probs]
    t0 = time.time()
    outs = llm.chat(msgs, sp, **ck)

    raws = [o.outputs[0].text for o in outs]
    finals = [answer_part(t) for t in raws]
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
    secs = time.time() - t0
    capped = sum(len(o.outputs[0].token_ids) >= A.max_tokens - 8 for o in outs)
    print(f"[hvt] {tag} {A.arm} think={A.think}: pass@1 = {p1:.4f} "
          f"({len(probs)} problems, {capped} capped, {secs:.0f}s)", flush=True)

    import torch
    td = "/workspace/instruct-traj/genbench_tokens"
    os.makedirs(td, exist_ok=True)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(M["path"])
    torch.save({"items": [{"doc_id": i, "ids": tk(r, add_special_tokens=False).input_ids}
                          for i, r in enumerate(raws)]},
               os.path.join(td, f"{tag}_{A.arm}_humaneval_think.pt"))

    with open(os.path.join(ABLATIONS, "instruct_genbench_vllm.csv"), "a",
              newline="") as fh:
        csv.writer(fh).writerow(
            [tag, M["E"], M["k"], A.arm, R or "", "humaneval_think",
             "pass@1,channel-aware", f"{p1:.6f}",
             A.limit or "full", A.max_tokens, f"{secs:.0f}"])


if __name__ == "__main__":
    main()
