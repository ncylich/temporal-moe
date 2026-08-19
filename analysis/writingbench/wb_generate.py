#!/usr/bin/env python3
"""WritingBench response generation under residency arms.

Boots the project's constrained vLLM stack (vllm_glue + vllm_residency + DEC),
generates responses for the English WritingBench queries, writes
responses/{record}_{arm}.jsonl with {"index", "response"}.

    wb_generate.py --model-path /workspace/instruct-models/qwen35-35b-a3b-instruct \
        --record qwen35_base --arm R8 --n 50 --think off
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/workspace/temporal-moe/analysis/residency")
sys.path.insert(0, "/workspace/temporal-moe/analysis")

WB = "/workspace/writingbench"
QUERIES = f"{WB}/upstream/benchmark_query/benchmark_all.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--arm", required=True, help="free or R<n>")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0, help="subset start within English queries")
    ap.add_argument("--suffix", default="", help="appended to output record name")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.95)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--think", choices=("default", "on", "off"), default="off")
    A = ap.parse_args()

    rows = [json.loads(l) for l in open(QUERIES)]
    en = [r for r in rows if r.get("lang") == "en"][A.offset: A.offset + A.n]
    print(f"[wb-gen] {len(en)} English queries", flush=True)

    import vllm_glue
    vllm_glue.install()
    import vllm_residency  # noqa: F401
    from decode_state import DEC
    from vllm import LLM, SamplingParams

    llm = LLM(model=A.model_path, enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_new + 2048, max_num_seqs=A.max_seqs,
              enable_prefix_caching=False)
    R = None if A.arm == "free" else int(A.arm.lstrip("R"))
    DEC.update(on=R is not None, R=R or 0, swaps=1)
    DEC["state"].clear()

    sp = SamplingParams(temperature=0.7, top_p=0.8, seed=1234, max_tokens=A.max_new)
    msgs = [[{"role": "user", "content": r["query"]}] for r in en]
    ctk = ({} if A.think == "default"
           else {"chat_template_kwargs": {"enable_thinking": A.think == "on"}})
    outs = llm.chat(msgs, sp, **ctk)

    os.makedirs(f"{WB}/responses", exist_ok=True)
    out = f"{WB}/responses/{A.record}_{A.arm}{A.suffix}.jsonl"
    with open(out, "w") as fh:
        for r, o in zip(en, outs):
            fh.write(json.dumps({"index": r["index"],
                                 "response": o.outputs[0].text}) + "\n")
    lens = [len(o.outputs[0].token_ids) for o in outs]
    print(f"[wb-gen] DONE {out}: {len(en)} responses, mean {sum(lens)/len(lens):.0f} "
          f"tokens, {sum(x >= A.max_new for x in lens)} capped", flush=True)


if __name__ == "__main__":
    main()
