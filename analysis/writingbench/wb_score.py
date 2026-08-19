#!/usr/bin/env python3
"""Score WritingBench responses with the official critic model (local, vLLM).

Each response is scored against its query's 5 instance-specific criteria using
the upstream prompt template; per-item score = mean of the 5, record score =
mean over items. Writes scores/{record}_{arm}.jsonl (per item, per criterion)
and appends a summary row to scores/summary.csv.

    wb_score.py --responses responses/qwen35_base_R8.jsonl \
        [--critic /workspace/writingbench/critic-model]
"""
import argparse
import csv
import json
import os
import re
import sys

WB = "/workspace/writingbench"
QUERIES = f"{WB}/upstream/benchmark_query/benchmark_all.jsonl"
sys.path.insert(0, f"{WB}/upstream")
from prompt import evaluate_system, evaluate_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--critic", default=f"{WB}/critic-model")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    A = ap.parse_args()

    q = {json.loads(l)["index"]: json.loads(l) for l in open(QUERIES)}
    from vllm import LLM, SamplingParams
    llm = LLM(model=A.critic, gpu_memory_utilization=A.gpu_mem, max_model_len=16384)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=2048)

    os.makedirs(f"{WB}/scores", exist_ok=True)
    for rf in A.responses:
        rec = os.path.basename(rf).replace(".jsonl", "")
        items = [json.loads(l) for l in open(rf)]
        prompts, meta = [], []
        for it in items:
            qq = q[it["index"]]
            for ci, crit in enumerate(qq["checklist"]):
                prompts.append([
                    {"role": "system", "content": evaluate_system},
                    {"role": "user", "content": evaluate_prompt.format(
                        criteria=json.dumps(crit), query=qq["query"],
                        response=it["response"])}])
                meta.append((it["index"], ci))
        outs = llm.chat(prompts, sp)
        per = {}
        unparsed = 0
        with open(f"{WB}/scores/{rec}.jsonl", "w") as fh:
            for (idx, ci), o in zip(meta, outs):
                t = o.outputs[0].text
                m = re.search(r'"score"\s*:\s*(\d+)', t)
                s = int(m.group(1)) if m else None
                if s is None:
                    unparsed += 1
                else:
                    per.setdefault(idx, []).append(s)
                fh.write(json.dumps({"index": idx, "criterion": ci, "score": s}) + "\n")
        item_means = [sum(v) / len(v) for v in per.values() if v]
        mean = sum(item_means) / len(item_means)
        n = len(item_means)
        se = (sum((x - mean) ** 2 for x in item_means) / max(1, n - 1) / n) ** 0.5
        print(f"[wb-score] {rec}: mean {mean:.3f} (se {se:.3f}, n {n}, "
              f"{unparsed} unparsed)", flush=True)
        with open(f"{WB}/scores/summary.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([rec, f"{mean:.4f}", f"{se:.4f}", n, unparsed])


if __name__ == "__main__":
    main()
