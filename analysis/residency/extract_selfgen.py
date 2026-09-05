#!/usr/bin/env python3
"""Turn a self-generated authoring pass into a prompts jsonl.

Stage A (gen_traj_vllm.py) had the model AUTHOR problems from topic seeds. This decodes
the generated half of each row -- the problem statement, not the seed instruction -- and
writes it as a prompts file. Stage B then generates real trajectories from these, so the
adapter trains on the model SOLVING its own problems, which is what math_selfgen was.

Quality gates, because a model asked for "only the problem" will still sometimes solve it
anyway, refuse, or emit a stub:
  * drop rows that leak an answer (the model ignored the instruction)
  * drop rows too short to be a problem, or longer than the prompt budget
  * drop near-duplicates on normalised text, since topic seeds repeat by construction

    extract_selfgen.py --tag selfgen_math_raw --model /dev/shm/gemma4-26b-it --lane math
"""
import argparse
import hashlib
import json
import os
import re

import torch

ANSWER_LEAK = re.compile(
    r"\b(the answer is|final answer|answer:|step 1|solution:|therefore,? (the|we)|"
    r"= *\$?\d+ *(dollars|apples|items)?\s*$)", re.I)
REFUSAL = re.compile(r"\b(i can(no|')t|i'm sorry|as an ai)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--traj", default="/workspace/instruct-traj")
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data")
    ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=1200)
    A = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    rows = torch.load(f"{A.traj}/{A.tag}.pt", weights_only=False)["rows"]

    kept, seen, drop = [], set(), {"leak": 0, "short": 0, "long": 0, "dup": 0, "refusal": 0}
    for r in rows:
        gen = tok.decode(r["ids"][int(r["prompt_len"]):], skip_special_tokens=True).strip()
        if REFUSAL.search(gen):
            drop["refusal"] += 1; continue
        if ANSWER_LEAK.search(gen):
            drop["leak"] += 1; continue          # it solved it despite being told not to
        if len(gen) < A.min_chars:
            drop["short"] += 1; continue
        if len(gen) > A.max_chars:
            drop["long"] += 1; continue
        norm = " ".join(gen.lower().split())
        if norm in seen:
            drop["dup"] += 1; continue
        seen.add(norm)
        kept.append({"idx": len(kept), "lane": f"selfgen_{A.lane}",
                     "source": "selfgen", "text": gen})

    path = os.path.join(A.out, f"selfgen_{A.lane}_prompts.jsonl")
    with open(path, "w") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"[extract] {len(rows)} authored -> {len(kept)} kept  dropped={drop}", flush=True)
    print(f"[extract] {path}\n[extract] sha256 {sha}", flush=True)
    for k in kept[:2]:
        print(f"[extract] sample: {k['text'][:150]}...", flush=True)


if __name__ == "__main__":
    main()
