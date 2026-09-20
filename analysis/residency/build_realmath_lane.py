#!/usr/bin/env python3
"""Real, lineage-clean math prompts from math.stackexchange (math-ai/StackMathQA).

Why real math, and why THIS source. Swapping the math lane to purely self-generated
problems halved GSM8K constraint damage on gemma (-5.5 -> -3.0 vs an unadapted base at
-6.0) but cost instruction-following: IFEval's within-run damage went -0.5 -> -5.5 even
though its lane was byte-identical across arms. The most likely cause is distributional
narrowing -- 2,671 problems generated from 2,700 TEMPLATED instructions share a phrasing
skeleton, and training on that narrows the model's output distribution. Real human-written
questions restore the variety without giving up the math.

Lineage. StackMathQA is questions real people asked on math.stackexchange: not a benchmark,
not derived from one. The obvious alternative, AI-MO/NuminaMath-CoT, is DISQUALIFIED -- its
own source field shows it bundles orca_math, gsm8k, math and synthetic_math, so it is
benchmark-family data in the precise sense the rule forbids, and Orca-Math is the lane that
produced the fake +8 GSM8K in the D1-vs-D4 ablation. Using the most popular "real math
dataset" would have reproduced that failure exactly.

Filtering keeps concrete, computational questions and drops proof/abstract-algebra style
ones, since the target benchmark is arithmetic word problems. The 8-gram screen in
build_d7_prompts.py still runs over whatever this produces.

    build_realmath_lane.py --n 1170
"""
import argparse
import hashlib
import json
import os
import re

COMPUTE = re.compile(r"\b(how many|how much|calculate|compute|find the (value|number|total|"
                     r"amount|cost|price|average|percentage)|what is the (value|total|cost|"
                     r"average|probability)|total cost|per hour|per day)\b", re.I)
ABSTRACT = re.compile(r"\b(prove|proof|theorem|lemma|axiom|isomorphi|topolog|manifold|"
                      r"homomorph|eigen|banach|hilbert|convergen|continuity|intuitive)\b", re.I)
LATEXY = re.compile(r"\\(begin|frac|int|sum|prod|lim)\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data")
    ap.add_argument("--scan-cap", type=int, default=400000)
    A = ap.parse_args()
    os.environ.setdefault("HF_TOKEN", open(os.path.expanduser(
        "~/.cache/huggingface/token")).read().strip())
    from datasets import load_dataset

    ds = load_dataset("math-ai/StackMathQA", split="train", streaming=True)
    kept, seen = [], set()
    for i, r in enumerate(ds):
        if i >= A.scan_cap or len(kept) >= A.n:
            break
        q = " ".join((r.get("Q") or "").split())
        if not (60 <= len(q) <= 700):
            continue
        if ABSTRACT.search(q) or LATEXY.search(q):    # keep it concrete and plain-text
            continue
        if not re.search(r"\d", q) or not COMPUTE.search(q):
            continue
        norm = q.lower()
        if norm in seen:
            continue
        seen.add(norm)
        kept.append({"idx": len(kept), "lane": "realmath", "source": "stackmathqa",
                     "text": q})
    path = os.path.join(A.out, f"realmath_{A.n}.jsonl")
    with open(path, "w") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"[realmath] kept {len(kept)} of {A.n} requested -> {path}", flush=True)
    print(f"[realmath] sha256 {sha}", flush=True)
    for k in kept[:2]:
        print(f"[realmath] sample: {k['text'][:150]}", flush=True)


if __name__ == "__main__":
    main()
