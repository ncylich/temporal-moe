#!/usr/bin/env python3
"""Assemble a training pool the way the committed D7 pool was actually assembled.

build_d7_prompts.py fills mathlane_v2 from WildChat by keyword; the pool that trained every
committed adapter instead took that lane from build_realmath_lane.py (StackMathQA) and
spliced it in -- undocumented until 2026-08-27. This makes the splice explicit so a pool
can be regenerated, scaled, or re-mixed without repeating that mistake.

    splice_pool.py --base d7_prompts.jsonl --realmath realmath_4700.jsonl \
                   --math-rows 4700 --total 8482 --out pool.jsonl
The math lane is taken from --realmath up to --math-rows; the remaining lanes are taken
from --base in their original proportions until --total is reached. Deterministic.
"""
import argparse, json, hashlib
from collections import defaultdict
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", required=True); ap.add_argument("--realmath", required=True)
ap.add_argument("--math-rows", type=int, required=True); ap.add_argument("--total", type=int, required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()
rm = [json.loads(l) for l in open(A.realmath)][: A.math_rows]
for r in rm: r["lane"] = "mathlane_v2"
lanes = defaultdict(list)
for l in open(A.base):
    r = json.loads(l)
    if r["lane"] != "mathlane_v2": lanes[r["lane"]].append(r)
other_total = sum(len(v) for v in lanes.values())
need = A.total - len(rm)
out = list(rm)
for lane, rows in lanes.items():
    take = round(need * len(rows) / other_total)
    out.extend(rows[:take])
seen = set(); dedup = []
for r in out:
    k = " ".join(r["text"].lower().split())
    if k in seen: continue
    seen.add(k); dedup.append(r)
with open(A.out, "w") as f:
    for i, r in enumerate(dedup):
        r["idx"] = i; f.write(json.dumps(r, ensure_ascii=False) + "\n")
counts = defaultdict(int)
for r in dedup: counts[r["lane"]] += 1
print(json.dumps({"out": A.out, "n": len(dedup), "lanes": dict(counts),
                  "sha256": hashlib.sha256(open(A.out, "rb").read()).hexdigest()[:16]}, indent=1))
