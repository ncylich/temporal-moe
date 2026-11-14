"""Is the null-battery gate flag on four models a property of the models, or of a 24-expert sample?

Direct test: re-run the same battery arm on the same models with the expert cap raised. If the
deviation from 0.500 shrinks as the sample grows, the flag was estimator noise. If it holds, the
models genuinely fail and their numbers must not be used.
"""
import os, sys, statistics as st
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, "analysis/probes")
import numpy as np
import delex_null_check as nc

FLAGGED = ["flame38m_g1_moe", "flame512_g1_temporal",
           "g3_moe_s0_1e16_sigmoid_seed2", "g3_tmoe_s0_1e16_mom"]

print(f"{'run':32} {'n=24':>10} {'n=256':>10}   verdict")
print("-" * 76)
for run in FLAGGED:
    out = {}
    for cap in (24, 256):
        rows, _ = nc.battery(run, max_experts=cap)
        iid = [float(r[-3]) for r in rows if r[5] == "iid"] if rows else []
        # locate the iid arm robustly: find the column holding the null name
        if not iid and rows:
            for r in rows:
                pass
        vals = []
        for r in rows or []:
            rr = list(r)
            if "iid" in [str(x) for x in rr]:
                for x in rr:
                    if isinstance(x, float) and 0.3 < x < 0.7:
                        vals.append(x); break
        out[cap] = st.median(vals) if vals else float("nan")
    a, b = out[24], out[256]
    da, db = abs(a - 0.5), abs(b - 0.5)
    verdict = ("noise: deviation shrinks with sample" if db < da * 0.7 else
               "PERSISTS: not a sample-size effect" if db > 0.002 else "within gate at n=256")
    print(f"{run:32} {a:>10.4f} {b:>10.4f}   {verdict}  (dev {da:.4f} -> {db:.4f})")
