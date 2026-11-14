#!/usr/bin/env python3
"""Bisect Skliar's lambda until its loads per token-layer equal ours (within --tol, relative) on the sweep rows.
Loads fall as lambda grows. Each probe is one ref_model.py run; its log is kept.

    python bisect_lambda.py --target 1.0 --lo 0.75 --hi 0.7763 --log-dir out -- <ref_model.py args>
"""
import argparse, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, required=True); ap.add_argument("--lo", type=float, required=True)
ap.add_argument("--hi", type=float, required=True); ap.add_argument("--tol", type=float, default=0.005)
ap.add_argument("--log-dir", required=True); ap.add_argument("--iters", type=int, default=12)
a, rest = ap.parse_known_args()
rest = [x for x in rest if x != "--"]


def probe(lam):
    out = subprocess.run([sys.executable, "scripts/serving_convert/ref_model.py", *rest, "--routing", "skliar",
                          "--cb-lambda", f"{lam:.5f}"], capture_output=True, text=True).stdout
    open(f"{a.log_dir}/bisect_{lam:.5f}.log", "w").write(out)
    ld = float(re.search(r"loads per token-layer ([0-9.]+)", out).group(1))
    loss = float(re.search(r"^loss ([0-9.]+)", out, re.M).group(1))
    print(f"lambda {lam:.5f}: loads {ld:.4f} loss {loss:.5f}", flush=True)
    return ld


lo, hi = a.lo, a.hi                    # loads(lo) >= target >= loads(hi)
for _ in range(a.iters):
    mid = (lo + hi) / 2
    ld = probe(mid)
    if abs(ld - a.target) <= a.tol * a.target:
        print(f"LAMBDA {mid:.5f}"); sys.exit(0)
    lo, hi = (mid, hi) if ld > a.target else (lo, mid)
print(f"LAMBDA {(lo + hi) / 2:.5f} (not within tol)"); sys.exit(0)
