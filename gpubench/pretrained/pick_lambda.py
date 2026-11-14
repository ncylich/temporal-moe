#!/usr/bin/env python3
"""Pick Skliar's lambda whose loads per token-layer equal ours, from ref_model.py sweep logs.
Loads fall as lambda grows; lambda* is the linear interpolation between the two bracketing grid points.

    python pick_lambda.py ours.log skliar_lam*.log  -> prints lambda* and the table
"""
import re, sys


def read(p):
    t = open(p).read()
    ld = float(re.search(r"loads per token-layer ([0-9.]+)", t).group(1))
    loss = float(re.search(r"^loss ([0-9.]+)", t, re.M).group(1))
    lam = re.search(r"lambda=([0-9.]+)", t)
    return (float(lam.group(1)) if lam else None), ld, loss


ours = read(sys.argv[1])
pts = sorted(read(p) for p in sys.argv[2:])
print(f"ours: loads {ours[1]:.4f} loss {ours[2]:.5f}")
for lam, ld, loss in pts:
    print(f"skliar lambda {lam:.3f}: loads {ld:.4f} loss {loss:.5f}")
for (l0, a0, _), (l1, a1, _) in zip(pts, pts[1:]):
    if (a0 - ours[1]) * (a1 - ours[1]) <= 0:
        lam = l0 + (ours[1] - a0) * (l1 - l0) / (a1 - a0) if a1 != a0 else l0
        print(f"LAMBDA {lam:.4f}")
        sys.exit(0)
print("LAMBDA none: ours' loads are outside the swept range"); sys.exit(1)
