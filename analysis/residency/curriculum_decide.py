#!/usr/bin/env python3
"""Decision rules of results/ablations/CURRICULUM_PLAN.md, applied to curriculum_1e17.csv so the
supervisor can pick the next stage unattended. Reference = the C0 environment control when it exists
(same code path, same numerics), else the recorded full-MoE baseline. Win bar 0.010 CE below the
reference, promising within 0.005 above it.

    curriculum_decide.py round2   -> arms to run next on grain 3 (empty = stop)
    curriculum_decide.py best     -> the best grain-3 arm (by CE) and its verdict
    curriculum_decide.py transfer -> the arm to run on grain 1 if the best is a win or promising
    curriculum_decide.py promote  -> the arm to promote to 1e18 if it won on grain 3 and did not lose on grain 1
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

CSV = os.path.join(ABLATIONS, "curriculum_1e17.csv")
BASE = {"g3": 3.507410, "g1": 3.492985}
WIN, PROMISING = -0.010, 0.005


def load():
    rows = [r for r in csv.DictReader(l for l in open(CSV) if not l.startswith("#"))]
    return {(r["grain"], r["arm"]): float(r["test_CE"]) for r in rows}


def ref(ce, grain):
    return ce.get((grain, "C0"), BASE[grain])


def verdicts(ce, grain):
    r = ref(ce, grain)
    return {a: v - r for (g, a), v in ce.items() if g == grain and a != "C0"}


def family(arm):
    """Refinements of an arm's family (round 2)."""
    def p(x):
        return f"{x:.3g}".replace(".", "p")
    if arm.startswith("SWW"):
        f = float(arm[3:].replace("p", "."))
        return [f"SWW{p(f - 0.15)}", f"SWW{p(f + 0.15)}"]
    if arm.startswith("SW"):
        f = float(arm[2:].replace("p", "."))
        return [f"SW{p(max(0.2, f - 0.15))}", f"SW{p(min(0.85, f + 0.15))}", f"SWW{p(f)}"]
    if arm.startswith("RAMP"):
        return ["RAMP0p6", "RAMP0p9"]
    if arm.startswith("HET"):
        return ["HET0p2-0p6", "HET0p6-1p0"]
    if arm.startswith("SHD"):
        return ["SHD0p003", "SHD0p03"]
    return []


def main():
    stage = sys.argv[1]
    ce = load()
    d = verdicts(ce, "g3")
    if not d:
        print(""); return
    best = min(d, key=d.get)
    if stage == "best":
        v = d[best]
        print(f"{best} {v:+.4f} {'win' if v <= WIN else 'promising' if v <= PROMISING else 'loss'}"); return
    if stage == "round2":
        v = d[best]
        if v <= WIN:
            arms = family(best)
        elif v <= PROMISING:
            arms = ["SAND"] + family(best)[:2]
        else:
            arms = []
        print(" ".join(a for a in arms if ("g3", a) not in ce)); return
    if stage == "transfer":
        print(best if d[best] <= PROMISING else ""); return
    if stage == "promote":
        if d[best] > WIN:
            print(""); return
        g1 = verdicts(ce, "g1")
        ok = (best in g1 and g1[best] <= PROMISING)
        print(best if ok else ""); return
    raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
