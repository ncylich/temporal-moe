#!/usr/bin/env python3
"""Decision rules of results/ablations/CURRICULUM_PLAN.md, applied to curriculum_1e17.csv so the
supervisor can pick the next stage unattended. Reference = the C0 control of the same grain (full MoE through the router path on the
same pythia-50k corpus; the recorded 16k-tokenizer 1e17 cells are not comparable), so nothing is
decided until C0 exists. Win bar 0.010 CE below the reference, promising within 0.005 above it.

    curriculum_decide.py round2   -> arms to run next on grain 3 (empty = stop)
    curriculum_decide.py best     -> the best grain-3 arm (by CE) and its verdict
    curriculum_decide.py transfer -> 'C0 <best>' to run on grain 1 if the best is a win or promising
    curriculum_decide.py promote  -> the arm to promote to 1e18 if it won on grain 3 and did not lose on grain 1
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

CSV = os.path.join(ABLATIONS, "curriculum_1e17.csv")
WIN, PROMISING = -0.010, 0.005


def load():
    rows = [r for r in csv.DictReader(l for l in open(CSV) if not l.startswith("#"))]
    return {(r["grain"], r["arm"]): float(r["test_CE"]) for r in rows}


def verdicts(ce, grain):
    r = ce.get((grain, "C0"))
    if r is None:
        return {}
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
            # every arm lost: two diagnostics before concluding, a switch late in the cosine tail
            # (the hard-switch damage was learning-rate gated) and one refinement of the best family
            arms = ["SW0p8"] + family(best)[:1]
        print(" ".join(a for a in arms if ("g3", a) not in ce)); return
    if stage == "transfer":
        # grain 1 needs its own C0 reference; the driver skips arms that already exist
        print(f"C0 {best}" if d[best] <= PROMISING else ""); return
    if stage == "promote":
        if d[best] > WIN:
            print(""); return
        g1 = verdicts(ce, "g1")            # needs a grain-1 C0 as well
        ok = (best in g1 and g1[best] <= PROMISING)
        print(best if ok else ""); return
    raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
