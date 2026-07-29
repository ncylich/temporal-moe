#!/usr/bin/env python3
"""Encode PLE_PLAN.md §5's ladder gate so the overnight orchestrator applies it mechanically.

    gate.py r32        -> "RUN" or "SKIP", per §5's rule for the r=32 rung
    gate.py winners    -> the rank(s) to carry into the CE cell, space-separated

§5's r=32 rule, verbatim in effect:
  SKIP r=32 if BPB degrades monotonically as rank drops, i.e. BPB(full) < BPB(512) < BPB(128) with
       each step worse by more than 2σ. Rank is binding and 32 can only be worse.
  RUN  r=32 if 128 is at least as good as the best of full and 512 (within 2σ). That covers both
       "ties all the way down" and "interior optimum, still descending".
  Anything else is neither -- §5 does not cover it, so this prints AMBIGUOUS and the orchestrator
  stops rather than guessing.

Winner rule: the best rank, plus any rank tying it within 2σ, because rank selection can shift with
surface as well as budget and a tie is an unresolved gate rather than a coin flip.
"""

import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR   # noqa: E402

TWO_SIGMA = 0.012


def cells():
    out = {}
    for p in glob.glob(os.path.join(DATA_DIR, "ple_ladder_*.json")):
        r = json.load(open(p))
        out[str(r["rank"])] = r["final_bpb"]
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "winners"
    c = cells()
    if what == "r32":
        need = ("full", "512", "128")
        if not all(k in c for k in need):
            print("AMBIGUOUS missing " + ",".join(k for k in need if k not in c)); return
        f, r512, r128 = c["full"], c["512"], c["128"]
        monotone = (r512 - f) > TWO_SIGMA and (r128 - r512) > TWO_SIGMA
        best_hi = min(f, r512)
        ties_or_better = (r128 - best_hi) < TWO_SIGMA
        if monotone:
            print("SKIP rank binds: full < 512 < 128 each by >2sigma")
        elif ties_or_better:
            print("RUN 128 is within 2sigma of the best of full/512")
        else:
            print("AMBIGUOUS neither monotone-degrading nor 128-competitive; §5 does not cover this")
    else:
        if not c:
            print(""); return
        best = min(c.values())
        win = sorted([k for k, v in c.items() if (v - best) < TWO_SIGMA],
                     key=lambda k: c[k])
        print(" ".join(win))


if __name__ == "__main__":
    main()
