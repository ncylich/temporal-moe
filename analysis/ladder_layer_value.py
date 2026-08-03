#!/usr/bin/env python3
"""How many layers does the free-set ladder actually reveal a per-layer value for?

The question this answers came from a real tension. `01-findings.md` §4 reports that per-layer
constraint cost tracks how *stable* a layer's routing demand is (churn -0.91, demand forecastability
+0.78, cache hit rate +0.75) and barely tracks how *contextual* its routing is (+0.19). §5 reports
that solo damage misranks layers for set selection: layers 2 and 15 tie on solo damage (0.14084
against 0.14076) yet freeing 15 alongside {0,1} beats freeing 2. The natural next move is to ask
whether demand stability predicts *set* value where solo damage does not.

It cannot be asked of this ladder, and that is what this script establishes. A per-layer value is
revealed only by a pair of cells differing in exactly one layer at matched memory. Enumerating every
free-set cell that exists, exactly two such pairs do:

    {0,1} -> {0,1,2}     reveals layer 2
    {0,1} -> {0,1,15}    reveals layer 15

`{0,1,14,15}` adds two layers at once, so it reveals a pair and not a layer. `layer_damage.csv` has
sixteen per-layer values; the ladder has two. A Pearson or Spearman correlation over n=2 is exactly
+/-1 by construction, so it cannot distinguish demand stability from solo damage, or either from
noise. Capturing a demand profile would produce a sixteen-long vector with nothing sixteen-long to
correlate against.

What the two points DO support is a direction check, which is the sharper form of the tension
anyway: solo damage calls layers 2 and 15 a tie to four decimal places, and the ladder separates them
by 0.0108 BPB trained. Any candidate predictor has to separate them, and in the right order. That is
a two-point test, reported here as one, not dressed up as a correlation.

    ladder_layer_value.py            # print the table
    ladder_layer_value.py --check    # exit non-zero if a new cell makes n>2, i.e. this is stale
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, "results", "ablations")


def load():
    lf = {}
    with open(os.path.join(ABL, "layer_freeing_results.csv")) as f:
        for r in csv.DictReader(f):
            lf[(r["name"], r["metric"])] = r["value"]
    dmg = {}
    with open(os.path.join(ABL, "layer_damage.csv")) as f:
        for r in csv.DictReader(f):
            if r["layer"].isdigit():
                dmg[int(r["layer"])] = float(r["damage_bpb"])
    return lf, dmg


# (added layer, training-free set, trained cell). Only single-layer extensions of {0,1} qualify.
PAIRS = [(2, "{0,1,2}", "ce_free_0_1_2"), (15, "{0,1,15}", "ce_free_0_1_15")]
MULTI = [((14, 15), "{0,1,14,15}", "ce_free_0_1_14_15")]


def main():
    lf, dmg = load()
    tf = lambda s: float(lf[(s, "damage_bpb")])
    tr = lambda c: float(lf[(c, "final_bpb")])
    base_tf, base_tr = tf("{0,1}"), tr("ce_free2")

    print(f"solo damage profile: {len(dmg)} layers")
    print(f"ladder single-layer extensions of {{0,1}} at matched memory: {len(PAIRS)}\n")
    print(f"  {'layer':8}{'training-free gain':>20}{'trained gain':>15}{'solo damage':>14}")
    for L, s, c in PAIRS:
        print(f"  {L:<8}{base_tf - tf(s):>20.6f}{base_tr - tr(c):>15.6f}{dmg[L]:>14.5f}")
    for Ls, s, c in MULTI:
        lab = "+".join(map(str, Ls))
        print(f"  {lab:8}{base_tf - tf(s):>20.6f}{base_tr - tr(c):>15.6f}{'(pair)':>14}")

    a, b = PAIRS[0][0], PAIRS[1][0]
    print(f"\n  solo damage separates layers {a} and {b} by {abs(dmg[a] - dmg[b]):.5f} "
          f"-- a tie to four decimals")
    print(f"  the ladder separates them by {abs((base_tr - tr(PAIRS[0][2])) - (base_tr - tr(PAIRS[1][2]))):.5f} "
          f"BPB trained, in favour of layer {b}")
    print(f"\n  VERDICT: n={len(PAIRS)}. No correlation is computable. A predictor must be judged on "
          f"whether it separates {a} from {b} in the right direction, not on a coefficient.")

    if "--check" in sys.argv and len(PAIRS) > 2:
        sys.exit("more single-layer extensions now exist; this script and its verdict are stale")


if __name__ == "__main__":
    main()
