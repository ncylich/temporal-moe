#!/usr/bin/env python3
"""Aux-loss strength across the free-set ladder: the confound this found, and its repair.

BEFORE (two branches). A constrained layer got `E*sum(f*P)` with `f` the resident fraction; a freed
layer got the importance loss `E*sum(P^2)`, on the reasoning that with no mask there is no load term.
At the uniform optimum those are `k` and `1`, so freed layers were regularised ~k times more weakly.
Because the returned aux is the mean over ALL layers, freed layers diluted it in proportion to the
size of the free set -- the very axis the ladder varies:

    free set      effective aux    vs full residency
    none              33.86              --
    {0,1}             31.68            -6.4%
    {0,1,2}           30.41           -10.2%
    {0,1,15}          29.43           -13.1%
    {0,1,14,15}       27.46           -18.9%

Monotone, with the best-BPB rung the least regularised: free-set size and regularisation strength
were confounded in every cell. Note {0,1,2} and {0,1,15} free the same NUMBER of layers and still
differ by 2.9 points, so it was never a function of count alone.

AFTER (one branch, matching temporal-moe). FLAME does not branch: `temporal_forward` masks to -inf
and calls the unmodified `routing()`, so residency changes the distribution and never the loss.
`aux_z_from_router_logits` now does the same -- `E*sum(f*P)` everywhere, `f` the dispatch fraction
from top-k of whichever distribution that layer sees. A freed layer is then exactly HF's
`load_balancing_loss_func`, asserted to 1.9e-06 by `checks.py auxparity`.

    free set      effective aux    vs full residency
    none              33.86              --
    {0,1}             34.24            +1.1%
    {0,1,2}           34.29            +1.3%
    {0,1,15}          34.24            +1.1%
    {0,1,14,15}       34.24            +1.1%

The confound is gone. What remains is that every layer-freeing cell on disk was TRAINED under the
old formula, so they are not comparable to anything produced after this change.

    aux_dilution.py --free-set 0,1,14,15
    aux_dilution.py --ladder            # every free set in the layer-freeing ladder
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                       # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                   # noqa: E402

OUT = os.path.join(ABLATIONS, "aux_dilution.csv")
LADDER = ["", "0,1", "0,1,2", "0,1,15", "0,1,14,15"]
HEADER = ["free_set", "n_freed", "n_constrained", "aux_freed_mean", "aux_constrained_mean",
          "scale_gap", "effective_aux", "pct_vs_full_residency", "freed_share_of_aux_pct"]


def measure(model, ids, free):
    """Per-layer aux from the LIVE function, not a copy of it.

    This used to reimplement both branches inline, which meant it measured the formulas as they were
    when it was written rather than as they are. It reported the old dilution unchanged after the
    branches were unified. Calling aux_z_from_router_logits one layer at a time costs a little more
    and cannot drift from the code it exists to characterise.
    """
    B, S = ids.shape
    RES.enable_residency(R=8)
    RES.set_free_layers(sorted(free) if free else None)
    out = model(ids, output_router_logits=True)
    freed, con = [], []
    for li, rl in enumerate(out.router_logits):
        # One layer at a time, with the free set rewritten so this layer keeps its real role.
        RES.set_free_layers([li] if li in free else [])
        a, _ = RES.aux_z_from_router_logits((rl,), B, S, 8)
        (freed if li in free else con).append(a.item())
    RES.set_free_layers(sorted(free) if free else None)
    return freed, con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-set", default=None)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()
    sets = LADDER if A.ladder else [A.free_set or ""]

    model, tok = RES.load_model()
    torch.manual_seed(0)
    ids = torch.randint(0, 50000, (2, A.seq), device="cuda")

    base = None
    rows = []
    for spec in sets:
        free = {int(x) for x in spec.split(",") if x.strip()}
        freed, con = measure(model, ids, free)
        n = len(freed) + len(con)
        eff = (sum(freed) + sum(con)) / n
        if base is None:
            base = eff                                        # the empty free set is the reference
        fm = sum(freed) / len(freed) if freed else float("nan")
        cm = sum(con) / len(con)
        rows.append([spec or "none", len(freed), len(con), f"{fm:.4f}", f"{cm:.4f}",
                     f"{cm / fm:.2f}" if freed else "", f"{eff:.4f}",
                     f"{100 * (eff - base) / base:+.1f}",
                     f"{100 * sum(freed) / (sum(freed) + sum(con)):.1f}" if freed else "0.0"])
        print(f"  {spec or 'none':12} freed {len(freed):>2}  aux_freed {fm:>8.4f}  "
              f"aux_con {cm:>8.4f}  effective {eff:>8.4f}  {100 * (eff - base) / base:>+6.1f}%",
              flush=True)

    with open(A.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["# Aux-loss strength by free-set size, measured by calling the live "
                    "aux_z_from_router_logits one layer at a time. Every layer now uses E*sum(f*P) "
                    "with f the dispatch fraction, so freed and constrained layers are on one scale "
                    "and pct_vs_full_residency should sit near zero. It did NOT before: freed layers "
                    "used the importance loss E*sum(P^2), ~k times weaker, diluting the all-layer "
                    "mean monotonically in free-set size (-6.4% at {0,1} to -18.9% at {0,1,14,15}) "
                    "along the exact axis the ladder varies. Producer: analysis/ple/aux_dilution.py"])
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {A.out}: {len(rows)} row(s)", flush=True)


if __name__ == "__main__":
    main()
