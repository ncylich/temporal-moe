#!/usr/bin/env python3
"""Freed layers are regularised by a different aux loss, and it dilutes the ladder.

`aux_z_from_router_logits` has two branches. A constrained layer gets the Switch load-balancing loss
on the residency-masked distribution, `E * sum(f * P)` with `f` the resident fraction. A freed layer
has no mask, so `f` is all ones and `E * sum(1 * P)` collapses to the constant `E` with no gradient;
the code substitutes the importance loss `E * sum(P^2)` instead. That substitution is reasonable --
there is no load term to use -- but the two land on different scales. At the uniform optimum the
first is `k` and the second is `1`, so they differ by roughly the top-k factor by construction.

Measured on the adapted OLMoE with the headline free set {0,1,14,15}: freed layers average 1.4533
and constrained layers 36.1976, a factor of 24.9. The four freed layers contribute 1.3% of the aux
while being 25% of the layers.

The consequence is not the gap itself but where it lands. The returned aux is the mean over ALL
layers, so freed layers dilute it, and the dilution grows with the size of the free set -- which is
the axis the layer-freeing ladder varies:

    free set        freed   effective aux   vs full residency
    none              0         36.20              --
    {0,1}             2         31.9             -12%
    {0,1,15}          3         29.7             -18%
    {0,1,14,15}       4         27.5             -24%

So each rung of that ladder trains under a weaker effective load-balancing strength than the one
below it, monotonically, and the rung with the best BPB is the least regularised. That does not show
the aux dilution explains the result -- the ladder spans 0.028 BPB and this is untested against it --
but it means free-set size and regularisation strength are confounded in the cells that exist.

The control is one 50M run of {0,1,14,15} with AUX_C scaled by 36.20/27.5 = 1.32, so its constrained
layers see the same pressure as a full-residency cell. Not run.

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
    from temporal.temporal_router import compute_resident_mask_accel as scan
    B, S = ids.shape
    RES.enable_residency(R=8)
    RES.set_free_layers(sorted(free) if free else None)
    out = model(ids, output_router_logits=True)
    freed, con = [], []
    for li, rl in enumerate(out.router_logits):
        N, E = rl.shape
        if li in free:
            P = torch.softmax(rl.float(), -1).mean(0)
            freed.append((E * (P * P).sum()).item())
        else:
            lg = rl.view(B, S, E).transpose(0, 1).contiguous()
            with torch.no_grad():
                m = scan(lg.float(), 8, evict="min_logit").transpose(0, 1).reshape(N, E)
            u = rl.masked_fill(~m, float("-inf")).float()
            f = torch.isfinite(u).float().mean(0)
            con.append((E * (f * torch.softmax(u, -1).mean(0)).sum()).item())
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
        w.writerow([f"# Aux-loss dilution by free-set size. Freed layers use E*sum(P^2), constrained "
                    f"layers E*sum(f*P); the returned aux is the mean over all layers, so freed "
                    f"layers dilute it. Produced by analysis/ple/aux_dilution.py."])
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {A.out}: {len(rows)} row(s)", flush=True)


if __name__ == "__main__":
    main()
