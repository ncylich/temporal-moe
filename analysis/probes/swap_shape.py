#!/usr/bin/env python3
"""Shape of the per-layer constraint-cost profile (C3), and whether it tracks lexicality.

C3 measures, for one trained checkpoint, the test-CE penalty of applying the residency constraint to
exactly one MoE layer (`impose_one`, on an unconstrained checkpoint) or of removing it from exactly
one layer (`unmask_one`, on a temporal checkpoint). Reading that profile as a single U -- ends
expensive, middle cheap -- conflates two separable things:

  * an **endpoint effect**: the first MoE layer sits directly after the dense block and the last one
    writes into the final norm and unembedding, so perturbing either moves CE through a short path
    regardless of what it routes on;
  * an **interior gradient**: across the layers with neither adjacency, cost varies with depth.

This separates them per arm, and tests the lexical account by correlating per-layer cost against that
layer's context-minus-token AUC from the locus probe on the *same* checkpoint.

Read the interior and full-range correlations together: they disagree, and the disagreement is the
point. Two endpoint layers are enough to drive the full-range rank correlation to ~0.

**Every group is reported separately** -- one row per (run, arm, perturbation, span). An earlier
version keyed only on `arm`, which silently collapsed every run into one dictionary and let the sham
rows overwrite the real ones once `swap_sweep.csv` grew a `perturbation` column and more runs. Nothing
here may assume a fixed run list, layer range, or perturbation set; all three come from the data.

`mean_cost` is emitted so that comparisons *between* perturbations are not read as if the
perturbations were the same size. A sham twice the magnitude of the real constraint can differ in
endpoint ratio for reasons of curvature alone, so the ratio is only interpretable alongside the scale.

One caution the output records rather than resolves. Within a single model the contextual share and
the depth index may be collinear over the interior, in which case `cost ~ depth` and
`cost ~ ctx_share` are the same measurement and their correlations come out identical. Where they
diverge -- the 14-layer 1e19 arms, whose contextual share turns over around layer 8 -- the two can be
told apart.

Reads   results/ablations/swap_sweep.csv        (per-layer cost, every arm and perturbation)
        results/ablations/mechinterp_locus{,_1e19}.csv   (context_minus_token per layer/expert)
Writes  results/ablations/swap_shape.csv

    python3 analysis/probes/swap_shape.py
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT  # noqa: E402

DATA = os.path.join(ROOT, "results", "ablations")
LOCUS_FILES = ("mechinterp_locus.csv", "mechinterp_locus_1e19.csv")
PER_LAYER_ARMS = ("impose_one", "unmask_one")


def spearman(a, b, n_perm=20000):
    """Rank correlation plus a two-sided permutation p-value.

    n is small here (6-13), so a permutation null is cheaper and more honest than a t-approximation.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan"), float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    draws = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(draws) >= abs(rho) - 1e-12).mean())


def read_cost():
    """-> {(run, arm, perturbation): {layer: delta test CE}}, each against its own native.

    The native is matched on (run, perturbation) so a sham arm is scored against the sham's own
    unperturbed pass; it falls back to the run's `real` native only if no matched one exists.
    """
    native, raw = {}, defaultdict(dict)
    with open(os.path.join(DATA, "swap_sweep.csv")) as f:
        for r in csv.DictReader(f):
            pert = r.get("perturbation") or "real"
            if r["arm"] == "native":
                native[(r["run"], pert)] = float(r["test_CE"])
            elif r["arm"] in PER_LAYER_ARMS:
                raw[(r["run"], r["arm"], pert)][int(r["layer"])] = float(r["test_CE"])
    out = {}
    for (run, arm, pert), layers in raw.items():
        base = native.get((run, pert), native.get((run, "real")))
        if base is None:
            print(f"[warn] no native pass for {run}/{pert}; skipping {arm}", file=sys.stderr)
            continue
        out[(run, arm, pert)] = {l: ce - base for l, ce in layers.items()}
    return out


def read_ctx():
    """-> {run: {layer: median context_minus_token}} at w=k on held-out documents.

    Keyed by `run`, not by the display `label`, because the two differ for several cells
    (g1_tmoe_coarse_1e19 is labelled temporal_coarse_1e19, and so on).
    """
    g = defaultdict(lambda: defaultdict(list))
    for name in LOCUS_FILES:
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rdr = csv.DictReader(f)
            fields = rdr.fieldnames or []
            for r in rdr:
                if r.get("variant") != "kfull":
                    continue
                if "split" in fields and r["split"] != "sequence":
                    continue
                try:
                    d = float(r["context_minus_token"])
                except (ValueError, TypeError):
                    continue
                if not np.isnan(d):
                    g[r["run"]][int(r["layer"])].append(d)
    return {run: {l: st.median(v) for l, v in layers.items()} for run, layers in g.items()}


def main():
    cost, ctx_all = read_cost(), read_ctx()
    rows = []
    for (run, arm, pert), prof in sorted(cost.items()):
        layers = sorted(prof)
        if len(layers) < 4:
            print(f"[warn] {run}/{arm}/{pert}: only {len(layers)} layers, no interior to fit",
                  file=sys.stderr)
            continue
        ctx = ctx_all.get(run, {})
        if not ctx:
            print(f"[warn] {run}: no locus capture, contextual-share columns left blank",
                  file=sys.stderr)
        first, last = layers[0], layers[-1]
        interior = layers[1:-1]
        ends_mean = st.mean(prof[l] for l in (first, last))
        int_mean = st.mean(prof[l] for l in interior)

        for span, name in ((layers, "full"), (interior, "interior")):
            c = [prof[l] for l in span]
            rho_d, p_d = spearman([float(l) for l in span], c)
            shared = [l for l in span if l in ctx]
            if len(shared) >= 3:
                rho_c, p_c = spearman([prof[l] for l in shared], [ctx[l] for l in shared])
            else:
                rho_c = p_c = float("nan")
            rows.append(dict(
                run=run, arm=arm, perturbation=pert, span=name,
                layers=f"{span[0]}-{span[-1]}", n_layers=len(span),
                n_layers_with_ctx=len(shared),
                rho_cost_vs_depth=round(rho_d, 4), p_cost_vs_depth=round(p_d, 4),
                rho_cost_vs_ctxshare=("" if np.isnan(rho_c) else round(rho_c, 4)),
                p_cost_vs_ctxshare=("" if np.isnan(p_c) else round(p_c, 4)),
                mean_cost=round(st.mean(prof[l] for l in layers), 5),
                endpoint_mean=round(ends_mean, 5), interior_mean=round(int_mean, 5),
                endpoint_over_interior=(round(ends_mean / int_mean, 3) if int_mean else ""),
                ctx_at_last_layer=(round(ctx[last], 4) if last in ctx else ""),
                ctx_rank_of_last_layer=(
                    1 + sorted(ctx[l] for l in layers if l in ctx).index(ctx[last])
                    if last in ctx else ""),
            ))

    out = os.path.join(DATA, "swap_shape.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows over "
          f"{len({(r['run'], r['arm'], r['perturbation']) for r in rows})} arms)")
    for r in rows:
        if r["span"] != "interior":
            continue
        cc = f"{r['rho_cost_vs_ctxshare']:>6}" if r["rho_cost_vs_ctxshare"] != "" else "     -"
        print(f"  {r['run']:24s} {r['arm']:11s} {r['perturbation']:7s} L{r['layers']:6s} "
              f"cost~depth {r['rho_cost_vs_depth']:+.3f} (p={r['p_cost_vs_depth']:.3f})  "
              f"cost~ctx {cc}  ends/int {r['endpoint_over_interior']}x  "
              f"mean {r['mean_cost']:+.3f}")


if __name__ == "__main__":
    main()
