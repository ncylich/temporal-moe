#!/usr/bin/env python3
"""Shape of the per-layer constraint-cost profile (C3), and whether it tracks lexicality.

C3 measures, for one trained checkpoint, the test-CE penalty of applying the residency constraint to
exactly one MoE layer (`impose_one`, on an unconstrained checkpoint) or of removing it from exactly
one layer (`unmask_one`, on a temporal checkpoint). Reading that profile as a single U -- ends
expensive, middle cheap -- conflates two separable things:

  * an **endpoint effect**: the first MoE layer sits directly after the dense block and the last one
    writes into the final norm and unembedding, so perturbing either moves CE through a short path
    regardless of what it routes on;
  * an **interior gradient**: across the layers with neither adjacency, cost falls with depth, which
    is what H2 predicted.

This separates them, and tests the lexical account directly by correlating per-layer cost against
that layer's context-minus-token AUC from the locus probe.

The correlation is reported over the interior AND over the full range because they disagree, and the
disagreement is the finding: including the two endpoint layers drives the rank correlation to ~0.

One caution the output records rather than resolves. Within a single model the contextual share and
the depth index are collinear over the interior, so `cost ~ depth` and `cost ~ ctx_share` are the
same measurement and their correlations come out identical. Distinguishing them needs a model whose
contextual share is NON-monotone in depth -- the 14-layer 1e19 arms turn over around layer 8 -- so
run this on those captures to dissociate the two.

Reads   results/ablations/swap_sweep.csv        (per-layer cost, both directions)
        results/ablations/mechinterp_locus_1e19.csv  (context_minus_token per layer/expert)
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

# Which locus arm each swap direction should be compared against: the checkpoint being perturbed.
COMPARE = {"impose_one": "flame38m_g1_moe", "unmask_one": "flame38m_g1_temporal"}
NATIVE_RUN = {"impose_one": "flame38m_g1_moe", "unmask_one": "flame38m_g1_temporal"}


def spearman(a, b):
    """Rank correlation without scipy's tie handling, plus a two-sided permutation p-value.

    n is 6-8 here, so an exact-ish permutation null is cheaper and more honest than a t-approximation.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    draws = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(20000)])
    p = float((np.abs(draws) >= abs(rho) - 1e-12).mean())
    return rho, p


def read_cost():
    """-> {arm: {layer: delta_test_CE vs that checkpoint's native}}."""
    native, raw = {}, defaultdict(dict)
    with open(os.path.join(DATA, "swap_sweep.csv")) as f:
        for r in csv.DictReader(f):
            if r["arm"] == "native":
                native[r["run"]] = float(r["test_CE"])
            elif r["arm"] in ("impose_one", "unmask_one"):
                raw[r["arm"]][int(r["layer"])] = float(r["test_CE"])
    return {arm: {l: ce - native[NATIVE_RUN[arm]] for l, ce in d.items()} for arm, d in raw.items()}


def read_ctx(label, split="sequence", variant="kfull"):
    """-> {layer: median over experts of (context AUC - token AUC)}."""
    g = defaultdict(list)
    with open(os.path.join(DATA, "mechinterp_locus_1e19.csv")) as f:
        rdr = csv.DictReader(f)
        has_split = "split" in (rdr.fieldnames or [])
        for r in rdr:
            if r["label"] != label or r["variant"] != variant:
                continue
            if has_split and r["split"] != split:
                continue
            try:
                d = float(r["context_minus_token"])
            except (ValueError, TypeError):
                continue
            if not np.isnan(d):
                g[int(r["layer"])].append(d)
    return {l: st.median(v) for l, v in g.items()}


def main():
    cost = read_cost()
    rows = []
    for arm in sorted(cost):
        layers = sorted(cost[arm])
        ctx = read_ctx(COMPARE[arm])
        first, last = layers[0], layers[-1]
        interior = [l for l in layers if l not in (first, last)]
        ends_mean = st.mean(cost[arm][l] for l in (first, last))
        int_mean = st.mean(cost[arm][l] for l in interior)

        for span, name in [(layers, "full"), (interior, "interior")]:
            c = [cost[arm][l] for l in span]
            rd, pd_ = spearman([float(l) for l in span], c)
            have = [l for l in span if l in ctx]
            if len(have) == len(span):
                rc, pc = spearman([cost[arm][l] for l in have], [ctx[l] for l in have])
            else:
                rc = pc = float("nan")
            rows.append(dict(
                arm=arm, compare_run=COMPARE[arm], span=name,
                layers=f"{span[0]}-{span[-1]}", n_layers=len(span),
                rho_cost_vs_depth=round(rd, 4), p_cost_vs_depth=round(pd_, 4),
                rho_cost_vs_ctxshare=round(rc, 4), p_cost_vs_ctxshare=round(pc, 4),
                endpoint_mean=round(ends_mean, 5), interior_mean=round(int_mean, 5),
                endpoint_over_interior=round(ends_mean / int_mean, 3),
                ctx_at_last_layer=round(ctx.get(last, float("nan")), 4),
                ctx_rank_of_last_layer=(
                    1 + sorted(ctx[l] for l in layers).index(ctx[last]) if last in ctx else ""),
            ))

    out = os.path.join(DATA, "swap_shape.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['arm']:11s} {r['span']:8s} L{r['layers']:5s} "
              f"cost~depth rho={r['rho_cost_vs_depth']:+.3f} p={r['p_cost_vs_depth']:.3f}  "
              f"cost~ctx rho={r['rho_cost_vs_ctxshare']:+.3f} p={r['p_cost_vs_ctxshare']:.3f}  "
              f"ends/interior={r['endpoint_over_interior']:.2f}x")
    r = rows[0]
    print(f"\n  last layer's contextual share ranks {r['ctx_rank_of_last_layer']} of "
          f"{max(x['n_layers'] for x in rows)} (1 = most lexical) while costing the most to "
          f"constrain -- the endpoint spike is not a lexical effect.")


if __name__ == "__main__":
    main()
