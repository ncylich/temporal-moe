#!/usr/bin/env python3
"""C10 -- cross-layer probe transfer: is routing the same function of the embedding at every depth,
or a qualitatively different one?

C10 was specified as "fit the token probe at layer l, evaluate at l'", but that form is ill-posed:
expert index e at layer l has no relationship to expert e at layer l', so there is no label to
evaluate a transferred probe against. What is well-posed, and answers the same question, is to compare
the *subspaces* the probes occupy. Each layer's token probe is a matrix W_l of shape [H, E] -- one
weight vector per expert -- and its column space is the set of embedding directions that layer's
routing is sensitive to. If deep routing is the same function weakening, those subspaces coincide; if
it is a different function, they separate.

Statistic: mean squared canonical correlation between the column spaces of W_l and W_l', i.e.
||Q_l^T Q_l'||_F^2 / r for orthonormal bases Q and r = min(rank_l, rank_l'). It is 1.0 when the
subspaces coincide and equals the *chance* level when they are unrelated.

**Chance is not zero and it is not small.** Two random r-dimensional subspaces of R^H overlap by about
r/H on this statistic, which is 64/800 = 0.08 for the coarse 1e19 model but 64/256 = 0.25 for the
coarse 1e18 one -- so an overlap of 0.3 means opposite things at the two scales. The chance level is
computed both analytically and by drawing random Gaussian subspaces of matched dimension, and every
number below is reported against it.

Output: results/ablations/mechinterp_transfer.csv, one row per (run, layer_from, layer_to).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delex_locus
import registry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

OUT = os.path.join(ABLATIONS, "mechinterp_transfer.csv")
HEADER = ["label", "run", "budget", "regime", "H", "E", "split", "layer_from", "layer_to",
          "rank", "subspace_overlap", "chance_overlap", "overlap_over_chance"]


def basis(W, tol=1e-10):
    """Orthonormal basis for the column space of W [H, E], dropping numerically null directions."""
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, s > tol * max(1.0, s[0])]


def overlap(Qa, Qb):
    """Mean squared canonical correlation: 1.0 if the subspaces coincide, ~r/H if unrelated."""
    r = min(Qa.shape[1], Qb.shape[1])
    if r == 0:
        return float("nan")
    return float((np.linalg.svd(Qa.T @ Qb, compute_uv=False) ** 2).sum() / r)


def chance_overlap(H, ra, rb, rng, trials=8):
    """Empirical chance level for two random subspaces of the same dimensions."""
    vals = []
    for _ in range(trials):
        vals.append(overlap(basis(rng.standard_normal((H, ra))), basis(rng.standard_normal((H, rb)))))
    return float(np.mean(vals))


def main():
    import torch
    split = "position" if "--split=position" in sys.argv else "sequence"
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True)
             if (not only or r.name in only) and os.path.exists(r.path("delex_capture.pt"))]
    if not cells:
        sys.exit("no captures on disk")
    rng = np.random.default_rng(0)
    rows = []
    for r in cells:
        d = torch.load(r.path("delex_capture.pt"), map_location="cpu", weights_only=False)
        emb = d["emb"].float().numpy()
        S, B, H = emb.shape
        ntok = S * B
        tr, te = delex_locus.split_index(S, B, split)
        design = delex_locus._Probe(
            delex_locus._standardize(emb.reshape(ntok, H).astype(np.float64)), tr, te)
        layers = registry.moe_layers(d)
        if not layers:
            print(f"[warn] {r.name}: capture holds no MoE layers, skipping (rerun its capture)")
            continue
        Q = {}
        E = None
        for L in layers:
            Ld = d["layers"][L]
            lg = Ld["logits"].float().numpy()
            mask = Ld["mask"].numpy() if Ld["mask"] is not None else None
            Y = delex_locus.firing(lg, mask, int(Ld["k"])).reshape(ntok, lg.shape[-1]) \
                .astype(np.float64)
            E = Y.shape[1]
            Ytr = Y[tr]
            Q[L] = basis(design.Ainv @ (design.Xtr.T @ (Ytr - Ytr.mean(0))))
        ch = chance_overlap(H, Q[layers[0]].shape[1], Q[layers[-1]].shape[1], rng)
        print(f"[run] {r.name} ({r.regime}, {r.grain_label}, {r.budget}) H={H} E={E}, "
              f"probe subspace rank {Q[layers[0]].shape[1]}, chance overlap {ch:.4f} "
              f"(analytic r/H = {Q[layers[0]].shape[1]/H:.4f})", flush=True)
        for a in layers:
            for b in layers:
                o = overlap(Q[a], Q[b])
                rows.append([r.name, r.name, r.budget, r.regime, H, E, split, a, b,
                             Q[a].shape[1], round(o, 4), round(ch, 4),
                             round(o / ch, 3) if ch else ""])
        # adjacent vs most-distant, the comparison that answers the question
        adj = np.mean([overlap(Q[a], Q[b]) for a, b in zip(layers, layers[1:])])
        far = overlap(Q[layers[0]], Q[layers[-1]])
        print(f"      adjacent layers {adj:.4f} ({adj/ch:.2f}x chance) | "
              f"L{layers[0]} vs L{layers[-1]} {far:.4f} ({far/ch:.2f}x chance)", flush=True)

    os.makedirs(ABLATIONS, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {OUT}: {len(rows)} rows")
    print("\nOff-diagonal collapse toward the chance column would mean routing is a qualitatively\n"
          "different function of the embedding with depth, rather than the same one weakening.")


if __name__ == "__main__":
    main()
