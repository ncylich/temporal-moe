#!/usr/bin/env python3
"""Which of the locus probes' two null controls actually floors the probe?

The locus probes calibrate against two nulls: an iid permutation of the labels, and a circular shift
of >=1000, the latter documented as "preserving the labels' residency-induced autocorrelation".
Running both at full depth over every expert (A2) shows they disagree, and not by a little:

    iid permutation      0.4996 - 0.5002   every feature, every window, every layer
    circular shift       0.5025 (token), 0.5046 / 0.5059 / 0.5121 (context at w=k/2, k, 32)

with the shift's excess growing monotonically in the context window width and with depth. Either the
models fail their floor -- in which case none of their AUCs may be reported -- or the shift is not a
valid null. This script decides, by running a battery of nulls that each preserve or destroy exactly
one structure in the label series:

    flat-roll   np.roll on the FLATTENED [S*B] stream: what the published null does. Note a shift of
                1009 is 1009/B sequence positions plus a rotation of the batch index, so it does not
                shift 1009 tokens along a sequence, and it partly scrambles which sequence a label
                series belongs to.
    seq-roll    roll along the sequence axis only: same sequence, position t -> t-16.
    perm-b      permute the batch index at each position: destroys the document association,
                preserves position within the sequence exactly.
    perm-t      permute positions within each sequence: destroys position, keeps each label series
                inside the sequence it came from.
    iid         permute everything.
    synth       labels from a two-state Markov chain matched to a real expert's rate and lag-1
                persistence, driven by fresh randomness: autocorrelated, and independent of the
                embeddings by construction.

Two candidate explanations were tested and rejected before the third fit. Generic label
autocorrelation is not the mechanism: `synth` lands at chance, and the real label series of an
unconstrained model has lag-1 autocorrelation of only 0.0023, so the autocorrelation the shift was
built to preserve is barely there. Position-within-sequence is not the mechanism either: `perm-b`
preserves position exactly and is nearly clean.

What survives is document-level association. A context feature is a moving average of embeddings
inside one document, so it is a good document descriptor, and expert e has a document-level base rate
(it serves some documents more than others). Any null that leaves a label series paired with its own
sequence keeps that rate match intact, and the probe scores above chance by predicting "this document
uses expert e a lot" -- a real effect, but not the one being measured, and not something a floor is
supposed to contain. It scales with w because a wider window is a better document descriptor.

    $PY analysis/probes/delex_null_check.py [run ...]      # default: every capture on disk

Writes results/ablations/mechinterp_null_battery.csv.
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

SHIFTS = delex_locus.SHIFTS
OUT = os.path.join(ABLATIONS, "mechinterp_null_battery.csv")
HEADER = ["run", "regime", "budget", "layer", "feature", "window", "null", "preserves",
          "median_AUC", "excess_over_chance", "n_experts", "label_autocorr_lag1"]
PRESERVES = {"real": "nothing destroyed (the measurement)", "flat-roll": "partial document + position",
             "seq-roll": "document + near-position", "perm-b": "position only",
             "perm-t": "document only", "iid": "nothing", "synth": "autocorrelation only"}


def markov_like(y, rng):
    """Two-state Markov chain matched to y's firing rate and lag-1 persistence.

    Same marginal P(fire) and same P(fire | fired at t-1) as the real series, so autocorrelation
    matches at lag 1 and decays geometrically. Uses only those two scalars plus fresh randomness, so
    it carries no information about the embeddings.
    """
    y = y.astype(bool)
    p11 = float((y[1:] & y[:-1]).sum() / max(1, y[:-1].sum()))
    p01 = float((y[1:] & ~y[:-1]).sum() / max(1, (~y[:-1]).sum()))
    out = np.zeros(len(y), bool)
    out[0] = y[0]
    u = rng.random(len(y))
    for t in range(1, len(y)):
        out[t] = u[t] < (p11 if out[t - 1] else p01)
    return out.astype(np.float64)


def autocorr1(y):
    y = np.asarray(y, np.float64)
    y = y - y.mean()
    v = (y * y).mean()
    return float((y[1:] * y[:-1]).mean() / v) if v > 0 else float("nan")


def battery(run, layer=None, min_usage=500, max_experts=256):
    # max_experts was 24, which gives the median iid null about +-0.002 of sampling noise -- the same
    # size as the 0.002 gate tolerance it is compared against. A test whose noise floor equals its
    # threshold flags healthy models at a steady rate, and it did: four of 26 fell outside the gate at
    # 24 experts and all four came back inside at 256 (deviations 0.0020-0.0025 -> 0.0001-0.0009,
    # shrinking like 1/sqrt(n) as a correctly-centred median should).
    """Run every null arm on one capture's chosen layer. Returns CSV rows + a printable table."""
    import torch
    r = registry.get(run)
    cap = r.path("delex_capture.pt")
    if not os.path.exists(cap):
        return [], None
    d = torch.load(cap, map_location="cpu", weights_only=False)
    emb = d["emb"].float().numpy()
    S, B, H = emb.shape
    ntok = S * B
    # Document-disjoint split, matching the locus probes. The old scalar 70% cut landed at a
    # sequence position, so every document appeared in both halves; _Probe now takes an index
    # pair precisely so that split is expressible.
    tr, te = delex_locus.split_index(S, B, "sequence")
    L = layer if layer is not None else registry.moe_layers(d)[0]
    Ld = d["layers"][L]
    lg = Ld["logits"].float().numpy()
    k = int(Ld["k"])
    mask = Ld["mask"].numpy() if Ld["mask"] is not None else None
    Y = delex_locus.firing(lg, mask, k).reshape(ntok, lg.shape[-1]).astype(np.float64)

    designs = [("token", 0, delex_locus._Probe(
        delex_locus._standardize(emb.reshape(ntok, H).astype(np.float64)), tr, te))]
    for vname in ("kfull", "base"):
        w = delex_locus.WINDOWS[vname](k)
        designs.append((f"context", w, delex_locus._Probe(delex_locus._standardize(
            delex_locus.context_mean(emb, w).reshape(ntok, H).astype(np.float64)), tr, te)))

    picks = [e for e in range(Y.shape[1]) if Y[:, e].sum() > min_usage][:max_experts]
    if not picks:
        return [], None
    rng = np.random.default_rng(0)
    real = Y[:, picks]
    nE = len(picks)
    synth = np.column_stack([markov_like(real[:, j], rng) for j in range(nE)])
    ac = float(np.median([autocorr1(real[:, j]) for j in range(nE)]))
    sb = lambda a: a.reshape(S, B, nE)
    flat = lambda a: a.reshape(ntok, nE)
    pb, pt, pi = rng.permutation(B), rng.permutation(S), rng.permutation(ntok)

    rows, table = [], []
    for feat, w, design in designs:
        arms = {
            "real": lambda: design.aucs(real),
            "flat-roll": lambda: np.nanmean([design.aucs(np.roll(real, s, axis=0)) for s in SHIFTS],
                                            axis=0),
            "seq-roll": lambda: design.aucs(flat(np.roll(sb(real), 16, axis=0))),
            "perm-b": lambda: design.aucs(flat(sb(real)[:, pb])),
            "perm-t": lambda: design.aucs(flat(sb(real)[pt])),
            "iid": lambda: design.aucs(real[pi]),
            "synth": lambda: design.aucs(synth),
        }
        vals = {}
        for arm, fn in arms.items():
            m = float(np.nanmedian(fn()))
            vals[arm] = m
            rows.append([run, r.regime, r.budget, L, feat, w, arm, PRESERVES[arm],
                         round(m, 4), round(m - 0.5, 4), nE, round(ac, 4)])
        table.append((f"{feat} w={w}" if feat == "context" else feat, vals))
    return rows, (run, r, L, k, S, B, ntok, len(tr), len(te), nE, ac, table)


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True)
             if (not only or r.name in only) and os.path.exists(r.path("delex_capture.pt"))]
    if not cells:
        sys.exit("no captures on disk")
    allrows = []
    arms = ["real", "flat-roll", "seq-roll", "perm-b", "perm-t", "iid", "synth"]
    for r in cells:
        rows, info = battery(r.name)
        if info is None:
            print(f"[skip] {r.name}: no sufficiently used experts")
            continue
        allrows += rows
        run, rr, L, k, S, B, ntok, n_fit, n_score, nE, ac, table = info
        print(f"\n=== {run} ({rr.regime}, {rr.grain_label}, {rr.budget}) MoE layer {L}, k={k}")
        print(f"    S={S} B={B} ({ntok} tokens; fit {n_fit}, score {n_score}), "
              f"{nE} experts with >500 firings")
        print(f"    lag-1 autocorrelation of the real label series: {ac:.4f}")
        print(f"    {'feature':14} " + " ".join(f"{a:>9}" for a in arms))
        for feat, vals in table:
            print(f"    {feat:14} " + " ".join(f"{vals[a]:9.4f}" for a in arms))
        print(f"    {'excess':14} " + " ".join(
            f"{'':>9}" if a == "real" else f"{vals[a]-0.5:+9.4f}" for a in arms)
              + "   <- last row is the widest context window")

    os.makedirs(ABLATIONS, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(allrows)
    print(f"\n[write] {OUT}: {len(allrows)} rows")
    iid = [r for r in allrows if r[6] == "iid"]
    roll = [r for r in allrows if r[6] == "flat-roll"]
    print(f"iid null:        max |excess| = {max(abs(r[9]) for r in iid):.4f} over {len(iid)} cells")
    print(f"flat-roll null:  max |excess| = {max(abs(r[9]) for r in roll):.4f} over {len(roll)} cells")
    print("\nVERDICT: the iid permutation is the null that floors the probe. The circular shift on "
          "\nthe flattened stream leaves a residual document-level association between the label "
          "\nseries and the feature, which inflates it; the effect is largest for the widest context "
          "\nwindow, i.e. exactly where the feature is the best document descriptor.")


if __name__ == "__main__":
    main()
