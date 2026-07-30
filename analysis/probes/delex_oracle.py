#!/usr/bin/env python3
"""C7 -- the nonparametric token-identity oracle: how well can *anything* predict expert firing from
the current token alone?

The locus probes read a falling token AUC as evidence that routing has moved off token identity. That
reading needs a ceiling, because a linear probe on an embedding could also fail for want of capacity.
This measures the ceiling directly, with no model class at all:

  oracle AUC   score each token by the empirical rate P(y_e = 1 | token id) estimated on the fit
               split, and score AUC on held-out documents. This is the best any function of the
               current token can do, up to estimation error, so it upper-bounds every token probe.
  I/H          I(expert ; token id) / H(expert), the share of the firing variable's entropy that
               token identity explains. Unlike AUC this is calibration-free and comparable across
               experts with very different firing rates.

Token ids are not stored in the capture, but the input embedding is a fixed vector per id, so
identity is recoverable: two random projections of the embedding row agree exactly iff the rows are
bit-identical, which happens iff the token is the same. The recovered count is reported and sanity
checked against the vocabulary.

An oracle fitted on finite data overfits rare ids, which would flatter the ceiling. Ids seen fewer
than MIN_COUNT times in the fit split fall back to the expert's global rate, and the fraction of
score rows that fall back is reported so the reader can judge how binding the ceiling is.

Output: results/ablations/mechinterp_oracle.csv, one row per (run, layer, expert).
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

OUT = os.path.join(ABLATIONS, "mechinterp_oracle.csv")
MIN_COUNT = 5            # ids rarer than this in the fit split fall back to the global rate
HEADER = ["label", "run", "budget", "regime", "layer", "expert", "split", "usage_count",
          "oracle_token_AUC", "mi_over_H", "n_token_ids", "frac_score_rows_backoff"]


def token_ids(emb):
    """[S,B,H] embeddings -> [S*B] integer pseudo-ids, one per distinct embedding row."""
    S, B, H = emb.shape
    X = emb.reshape(S * B, H).astype(np.float64)
    rng = np.random.default_rng(0)
    # Two independent projections: identical rows always agree, distinct rows agree with
    # probability ~0. Cheaper and less memory-hungry than np.unique over 131k x H rows.
    key = np.stack([X @ rng.standard_normal(H) for _ in range(2)], axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def oracle(ids, Y, tr, te, nid):
    """Empirical-rate oracle per expert.

    Returns (auc[E], mi_over_H[E], frac_backoff) where the score for token t is the fit-split firing
    rate of its id, backing off to the expert's global rate for ids seen < MIN_COUNT times.
    """
    E = Y.shape[1]
    cnt = np.bincount(ids[tr], minlength=nid).astype(np.float64)
    fires = np.zeros((nid, E))
    np.add.at(fires, ids[tr], Y[tr])
    seen = cnt >= MIN_COUNT
    rate = np.zeros((nid, E))
    rate[seen] = fires[seen] / cnt[seen, None]
    glob = Y[tr].mean(0)
    rate[~seen] = glob
    scores = rate[ids[te]]                                   # [nte, E]
    auc = delex_locus.auc_batch(scores, Y[te])

    # I(y_e ; id) / H(y_e), estimated on the fit split over ids that were actually seen
    p = cnt / cnt.sum()
    q = np.where(seen[:, None], np.clip(fires / np.maximum(cnt, 1)[:, None], 1e-12, 1 - 1e-12),
                 np.clip(glob, 1e-12, 1 - 1e-12)[None, :])
    h_cond = -(p[:, None] * (q * np.log(q) + (1 - q) * np.log1p(-q))).sum(0)
    g = np.clip(glob, 1e-12, 1 - 1e-12)
    h_y = -(g * np.log(g) + (1 - g) * np.log1p(-g))
    mi_over_h = np.clip((h_y - h_cond) / np.maximum(h_y, 1e-12), 0.0, 1.0)
    frac_back = float((~seen[ids[te]]).mean())
    return auc, mi_over_h, frac_back


def main():
    import torch
    split = "position" if "--split=position" in sys.argv else "sequence"
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True)
             if (not only or r.name in only) and os.path.exists(r.path("delex_capture.pt"))]
    if not cells:
        sys.exit("no captures on disk")
    rows, summary = [], []
    for r in cells:
        d = torch.load(r.path("delex_capture.pt"), map_location="cpu", weights_only=False)
        emb = d["emb"].float().numpy()
        S, B, H = emb.shape
        ntok = S * B
        ids = token_ids(emb)
        nid = int(ids.max()) + 1
        tr, te = delex_locus.split_index(S, B, split)
        layers = registry.moe_layers(d)
        if not layers:
            print(f"[warn] {r.name}: capture holds no MoE layers, skipping (rerun its capture)")
            continue
        print(f"[run] {r.name} ({r.regime}, {r.grain_label}, {r.budget}) "
              f"{nid} distinct token ids in {ntok} tokens, layers {layers[0]}-{layers[-1]}, "
              f"split={split}", flush=True)
        for L in layers:
            Ld = d["layers"][L]
            lg = Ld["logits"].float().numpy()
            mask = Ld["mask"].numpy() if Ld["mask"] is not None else None
            Y = delex_locus.firing(lg, mask, int(Ld["k"])).reshape(ntok, lg.shape[-1]) \
                .astype(np.float64)
            auc, mih, back = oracle(ids, Y, tr, te, nid)
            usage = Y.sum(0).astype(int)
            for e in range(Y.shape[1]):
                rows.append([r.name, r.name, r.budget, r.regime, L, e, split, int(usage[e]),
                             delex_locus._r(auc[e]), delex_locus._r(mih[e]), nid, round(back, 4)])
            fin = np.isfinite(auc)
            print(f"    L{L:<3} oracle token AUC median = {np.median(auc[fin]):.4f}   "
                  f"I/H median = {np.median(mih[fin]):.4f}   backoff rows = {back*100:.1f}%",
                  flush=True)
            summary.append((r.name, L, float(np.median(auc[fin])), float(np.median(mih[fin]))))

    os.makedirs(ABLATIONS, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {OUT}: {len(rows)} rows")
    print("\nmedian oracle token AUC by layer (the ceiling any token probe is measured against):")
    runs = sorted({s[0] for s in summary})
    layers = sorted({s[1] for s in summary})
    print(f"  {'run':24} " + " ".join(f"L{l:<5}" for l in layers))
    for run in runs:
        by = {s[1]: s[2] for s in summary if s[0] == run}
        print(f"  {run:24} " + " ".join(f"{by[l]:.3f} " if l in by else "  --  " for l in layers))


if __name__ == "__main__":
    main()
