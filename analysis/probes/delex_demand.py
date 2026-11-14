#!/usr/bin/env python3
"""A10 / C6 -- demand forecastability, per MoE layer.

Causal, history-only probe: predict next-token selection membership y_e(t+1) from features available
at t -- the current gate g_e(t), demand lags y_e(t..t-3), and fast/slow EMAs of demand. No token
embeddings, no hidden states, so this measures how predictable routing demand is from its own past,
which is the property a prefetcher would exploit and the mechanism H2 leans on.

Two changes from the version that produced the published numbers:

1. **One probe per layer**, not one probe pooled over layers 2-6. Pooling discarded the layer key the
   capture already had, and the per-layer curve is the quantity C6 asks for -- directly comparable to
   the per-layer cache hit rate. Layers come from the capture, so nothing is skipped.

2. **Document-disjoint fit/score split.** The published split cut the flattened stream at 70%, which
   lands at a sequence *position*, putting every document in both halves. That matters more here than
   for the locus probes, because the features are the label's own recent history: a probe fitted on
   the first 70% of a document and scored on its last 30% can exploit that document's base rate.
   delex_locus.split_index provides both; 'sequence' is the default.

Output: results/ablations/mechinterp_demand_1e19.csv, one row per (run, layer).
"""
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delex_locus
import registry
import safe_csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

DEFAULT_OUT = os.path.join(ABLATIONS, "mechinterp_demand_1e19.csv")


def out_path():
    """Destination CSV; pass --out=PATH to write elsewhere.

    Third of the three producers whose default name encodes a scope rather than a measurement. The
    other two were repointed from an unsuffixed name to the 1e19 one, silently orphaning files that
    are now the only record of eight runs and cannot be regenerated. Nothing has been orphaned here
    yet; the flag is what keeps it that way, since the temptation arrives with the next budget.

    The name says 1e19 but the file spans every budget from 1e16 up; the suffix is historical.
    """
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            return a.split("=", 1)[1]
    return DEFAULT_OUT
BF, BS = 0.5, 0.9        # fast / slow EMA decay on demand
MAX_FIT = 400_000        # cap on fit rows per layer; pooled over experts this is already large
FEATS = ["gate", "y_t", "y_lag1", "y_lag2", "y_lag3", "ema_fast", "ema_slow"]
HEADER = ["label", "run", "budget", "regime", "layer", "split", "demand_AUC", "n_score_rows",
          "n_fit_rows", "base_rate"]


def _ema(y, beta):
    """Causal EMA along time within each sequence: e_t = beta*e_{t-1} + (1-beta)*y_t."""
    out = np.empty_like(y)
    acc = np.zeros_like(y[0])
    for t in range(y.shape[0]):
        acc = beta * acc + (1 - beta) * y[t]
        out[t] = acc
    return out


def features(lg, mask, k):
    """-> X [S,B,E,7], target [S,B,E] (next-token firing), valid [S] (last position dropped)."""
    S, B, E = lg.shape
    z = lg - lg.max(-1, keepdims=True)
    g = np.exp(z)
    g /= g.sum(-1, keepdims=True)
    y = delex_locus.firing(lg, mask, k).astype(np.float32)
    lag = lambda n: np.concatenate([np.zeros((n, B, E), np.float32), y[:S - n]], axis=0)
    X = np.stack([g.astype(np.float32), y, lag(1), lag(2), lag(3), _ema(y, BF), _ema(y, BS)], -1)
    tgt = np.concatenate([y[1:], np.zeros((1, B, E), np.float32)], axis=0)
    valid = np.ones(S, bool)
    valid[S - 1] = False                       # no next token for the final position
    return X, tgt, valid


def main():
    split = "position" if "--split=position" in sys.argv else "sequence"
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True)
             if (not only or r.name in only) and os.path.exists(r.path("delex_capture.pt"))]
    if not cells:
        sys.exit("no captures on disk")
    import torch

    rows = []
    rng = np.random.default_rng(0)
    for r in cells:
        d = torch.load(r.path("delex_capture.pt"), map_location="cpu", weights_only=False)
        layers = registry.moe_layers(d)
        if not layers:
            print(f"[warn] {r.name}: capture holds no MoE layers, skipping (rerun its capture)")
            continue
        print(f"[run] {r.name} ({r.regime}, {r.grain_label}, {r.budget}) "
              f"layers {layers[0]}-{layers[-1]}, split={split}", flush=True)
        for L in layers:
            Ld = d["layers"][L]
            lg = Ld["logits"].float().numpy()
            mask = Ld["mask"].numpy() if Ld["mask"] is not None else None
            X, tgt, valid = features(lg, mask, int(Ld["k"]))
            S, B, E, F = X.shape
            # Split on whole sequences, then flatten (position, batch, expert) rows.
            tr_b, te_b = delex_locus.split_index(1, B, split if split == "sequence" else "position")
            if split == "position":
                nb = np.arange(B)
                tr_b, te_b = nb, nb                    # position split cuts time, not sequences
                tcut = int(0.7 * S)
                trm = valid.copy(); trm[tcut:] = False
                tem = valid.copy(); tem[:tcut] = False
                Xtr, ytr = X[trm].reshape(-1, F), tgt[trm].reshape(-1)
                Xte, yte = X[tem].reshape(-1, F), tgt[tem].reshape(-1)
            else:
                Xtr = X[valid][:, tr_b].reshape(-1, F); ytr = tgt[valid][:, tr_b].reshape(-1)
                Xte = X[valid][:, te_b].reshape(-1, F); yte = tgt[valid][:, te_b].reshape(-1)
            idx = rng.permutation(len(ytr))[:MAX_FIT]
            clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr[idx], ytr[idx])
            auc = roc_auc_score(yte, clf.decision_function(Xte))
            rows.append([r.name, r.name, r.budget, r.regime, L, split, round(auc, 4),
                         len(yte), len(idx), round(float(yte.mean()), 4)])
            print(f"    L{L:<3} demand AUC = {auc:.4f}   (score rows {len(yte)}, "
                  f"base rate {yte.mean():.3f})", flush=True)

    out = out_path()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    safe_csv.guard(out, rows, key_index=HEADER.index("run") if "run" in HEADER else None)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {out}: {len(rows)} rows")
    print("\nper-layer demand AUC (higher = demand more predictable from its own history):")
    runs = sorted({x[1] for x in rows})
    layers = sorted({x[4] for x in rows})
    print(f"  {'run':24} " + " ".join(f"L{l:<5}" for l in layers))
    for run in runs:
        by = {x[4]: x[6] for x in rows if x[1] == run}
        print(f"  {run:24} " + " ".join(f"{by[l]:.3f} " if l in by else "  --  " for l in layers))


if __name__ == "__main__":
    main()
