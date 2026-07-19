#!/usr/bin/env python3
"""delex-1e19 Part 1 (c): demand-prediction probe. Causal, history-only: predict next-token
selection membership y_e(t+1) from features available at t — current gate g_e(t), demand lags
y_e(t..t-3), and fast/slow EMAs of demand (no token embeddings, no hidden states). One logistic
probe per model over all experts of layers 2-6 (features per (expert,token), pooled), 70/30 split.
Reports AUC per model (the paper: temporal demand far more forecastable, ~0.85 vs ~0.64 baseline).
Output: printed AUC per model (+ appended to a small CSV for the record).
"""
import os, sys, csv, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = "/workspace/FLAME-MoE"; RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/mechinterp_demand_1e19.csv")
LAYERS = [2, 3, 4, 5, 6]
CELLS = [("moe_coarse_1e19", "moe_coarse_1e19"),
         ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
         ("temporal_fine_1e19", "temporal_fine_g3_1e19")]
BF, BS = 0.5, 0.9   # fast/slow EMA decay


def firing(lg, mask, k):
    if mask is not None:
        return mask.float()
    sel = torch.zeros_like(lg, dtype=torch.bool).scatter_(-1, lg.topk(k, dim=-1).indices, True)
    return sel.float()


def ema(y, beta):
    """causal EMA along time (dim 0) per sequence: e_t = beta*e_{t-1} + (1-beta)*y_t."""
    S = y.shape[0]; out = torch.zeros_like(y); acc = torch.zeros_like(y[0])
    for t in range(S):
        acc = beta * acc + (1 - beta) * y[t]; out[t] = acc
    return out


def build(cap):
    d = torch.load(cap, map_location="cpu", weights_only=False)
    feats, targs = [], []
    for L in LAYERS:
        if L not in d["layers"]:
            continue
        Ld = d["layers"][L]; lg = Ld["logits"].float(); k = Ld["k"]; E = lg.shape[-1]
        g = torch.softmax(lg, -1)                      # [S,B,E]
        y = firing(lg, Ld["mask"], k)                  # [S,B,E]
        S, B, _ = y.shape
        ef, es = ema(y, BF), ema(y, BS)
        # lags along time within sequence (pad start with 0)
        def lag(x, n):
            z = torch.zeros_like(x); z[n:] = x[:S - n]; return z
        F = torch.stack([g, y, lag(y, 1), lag(y, 2), lag(y, 3), ef, es], dim=-1)  # [S,B,E,7]
        tgt = torch.zeros_like(y); tgt[:S - 1] = y[1:]                             # next-token firing
        valid = torch.ones(S, dtype=torch.bool); valid[S - 1] = False             # drop last pos/seq
        F = F[valid].reshape(-1, 7); tgt = tgt[valid].reshape(-1)
        feats.append(F.numpy()); targs.append(tgt.numpy())
    X = np.concatenate(feats); yv = np.concatenate(targs)
    return X, yv


def main():
    rows = []
    for label, run in CELLS:
        cap = os.path.join(RUNS, run, "delex_capture.pt")
        if not os.path.exists(cap):
            print(f"[skip] {label}"); continue
        X, y = build(cap)
        n = len(y); cut = int(0.7 * n)
        # subsample train for speed (pooled over experts is large)
        idx = np.random.default_rng(0).permutation(cut)[:400000]
        clf = LogisticRegression(max_iter=300, C=1.0).fit(X[idx], y[idx])
        auc = roc_auc_score(y[cut:], clf.decision_function(X[cut:]))
        rows.append([label, run, round(auc, 4), int(n)])
        print(f"[ok] {label}: demand-prediction AUC = {auc:.4f}  (n={n})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "run", "demand_AUC", "n_examples"]); w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
