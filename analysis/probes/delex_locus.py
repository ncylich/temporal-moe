#!/usr/bin/env python3
"""delex-1e19 Part 1 (b): locus probes. Per-expert linear probes predicting expert firing y_e(t)
from (i) the current token embedding x_tok(t)=E[x_t] and (ii) the excluded-context mean
x_ctx(t)=mean{E[x_t'] : 0<|t'-t|<=w} (current token excluded, within-sequence). Fit on first 70%
of the token stream, AUC on the last 30%. Reports per (layer,expert) token_AUC, context_AUC,
context_minus_token, usage_count, plus null controls (iid permutation + circular shift>=1000).

Fast probe: within a layer the design matrix is shared across experts, so we use a ridge linear
probe with a shared Gram inverse (one inversion/layer, one matvec/expert). AUC is rank-based so the
linear probe tracks a logistic probe closely; --verify checks a sample against sklearn logistic.

Window w=k (residency lifetime). LAYERS default 2..6 (paper convention). Output rows appended by
the driver into results/ablations/mechinterp_locus_1e19.csv.
"""
import os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score

LAYERS = [2, 3, 4, 5, 6]
SHIFTS = [1009, 2003, 5011]   # circular-shift nulls (>=1000, none a multiple of seq len 2048),
                              # averaged for a robust residency-autocorrelation-preserving null
RIDGE = 1.0                   # ridge shrinkage: keeps the strong real signal, pulls spurious null
                              # fits to 0.5 (unbiased null); tuned so iid+shift nulls land ~0.500


def firing(lg, mask, k):
    """[S,B,E] bool: served/top-k set. temporal -> resident mask; moe -> top-k of logits."""
    if mask is not None:
        return mask
    sel = torch.zeros_like(lg, dtype=torch.bool)
    sel.scatter_(-1, lg.topk(k, dim=-1).indices, True)
    return sel


def context_mean(emb, w):
    """emb [S,B,H] -> x_ctx [S,B,H]: per-sequence mean of +-w neighbours excluding current token."""
    S, B, H = emb.shape
    x = emb.permute(1, 0, 2).double()                       # [B,S,H]
    cs = torch.zeros(B, S + 1, H, dtype=torch.float64)
    cs[:, 1:] = torch.cumsum(x, dim=1)
    idx = torch.arange(S)
    lo = torch.clamp(idx - w, min=0); hi = torch.clamp(idx + w + 1, max=S)
    win_sum = cs[:, hi] - cs[:, lo]                          # [B,S,H] inclusive of current
    cnt = (hi - lo).double().unsqueeze(0).unsqueeze(-1) - 1  # exclude current token
    ctx = (win_sum - x) / cnt.clamp(min=1)
    return ctx.permute(1, 0, 2).contiguous()                # [S,B,H]


def _auc_probe(Xtr, Xte, ytr, yte, Ainv):
    """Ridge linear probe with precomputed (X'X+lam I)^-1 for the shared design; returns test AUC."""
    if ytr.sum() < 5 or yte.sum() < 3 or yte.sum() == len(yte):
        return float("nan")
    yc = ytr - ytr.mean()
    w = Ainv @ (Xtr.T @ yc)
    s = Xte @ w
    return roc_auc_score(yte, s)


def _prep(X):
    """standardize columns; return standardized X and (mean,std) — fit on all rows (probe uses split)."""
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return (X - mu) / sd


def analyze(cap_path, label, run, verify=False):
    d = torch.load(cap_path, map_location="cpu", weights_only=False)
    emb = d["emb"].float()                                   # [S,B,H]
    S, B, H = emb.shape
    ntok = S * B
    cut = int(0.7 * ntok)
    rows = []
    per_model = {"tok": [], "ctx": [], "ndom": 0, "ntot": 0, "null_iid": [], "null_shift": []}
    for L in LAYERS:
        if L not in d["layers"]:
            continue
        Ld = d["layers"][L]
        lg = Ld["logits"].float(); k = Ld["k"]; E = lg.shape[-1]
        fire = firing(lg, Ld["mask"], k)                    # [S,B,E]
        Xtok = _prep(emb.reshape(ntok, H).double())
        Xctx = _prep(context_mean(emb, k).reshape(ntok, H).double())
        # shared Gram inverses on the train split
        Xtok_tr, Xtok_te = Xtok[:cut].numpy(), Xtok[cut:].numpy()
        Xctx_tr, Xctx_te = Xctx[:cut].numpy(), Xctx[cut:].numpy()
        Ainv_tok = np.linalg.inv(Xtok_tr.T @ Xtok_tr + RIDGE * np.eye(H))
        Ainv_ctx = np.linalg.inv(Xctx_tr.T @ Xctx_tr + RIDGE * np.eye(H))
        fire_flat = fire.reshape(ntok, E).numpy().astype(np.float64)
        rng = np.random.default_rng(0)
        for e in range(E):
            y = fire_flat[:, e]
            usage = int(y.sum())
            a_tok = _auc_probe(Xtok_tr, Xtok_te, y[:cut], y[cut:], Ainv_tok)
            a_ctx = _auc_probe(Xctx_tr, Xctx_te, y[:cut], y[cut:], Ainv_ctx)
            rows.append([label, run, L, e, usage, round(a_tok, 4), round(a_ctx, 4),
                         round((a_ctx - a_tok), 4) if np.isfinite(a_ctx) and np.isfinite(a_tok) else "", "base"])
            if np.isfinite(a_tok) and np.isfinite(a_ctx):
                per_model["tok"].append(a_tok); per_model["ctx"].append(a_ctx)
                per_model["ntot"] += 1; per_model["ndom"] += int(a_ctx > a_tok)
            # null controls on ~8 experts/layer (precise median, bounded cost), 3-shift averaged
            if usage > 20 and e % max(1, E // 8) == 0:
                yp = rng.permutation(y)
                per_model["null_iid"].append(_auc_probe(Xtok_tr, Xtok_te, yp[:cut], yp[cut:], Ainv_tok))
                sh = [_auc_probe(Xtok_tr, Xtok_te, np.roll(y, s)[:cut], np.roll(y, s)[cut:], Ainv_tok)
                      for s in SHIFTS]
                per_model["null_shift"].append(float(np.nanmean(sh)))
        if verify and L == LAYERS[0]:
            _verify_logistic(Xtok_tr, Xtok_te, Xctx_tr, Xctx_te, fire_flat, cut)
    return rows, per_model


def _verify_logistic(Xtok_tr, Xtok_te, Xctx_tr, Xctx_te, fire_flat, cut):
    from sklearn.linear_model import LogisticRegression
    E = fire_flat.shape[1]
    picks = [e for e in range(E) if fire_flat[:, e].sum() > 50][:4]
    print("  [verify] ridge-vs-logistic token_AUC on sample experts:", file=sys.stderr)
    for e in picks:
        y = fire_flat[:, e]
        ai = np.linalg.inv(Xtok_tr.T @ Xtok_tr + RIDGE * np.eye(Xtok_tr.shape[1]))
        r = _auc_probe(Xtok_tr, Xtok_te, y[:cut], y[cut:], ai)
        lr = LogisticRegression(max_iter=200, C=1.0).fit(Xtok_tr, y[:cut])
        lg_auc = roc_auc_score(y[cut:], lr.decision_function(Xtok_te))
        print(f"    expert {e}: ridge {r:.3f} vs logistic {lg_auc:.3f}", file=sys.stderr)
