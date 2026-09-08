#!/usr/bin/env python3
"""Locus + routing-stability probe for the Qwen families, free routing vs untrained R=8.

Same protocol as locus.py (OLMoE): for each expert in a band of layers at 12.5-37.5% relative
depth, ridge-probe whether its firing is predicted by the current token's embedding (token_AUC)
or by the mean embedding of the surrounding w=k=8 tokens excluding the token (context_AUC);
report the median and context_minus_token. Both cells run in one model load: 'free' is the
residency code path with every layer in the free set, 'R8' is min_logit rolling residency.

The free pass additionally measures routing temporal stability - the mechanism candidate for
why residency costs so little on fine-grained models:
  churn_1        mean |top8(t) ∩ top8(t-1)| / 8          (higher = stabler demand)
  window_cover   mean fraction of top8(t) inside the union of the previous w tokens' top8
  top8_mass      mean softmax mass on the selected 8      (gate concentration)

    locus_qwen.py --family qwen3
    locus_qwen.py --family qwen3_5
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                              # noqa: E402
import residency_qwen as RQ                                          # noqa: E402
import train_qwen as TQ                                              # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FIRE, LOGITS = {}, {}
DEFAULT_LAYERS = {"qwen3": "6,9,12,15,18", "qwen3_5": "5,8,10,13,15"}


def _hook(idx, want_logits):
    def fn(mod, inp, out):
        if isinstance(out, tuple) and len(out) == 3:
            FIRE[idx] = out[2].detach()
            if want_logits:
                LOGITS[idx] = out[0].detach().float()
    return fn


def auc(scores, labels):
    """Rank-based AUC; 0.5 when a class is absent. (Same as locus.py.)"""
    p, n = int(labels.sum()), int((~labels).sum())
    if p == 0 or n == 0:
        return 0.5
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64, device=scores.device)
    return float((ranks[labels].sum() - p * (p + 1) / 2) / (p * n))


def ridge_auc(X, y, XtX_reg, lam=1.0, split=0.7):
    """XtX_reg = X[:tr].T @ X[:tr] + lam*I, precomputed once per design matrix: the same probe
    runs for every expert, and the Gram matrix is the expensive part."""
    n = X.shape[0]
    tr = int(n * split)
    Xt, yt = X[:tr], y[:tr].double()
    w = torch.linalg.solve(XtX_reg, Xt.T @ yt.to(X.dtype))
    return auc((X[tr:] @ w).double(), y[tr:])


def gram(X, lam=1.0, split=0.7):
    tr = int(X.shape[0] * split)
    Xt = X[:tr]
    return Xt.T @ Xt + lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)


def run_cell(model, emb, ids, layers, A, free):
    """One routing state over all packs: returns locus summary (+ churn stats if free)."""
    L = getattr(model.config, "text_config", model.config).num_hidden_layers
    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
    RES.set_free_layers(list(range(L)) if free else None)
    tok_f, ctx_f = [], []
    fired = {l: [] for l in layers}
    churn1, cover, mass = [], [], []
    with torch.no_grad():
        for i in range(A.packs):
            x = ids[i, :A.seqlen].unsqueeze(0).to("cuda").long()
            FIRE.clear(), LOGITS.clear()
            model(x)
            e = emb[x[0]].float()
            pad = torch.nn.functional.pad(e.T.unsqueeze(0), (A.w, A.w), mode="replicate")[0].T
            win = sum(pad[A.w + d: A.w + d + e.shape[0]] for d in range(-A.w, A.w + 1) if d != 0)
            tok_f.append(e)
            ctx_f.append(win / (2 * A.w))
            for l in layers:
                idx = FIRE[l].reshape(A.seqlen, -1)                    # [S, k]
                fired[l].append(idx.cpu())
                if free:
                    k = idx.shape[1]
                    same = (idx[1:, :, None] == idx[:-1, None, :]).any(-1).float().mean()
                    churn1.append(float(same))
                    cov = []
                    for t in range(A.w, A.seqlen):
                        u = idx[t - A.w: t].reshape(-1)
                        cov.append(float((idx[t][:, None] == u[None, :]).any(-1).float().mean()))
                    cover.append(sum(cov) / len(cov))
                    if l in LOGITS:
                        p = torch.softmax(LOGITS[l].reshape(A.seqlen, -1), -1)
                        mass.append(float(p.gather(-1, idx.to(p.device)).sum(-1).mean()))
    X_tok = torch.cat(tok_f).cuda()
    X_ctx = torch.cat(ctx_f).cuda()
    G_tok, G_ctx = gram(X_tok), gram(X_ctx)
    t_auc, c_auc = [], []
    E = getattr(model.config, "text_config", model.config).num_experts
    for l in layers:
        f = torch.cat(fired[l])                                        # [N, k]
        onehot = torch.zeros(f.shape[0], E, dtype=torch.bool)
        onehot.scatter_(1, f.long(), True)
        for e_i in range(E):
            y = onehot[:, e_i].cuda()
            frac = float(y.float().mean())
            if frac < 0.01 or frac > 0.99:
                continue
            t_auc.append(ridge_auc(X_tok, y, G_tok))
            c_auc.append(ridge_auc(X_ctx, y, G_ctx))
    t_med = float(torch.tensor(t_auc).median()) if t_auc else float("nan")
    c_med = float(torch.tensor(c_auc).median()) if c_auc else float("nan")
    out = {"n_experts_probed": len(t_auc), "token_AUC_median": round(t_med, 6),
           "context_AUC_median": round(c_med, 6),
           "context_minus_token_median": round(c_med - t_med, 6)}
    if free:
        out.update(churn1=round(sum(churn1) / len(churn1), 4),
                   window_cover=round(sum(cover) / len(cover), 4),
                   top8_mass=round(sum(mass) / max(1, len(mass)), 4) if mass else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("qwen3", "qwen3_5"))
    ap.add_argument("--packs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--w", type=int, default=8)
    A = ap.parse_args()
    layers = [int(x) for x in (A.layers or DEFAULT_LAYERS[A.family]).split(",")]

    FAM = TQ.resolve(A.family)
    model, _ = RQ.load_model(path=FAM["model"], family=A.family)
    RQ.tag_layers(model)
    routers = [m for m in model.modules() if "TopKRouter" in type(m).__name__]
    for i, r in enumerate(routers):
        if i in layers:
            r.register_forward_hook(_hook(i, want_logits=True))
    model.eval()
    base = getattr(model, "model", model)
    emb = base.embed_tokens.weight.detach() if hasattr(base, "embed_tokens") \
        else base.model.embed_tokens.weight.detach()

    ids = torch.load(f"{FAM['data']}/bpb_slice_ids_{FAM['suffix']}.pt", weights_only=False)

    rows = []
    for cell, free in (("free", True), ("R8", False)):
        summ = run_cell(model, emb, ids, layers, A, free)
        summ.update(tag=f"locus_{A.family}_{cell}", packs=A.packs, seqlen=A.seqlen,
                    window_w=A.w, layers=",".join(map(str, layers)))
        print(f"[locus] {json.dumps(summ, indent=1)}", flush=True)
        rows.append(summ)

    path = os.path.join(ABLATIONS, "locus_qwen.csv")
    keys = ["tag", "n_experts_probed", "packs", "seqlen", "window_w", "layers",
            "token_AUC_median", "context_AUC_median", "context_minus_token_median",
            "churn1", "window_cover", "top8_mass"]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
