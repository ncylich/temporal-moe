#!/usr/bin/env python3
"""Stage 3 (2): de-lexicalization probe battery on the HF OLMoE, CE-adapted-R8 vs base-impose-R8.
Replicates the FLAME 1e19 probes (analysis/probes/delex_*.py) for the HF model + residency:
  - SELECTIVITY: per-expert PR = 1/(N sum_t q_e(t)^2), q=renorm gate mass -> PR_median, generalist_frac
    (PR>0.5), router_entropy (mean per-token gate entropy / ln E).
  - DEMAND (history-only): predict firing y_e(t+1) from [g_e(t), y_e(t), lag1-3, ema_fast, ema_slow];
    logistic AUC, 70/30 split, layers 2-6 pooled.
  - LOCUS: per-expert firing y_e(t) from token-emb E[x_t] (token_AUC) vs context-emb mean of neighbors
    (context_AUC); ridge probe, AUC, context_minus_token. Layers 2-6.
Q: does adaptation reproduce the from-scratch de-lex signature (context>token locus, high demand
forecastability, selective experts)? Writes olmoe_adapt_forensics.csv.

Usage: hf_delex.py <n_packs> [seqlen]
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES
from temporal.temporal_router import compute_resident_mask
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SL = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
LAYERS = list(range(2, 7)); K = 8; W = 8; BF, BS = 0.5, 0.9
OUT = "/workspace/FLAME-MoE/results/ablations/olmoe_adapt_forensics.csv"
ids = torch.load("/workspace/olmoe-adapt/data/bpb_slice_ids.pt")[:N, :SL].long()


def capture(model, tok):
    """-> per layer in LAYERS: g [T,E] gate softmax (resident-masked), y [T,E] firing (resident);
       and emb [T,H] token input embeddings. Concatenated over packs (kept per-pack for lags/context)."""
    E = model.config.num_experts
    embs, gs, ys = [], {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    hook_out = {}
    h = model.model.embed_tokens.register_forward_hook(lambda m, i, o: hook_out.__setitem__("emb", o.detach()))
    RES.enable_residency(R=8)
    with torch.no_grad():
        for p in range(ids.shape[0]):
            x = ids[p:p + 1].to("cuda")
            out = model(x, output_router_logits=True)
            embs.append(hook_out["emb"][0].float().cpu())            # [S,H]
            for l in LAYERS:
                rl = out.router_logits[l].float()                    # [S,E]
                lg = rl.unsqueeze(1)                                 # [S,1,E]
                mask = compute_resident_mask(lg, K, evict="min_logit").squeeze(1)  # [S,E] bool (8 resident)
                g = torch.softmax(rl.masked_fill(~mask, float("-inf")), -1)        # resident-masked gate
                gs[l].append(g.cpu()); ys[l].append(mask.float().cpu())
    h.remove()
    return embs, gs, ys, E


def selectivity(gs, E):
    PR = []; Hs = []
    for l in LAYERS:
        g = torch.cat(gs[l], 0)                                      # [Ttot,E]
        N_ = g.shape[0]; mass = g.sum(0); q = g / mass.clamp(min=1e-12)
        pr = 1.0 / (N_ * (q * q).sum(0)).clamp(min=1e-12); PR.extend(pr.tolist())
        Hs.append(float((-(g * g.clamp(min=1e-12).log()).sum(-1)).mean() / np.log(E)))
    PR = np.array(PR)
    return float(np.median(PR)), float((PR > 0.5).mean()), float(np.mean(Hs))


def demand_auc(gs, ys):
    def ema(y, beta):
        e = torch.zeros_like(y); prev = torch.zeros(y.shape[1])
        for t in range(y.shape[0]):
            prev = beta * prev + (1 - beta) * y[t]; e[t] = prev
        return e
    def lag(y, n):
        z = torch.zeros_like(y); z[n:] = y[:-n] if n < y.shape[0] else z; return z
    X, Y = [], []
    for l in LAYERS:
        for g, y in zip(gs[l], ys[l]):                              # per pack (causal within seq)
            ef, es = ema(y, BF), ema(y, BS)
            F = torch.stack([g, y, lag(y, 1), lag(y, 2), lag(y, 3), ef, es], -1)  # [S,E,7]
            tgt = torch.zeros_like(y); tgt[:-1] = y[1:]             # y(t+1)
            X.append(F[:-1].reshape(-1, 7)); Y.append(tgt[:-1].reshape(-1))
    X = torch.cat(X).numpy(); Y = torch.cat(Y).numpy()
    n = len(Y); cut = int(0.7 * n)
    clf = LogisticRegression(max_iter=200, C=1.0).fit(X[:cut], Y[:cut])
    return float(roc_auc_score(Y[cut:], clf.predict_proba(X[cut:])[:, 1])), n


def locus_auc(embs, ys):
    # token emb E[x_t] vs context emb = mean neighbor emb within +-W (exclude self). Vectorized context
    # (cumsum) + SHARED per-(layer,feature) ridge Gram inverse (one solve/feature, one matvec/expert).
    tok_a, ctx_a = [], []
    for l in LAYERS:
        Xtok, Xctx, Y = [], [], []
        for emb, y in zip(embs, ys[l]):                            # emb [S,H], y [S,E]
            S, H = emb.shape
            cs = torch.cat([torch.zeros(1, H), emb.cumsum(0)], 0)  # [S+1,H]
            ar = torch.arange(S)
            lo = (ar - W).clamp(min=0); hi = (ar + W + 1).clamp(max=S)
            wsum = cs[hi] - cs[lo]; cnt = (hi - lo).float().unsqueeze(1)
            ctx = (wsum - emb) / (cnt - 1).clamp(min=1)            # windowed neighbor mean, self excluded
            Xtok.append(emb); Xctx.append(ctx); Y.append(y)
        Xt = torch.cat(Xtok).double().numpy(); Xc = torch.cat(Xctx).double().numpy(); Yl = torch.cat(Y).numpy()
        n = len(Yl); cut = int(0.7 * n); H = Xt.shape[1]
        for X, store in ((Xt, tok_a), (Xc, ctx_a)):
            Xtr = X[:cut]; Xte = X[cut:]
            P = np.linalg.solve(Xtr.T @ Xtr + 1e-2 * np.eye(H), Xtr.T)   # [H,cut] ridge operator
            for e in range(Yl.shape[1]):
                ytr = Yl[:cut, e]; yte = Yl[cut:, e]
                if ytr.sum() < 3 or yte.sum() < 1 or yte.sum() == len(yte):
                    continue
                store.append(roc_auc_score(yte, Xte @ (P @ ytr)))
    return float(np.mean(tok_a)), float(np.mean(ctx_a)), len(tok_a)


def run_cell(name, model, tok):
    embs, gs, ys, E = capture(model, tok)
    pr_med, gen_frac, r_ent = selectivity(gs, E)
    d_auc, d_n = demand_auc(gs, ys)
    tok_auc, ctx_auc, l_n = locus_auc(embs, ys)
    print(f"[{name}] PR_med={pr_med:.4f} generalist={gen_frac:.4f} router_H={r_ent:.4f} | "
          f"demand_AUC={d_auc:.4f} | token_AUC={tok_auc:.4f} context_AUC={ctx_auc:.4f} ctx-tok={ctx_auc-tok_auc:+.4f}", flush=True)
    return dict(cell=name, PR_median=pr_med, generalist_frac=gen_frac, router_entropy=r_ent,
                demand_AUC=d_auc, token_AUC=tok_auc, context_AUC=ctx_auc, ctx_minus_tok=ctx_auc - tok_auc,
                demand_n=d_n, locus_n=l_n)


rows = []
m, t = RES.load_model()                                             # base -> base-impose-R8
rows.append(run_cell("base_impose_R8", m, t)); del m; torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer
mc = AutoModelForCausalLM.from_pretrained("/workspace/olmoe-adapt/merged_ce_model", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
tc = AutoTokenizer.from_pretrained("/workspace/olmoe-adapt/merged_ce_model")
RES.install_patch()
rows.append(run_cell("CE_adapt_R8", mc, tc))

cols = ["cell", "PR_median", "generalist_frac", "router_entropy", "demand_AUC", "token_AUC", "context_AUC", "ctx_minus_tok", "demand_n", "locus_n"]
with open(OUT, "w") as f:
    f.write("# Stage 3 (2) de-lex probe battery, HF OLMoE. layers 2-6, %d packs x %d tok, R=8.\n" % (N, SL))
    f.write("# selectivity: PR_median (low=selective), generalist_frac (PR>0.5); demand_AUC (history-only firing forecast);\n")
    f.write("# locus: token_AUC vs context_AUC (context>token = de-lexicalized: fires on context not identity).\n")
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
print(f"[delex] wrote {OUT}", flush=True)
b, c = rows[0], rows[1]
print(f"[delex] DE-LEX SIGNATURE (CE-adapt vs base-impose): ctx-tok {b['ctx_minus_tok']:+.4f}->{c['ctx_minus_tok']:+.4f}, "
      f"demand_AUC {b['demand_AUC']:.3f}->{c['demand_AUC']:.3f}, PR_med {b['PR_median']:.3f}->{c['PR_median']:.3f}", flush=True)
