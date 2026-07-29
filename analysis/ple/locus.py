#!/usr/bin/env python3
"""Locus probe (PLE_PLAN.md §8 item 1): is expert firing predicted by the token, or by its context?

Follows the protocol the adaptation program's hf_delex.py used, so the numbers are comparable to
`olmoe_adapt_forensics.csv`:

  For each expert e in layers 2-6, predict its firing y_e(t) from
      token_AUC   : the current token's embedding E[x_t] alone
      context_AUC : the mean embedding of the surrounding w=k=8 tokens, EXCLUDING x_t itself
  Ridge probe, held-out split, AUC per expert, then the median across experts.

  context_minus_token is the quantity that moved under adaptation: -0.0041 under the untrained mask
  to +0.0493 after CE adaptation, i.e. the constraint pushed routing off token identity and onto
  context.

WHY IT DECIDES A CLAIM. §1's hypothesis is that PLE helps *because* it restores the token-specific
information residency removed. §8 pre-registers the test: token_AUC should RISE and
context_minus_token should move back toward zero. If BPB improves and the locus does not move, the
gain is generic capacity rather than lexical restoration, and the paper claim must be weakened.
Reporting BPB without this cannot distinguish the two.

Ridge is solved in closed form rather than by an iterative logistic fit: on this data the published
protocol used a ridge probe, and a closed form removes optimizer settings as a source of drift
between runs.
"""

import argparse, csv, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES               # noqa: E402
import ple as PLE                     # noqa: E402
from olmoe_paths import DATA_DIR      # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS           # noqa: E402

FIRE = {}


def _hook(idx):
    def fn(mod, inp, out):
        # OlmoeTopKRouter returns (router_logits, top_k_weights, top_k_index)
        if isinstance(out, tuple) and len(out) == 3:
            FIRE[idx] = out[2].detach()
    return fn


def auc(scores, labels):
    """Rank-based AUC; returns 0.5 when a class is absent."""
    p, n = int(labels.sum()), int((~labels).sum())
    if p == 0 or n == 0:
        return 0.5
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64, device=scores.device)
    return float((ranks[labels].sum() - p * (p + 1) / 2) / (p * n))


def ridge_auc(X, y, lam=1.0, split=0.7):
    """Closed-form ridge from features X to binary y, AUC on the held-out tail."""
    ntr = int(X.shape[0] * split)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr].double(), X[ntr:], y[ntr:]
    mu = Xtr.mean(0, keepdim=True)
    Xc = (Xtr - mu).double()
    A = Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=torch.float64, device=X.device)
    w = torch.linalg.solve(A, Xc.T @ (ytr - ytr.mean()))
    return auc(((Xte - mu).double() @ w), yte.bool())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--csurf", default=None)
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--packs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--layers", default="2,3,4,5,6")
    ap.add_argument("--w", type=int, default=8, help="context window = k")
    A = ap.parse_args()

    from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter
    model, _ = RES.load_model()
    if A.lora:
        RES.add_lora(model, r=A.lora, alpha=2 * A.lora)
    if A.csurf:
        ck = torch.load(A.csurf, map_location="cuda")
        tp = RES.router_params(model) + RES.norm_params(model)
        with torch.no_grad():
            for p, m in zip(tp, ck["masters"]):
                p.data.copy_(m.to("cuda").to(p.dtype))
    if A.table:
        sd = torch.load(A.table, map_location="cuda")
        rank = sd.pop("rank")
        t = PLE.install(model, rank if rank == "full" else int(rank), device="cuda")
        with torch.no_grad():
            for k, v in sd.items():
                getattr(t, k).copy_(v.to("cuda"))
    else:
        PLE.uninstall()

    routers = [m for m in model.modules() if isinstance(m, OlmoeTopKRouter)]
    for i, r in enumerate(routers):
        r.register_forward_hook(_hook(i))
    RES.enable_residency(R=8)
    model.eval()
    E = model.config.num_experts
    emb = model.model.embed_tokens.weight.detach()

    ids_all = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
    layers = [int(x) for x in A.layers.split(",")]
    tok_f, ctx_f, fired = [], [], {l: [] for l in layers}

    with torch.no_grad():
        for i in range(A.packs):
            x = ids_all[i, :A.seqlen].unsqueeze(0).to("cuda").long()
            FIRE.clear(); model(x)
            e = emb[x[0]].float()                                     # [S,H] token embeddings
            # context = mean of the w neighbours on each side, token itself excluded
            pad = torch.nn.functional.pad(e.T.unsqueeze(0), (A.w, A.w), mode="replicate")[0].T
            win = sum(pad[A.w + d: A.w + d + e.shape[0]] for d in range(-A.w, A.w + 1) if d != 0)
            ctx = win / (2 * A.w)
            tok_f.append(e.cpu()); ctx_f.append(ctx.cpu())
            for l in layers:
                idx = FIRE[l].reshape(x.shape[1], -1)                 # [S, top_k]
                oh = torch.zeros(x.shape[1], E, dtype=torch.bool)
                oh.scatter_(1, idx.cpu(), True)
                fired[l].append(oh)

    T = torch.cat(tok_f).cuda(); C = torch.cat(ctx_f).cuda()
    rows = []
    for l in layers:
        Y = torch.cat(fired[l]).cuda()
        for e_i in range(E):
            y = Y[:, e_i]
            if y.sum() < 20 or (~y).sum() < 20:
                continue
            ta, ca = ridge_auc(T, y), ridge_auc(C, y)
            rows.append({"tag": A.tag, "layer": l, "expert": e_i,
                         "token_AUC": round(ta, 6), "context_AUC": round(ca, 6),
                         "context_minus_token": round(ca - ta, 6)})
    if not rows:
        print("[locus] no experts had enough firing variation"); return
    med = lambda k: float(torch.tensor([r[k] for r in rows]).median())
    summ = {"tag": A.tag, "n_experts_probed": len(rows), "packs": A.packs, "seqlen": A.seqlen,
            "window_w": A.w, "layers": A.layers,
            "token_AUC_median": round(med("token_AUC"), 6),
            "context_AUC_median": round(med("context_AUC"), 6),
            "context_minus_token_median": round(med("context_minus_token"), 6),
            "ref_base_impose_ctx_minus_tok": -0.0041,
            "ref_CE_adapted_ctx_minus_tok": 0.0493}
    print("[locus]", json.dumps(summ, indent=1), flush=True)

    path = os.path.join(ABLATIONS, "ple_locus.csv")
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ.keys()))
        if not exists:
            w.writeheader()
        w.writerow(summ)
    print("wrote", path)


if __name__ == "__main__":
    main()
