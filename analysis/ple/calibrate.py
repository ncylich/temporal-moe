#!/usr/bin/env python3
"""Build a calibrated PLE table from the untrained impose damage, for with/without comparison.

Delta is captured on the UNTRAINED base model:

    Delta[t, l] = MoE_free(x)[l] - MoE_residency(x)[l]     averaged over occurrences of token t

so it is the raw damage the residency constraint does at initialization, before the router or norm
gains have adapted to anything. That differs from PLE_PLAN.md §9, which captures against the
C-adapted model to measure the residual PLE would still have to fix. For INITIALIZING a table the
untrained difference is the right object: at step 0 the correction PLE should apply is the damage
that exists at step 0.

Sign convention: PLE is ADDED to the layer output, and the residency model's output is what needs
repairing toward the free model's, so Delta = free - residency.

Shrinkage, per §9:

    p[t, l] = sum_t(Delta) / (n_t + lambda)   =   mean_t * n_t/(n_t + lambda)

lambda is ESTIMATED as sigma^2_within / sigma^2_between from the same capture, not grid-searched.
Tokens never seen get n_t = 0 and land at exactly zero by construction, which preserves the zero
property the trained tables also satisfy.

Low-rank rungs use precision-weighted SVD, closed form L = D^-1 SVD_r(D T) with D = diag(sqrt(w_t)),
w_t = n_t/(n_t+lambda). The weights choose the basis, so frequent tokens set it and rare tokens
borrow strength instead of contributing noise. Rows are shrunk BEFORE the SVD, because unshrunk
noisy rows do not merely add noise, they distort a basis computed from all rows.

Writes calib_table_r<rank>.pt next to the corpus, plus calib_meta.json.
"""

import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "residency"))  # sibling dir (2026-08 split)
import residency as RES               # noqa: E402
from olmoe_paths import DATA_DIR      # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS           # noqa: E402

CAP = {"on": False, "buf": {}}


def _hook(idx):
    def fn(mod, inp, out):
        if CAP["on"]:
            CAP["buf"][idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=1024, help="corpus sequences to capture over")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--ranks", default="512,full")
    ap.add_argument("--resume-c", default=None,
                    help="capture Delta against a C-surface checkpoint instead of the untrained "
                         "base. Required when the table will be installed mid-run: by then the "
                         "router and norm gains have already repaired part of the impose damage, "
                         "so the untrained Delta over-corrects and a null would be uninterpretable.")
    ap.add_argument("--suffix", default="", help="tag appended to the output table filenames")
    ap.add_argument("--free-same", action="store_true",
                    help="run the FREE pass on the target surface too, instead of on the base "
                         "model. Removes the scale mismatch between an offset computed in base "
                         "scale and a surface whose norms have moved, at the cost of making the "
                         "target 'this surface with residency off' rather than the true gold "
                         "standard of base free routing.")
    A = ap.parse_args()

    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    model, _ = RES.load_model()
    model.eval()
    # Delta = MoE_free - MoE_residency. The FREE side is always the BASE model with free routing,
    # because that is the behaviour PLE is trying to restore -- it is the gold standard the whole
    # program measures against (0.6727 BPB). The RESIDENCY side is whatever surface the table will
    # sit on.
    #
    # An earlier version ran BOTH passes on the adapted weights when --resume-c was given, making
    # the target "the adapted model with residency switched off". That model's router was trained
    # under masking, so running it unmasked is not the gold standard and is not even good: the
    # resulting table, evaluated training-free on the 50M surface, scored 2.2142 BPB against 0.8779
    # for the surface alone. A correction toward the right target cannot make the model three times
    # worse; that number is what exposed the bug.
    ADAPT = None
    if A.resume_c:
        _ck = torch.load(A.resume_c, map_location="cuda")
        _tp = RES.router_params(model) + RES.norm_params(model)
        assert len(_tp) == len(_ck["masters"]), (len(_tp), len(_ck["masters"]))
        BASEW = [p.detach().clone() for p in _tp]
        ADAPT = [m.to("cuda").to(p.dtype) for m, p in zip(_ck["masters"], _tp)]
        _fs = "the SAME surface (--free-same)" if A.free_same else "the BASE model"
        print(f"[calib] Delta = free routing on {_fs}  minus  residency on the surface at "
              f"{_ck['seen']/1e6:.0f}M tokens ({os.path.basename(A.resume_c)})", flush=True)

    def _set(ws):
        if ws is None:
            return
        with torch.no_grad():
            for _p, _w in zip(RES.router_params(model) + RES.norm_params(model), ws):
                _p.data.copy_(_w)
    blocks = [m for m in model.modules() if isinstance(m, OlmoeSparseMoeBlock)]
    for i, b in enumerate(blocks):
        b.register_forward_hook(_hook(i))
    L, H = len(blocks), model.config.hidden_size
    V = model.config.vocab_size
    print(f"[calib] {L} MoE blocks, hidden {H}, vocab {V}", flush=True)

    sums = torch.zeros(V, L, H, dtype=torch.float32, device="cuda")
    counts = torch.zeros(V, dtype=torch.float32, device="cuda")
    sq = torch.zeros(V, dtype=torch.float64, device="cuda")     # scalar ||Delta||^2 per token

    corpus = torch.load(os.path.join(DATA_DIR, "finetune_ids.pt"))
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for s in range(0, A.seqs, A.mb):
            ids = corpus[order[s:s + A.mb]].to("cuda").long()
            CAP["on"] = True
            # free side: BASE weights by default; the target surface under --free-same
            _set((ADAPT if A.free_same else BASEW) if ADAPT is not None else None)
            RES.disable_residency()
            CAP["buf"] = {}; model(ids); free = {k: v for k, v in CAP["buf"].items()}
            _set(ADAPT)                                     # residency side: the target surface
            RES.enable_residency(R=8)
            CAP["buf"] = {}; model(ids); res = {k: v for k, v in CAP["buf"].items()}
            CAP["on"] = False
            flat = ids.reshape(-1)
            d = torch.stack([(free[i] - res[i]).reshape(-1, H) for i in range(L)], dim=1)  # [N,L,H]
            sums.index_add_(0, flat, d)
            counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
            sq.index_add_(0, flat, d.pow(2).sum((1, 2)).double())
            del free, res, d
            done += ids.shape[0]
            if (s // A.mb) % 25 == 0:
                print(f"  {done}/{A.seqs} seqs  {done*4096/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)

    seen = counts > 0
    n = counts.clamp(min=1)
    mean = sums / n[:, None, None]
    mean_sq = mean.pow(2).sum((1, 2)).double()
    # within-token variance: E||d||^2 - ||E d||^2, pooled over tokens with n>=2
    # sigma^2_within : per-token mean squared deviation of Delta about that token's own mean,
    #   E||d||^2 - ||mean||^2, pooled over tokens with at least two observations.
    # sigma^2_between: spread of the TRUE per-token mean vectors, E_t||mean_t - grand||^2.
    #   ||mean_t||^2 is biased upward by within/n_t because mean_t is itself estimated from n_t
    #   samples, and that bias is largest for exactly the rare rows shrinkage exists to handle,
    #   so it is subtracted. An earlier version of this used var(||mean_t||^2), the variance of a
    #   squared norm, which is a fourth-moment quantity and not sigma^2_between at all.
    m2 = seen & (counts >= 2)
    per_tok_within = ((sq[m2] / n[m2].double()) - mean_sq[m2]).clamp(min=0)
    within = float(per_tok_within.mean())
    grand = mean[seen].mean(0)                                   # [L,H]
    dev = (mean[m2] - grand).pow(2).sum((1, 2)).double()          # ||mean_t - grand||^2
    between = float((dev - per_tok_within / n[m2].double()).clamp(min=0).mean())
    lam = within / max(between, 1e-12)
    print(f"[calib] tokens seen {int(seen.sum())}/{V}  within {within:.6g}  between {between:.6g}  "
          f"lambda* {lam:.4f}", flush=True)

    w = counts / (counts + lam)                       # shrinkage weight per token
    T = mean * w[:, None, None]                       # shrunk table [V,L,H]
    meta = {"seqs": A.seqs, "tokens": int(counts.sum()), "tokens_seen": int(seen.sum()),
            "lambda_star": lam, "within": within, "between": between,
            "delta": ("MoE_free - MoE_residency on the UNTRAINED base model" if not A.resume_c
                      else f"MoE_free - MoE_residency on the C surface at {A.resume_c}"),
            "reference": A.resume_c or "untrained base",
            "free_side": ("target surface (--free-same)" if A.free_same else "base model"),
            "shrunk_before_svd": True}

    for rk in A.ranks.split(","):
        if rk == "full":
            torch.save({"rank": "full", "P": T.cpu()}, os.path.join(DATA_DIR, f"calib_table_rfull{A.suffix}.pt"))
            print(f"[calib] wrote full-rank table, ||T||={float(T.norm()):.4f}", flush=True)
        else:
            r = int(rk)
            D = w.sqrt()
            M = (D[:, None] * T.reshape(V, L * H))                     # precision-weighted
            U, S, Vh = torch.svd_lowrank(M, q=min(r + 32, min(M.shape)), niter=4)
            U, S, Vh = U[:, :r], S[:r], Vh[:, :r]
            Uple = (U * S) / D.clamp(min=1e-8)[:, None]                # D^-1 U S  -> [V, r]
            Vple = Vh.T.reshape(r, L, H).contiguous()                  # [r, L, H]
            Uple[~seen] = 0                                            # unseen rows exactly zero
            approx = (Uple @ Vple.reshape(r, -1))
            rel = float((approx - T.reshape(V, L * H)).norm() / T.reshape(V, L * H).norm())
            torch.save({"rank": str(r), "U": Uple.cpu(), "V": Vple.cpu()},
                       os.path.join(DATA_DIR, f"calib_table_r{r}{A.suffix}.pt"))
            meta[f"rel_recon_err_r{r}"] = rel
            print(f"[calib] wrote r={r} table, relative reconstruction error {rel:.4f}", flush=True)

    json.dump(meta, open(os.path.join(DATA_DIR, f"calib_meta{A.suffix}.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(ABLATIONS, f"ple_calib_meta{A.suffix}.json"), "w"), indent=1)
    print("[calib] done", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
