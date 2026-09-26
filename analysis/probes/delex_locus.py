#!/usr/bin/env python3
"""Locus probes (A1-A3): per-expert linear probes predicting expert firing y_e(t) from
(i) the current token embedding x_tok(t)=E[x_t] and (ii) the excluded-context mean
x_ctx(t)=mean{E[x_t'] : 0<|t'-t|<=w} (current token excluded, within-sequence). Fit on the first 70%
of the token stream, AUC on the last 30%. Above 0.5 is signal; A_ctx > A_tok means that layer's
routing is better predicted by the surroundings than by the token being processed.

Emits per (variant, layer, expert): token_AUC, context_AUC, context_minus_token, usage_count, and
per (variant, layer, feature, null_type): the measured chance floor.

Three things changed here relative to the version that produced the published numbers, all of them
coverage fixes required by MECHINTERP_RERUN_PLAN.md:

1. **Layers come from the capture, not from a constant.** The old `LAYERS = [2,3,4,5,6]` plus
   `if L not in d["layers"]: continue` dropped every deeper layer silently. The 1e19 captures hold
   MoE layers 2-14, so that was 5 of 13 layers measured and 8 discarded with no warning. Note this
   is also more than the re-run plan asked for: `range(2, 10)` would still have dropped 10-14. Any
   layer this script cannot cover is warned about and recorded in the coverage output, never
   dropped.

2. **All three context windows for every cell** (A3), not one. `variant` keeps the published
   encoding -- kwin = w=k/2, kfull = w=k, base = w=32 -- and a `window` column now states w
   outright, so the semantics no longer decode only by cross-referencing prose. The old
   `mechinterp_locus_1e19.csv` wrote w=k under the name `base`, colliding with the w=32 meaning it
   has in `mechinterp_locus.csv`; that collision is why `window` exists.

3. **Null floors per layer** (A2), over every probed expert rather than a ~8-per-layer subsample,
   because the gate they feed is what licenses reading any of the AUCs as signal.

Probes are ridge-linear with a shared Gram inverse: within a (variant, layer) the design matrix is
identical across experts, so one inversion and one GEMM serve all of them. The design does not
depend on the layer either, so the inversion is hoisted out of the layer loop entirely. AUC is
rank-based, so a linear probe tracks a logistic one closely; --verify checks both the ridge-vs-
logistic gap and the batched-AUC implementation against sklearn.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry

SHIFTS = [1009, 2003, 5011]   # circular-shift nulls (>=1000, none a multiple of seq len 2048),
                              # averaged for a robust residency-autocorrelation-preserving null
RIDGE = 1.0                   # ridge shrinkage: keeps the strong real signal, pulls spurious null
                              # fits to 0.5 (unbiased null); tuned so iid+shift nulls land ~0.500
MIN_TRAIN_POS = 5             # an expert needs this many firings in the fit split to be probed
MIN_TEST_POS = 3              # ...and this many in the score split for a defined AUC

# variant name -> context half-width w, as a function of the layer's top-k. The published encoding.
WINDOWS = {"kwin": lambda k: max(1, k // 2), "kfull": lambda k: k, "base": lambda k: 32}


def firing(lg, mask, k):
    """[S,B,E] bool: the served set. temporal -> resident mask; unconstrained -> top-k of logits."""
    if mask is not None:
        return mask
    idx = np.argpartition(-lg, k - 1, axis=-1)[..., :k]
    sel = np.zeros(lg.shape, bool)
    np.put_along_axis(sel, idx, True, -1)
    return sel


def context_mean(emb, w):
    """emb [S,B,H] -> x_ctx [S,B,H]: per-sequence mean of +-w neighbours, current token excluded."""
    S, B, H = emb.shape
    x = emb.transpose(1, 0, 2).astype(np.float64)             # [B,S,H]
    cs = np.zeros((B, S + 1, H), np.float64)
    np.cumsum(x, axis=1, out=cs[:, 1:])
    idx = np.arange(S)
    lo = np.clip(idx - w, 0, None)
    hi = np.clip(idx + w + 1, None, S)
    win_sum = cs[:, hi] - cs[:, lo]                            # [B,S,H], inclusive of current
    cnt = (hi - lo).astype(np.float64)[None, :, None] - 1.0    # exclude the current token
    ctx = (win_sum - x) / np.clip(cnt, 1, None)
    return np.ascontiguousarray(ctx.transpose(1, 0, 2))        # [S,B,H]


def _standardize(X):
    """Column-standardize. Fit on all rows: the probe's train/score split is over rows, and the
    scaling is a fixed affine reparametrisation, so it leaks no label information."""
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    return (X - mu) / sd


def auc_batch(scores, Y):
    """Held-out AUC for many probes at once.

    scores, Y: [n, E] float / bool, column e being one expert's scores and labels. Returns [E] with
    NaN where the column has no positives or no negatives. Rank-based Mann-Whitney form with proper
    average ranks for ties, which is exactly sklearn's roc_auc_score (checked by --verify).
    """
    n, E = scores.shape
    out = np.full(E, np.nan)
    order = np.argsort(scores, axis=0, kind="stable")
    srt = np.take_along_axis(scores, order, axis=0)
    pos = np.arange(n, dtype=np.float64)[:, None]
    # Average ranks within tie groups. A tie group spans sorted positions [first, last], so its
    # shared rank is (first+last)/2 + 1. Ties are not hypothetical: an all-zero label column gives
    # an all-zero score column, and that column must not crash before its AUC is set to NaN.
    starts = np.ones((n, E), bool)
    starts[1:] = srt[1:] != srt[:-1]
    ends = np.ones((n, E), bool)
    ends[:-1] = srt[:-1] != srt[1:]
    first = np.maximum.accumulate(np.where(starts, pos, -1.0), axis=0)
    last = np.minimum.accumulate(np.where(ends, pos, float(n))[::-1], axis=0)[::-1]
    ranks = np.empty((n, E), np.float64)
    np.put_along_axis(ranks, order, (first + last) / 2.0 + 1.0, axis=0)
    npos = Y.sum(0).astype(np.float64)
    nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    # Compute only on the defined columns. Evaluating the ratio everywhere and masking afterwards is
    # correct but divides by zero for any all-positive or all-negative column, which is routine when
    # scoring a frequency stratum or a lightly-used expert; the warning it raised was pure noise.
    rpos = (ranks * Y).sum(0)[ok]
    p, q = npos[ok], nneg[ok]
    out[ok] = (rpos - p * (p + 1) / 2) / (p * q)
    return out


def split_index(S, B, mode="sequence", frac=0.7):
    """Fit/score index arrays over the flattened [S*B] stream, where index = s*B + b.

    mode='sequence' (default): hold out whole sequences. The first round(frac*B) batch elements are
    fit, the rest scored, so no document appears in both halves.

    mode='position': the published split, `cut = int(frac*S*B)`. Because the stream is flattened with
    the batch dimension innermost, that cut falls at a sequence *position*, not at a token boundary:
    it fits on the first ~70% of every document and scores on the last ~30% of those same documents.
    Every document is therefore present in both halves, which lets a probe score above chance from
    document identity alone -- worth ~0.03-0.06 AUC on the widest context window, measured by
    delex_null_check.py's `perm-t` arm. Retained only to reproduce the published numbers.
    """
    if mode == "position":
        cut = int(frac * S * B)
        idx = np.arange(S * B)
        return idx[:cut], idx[cut:]
    if mode != "sequence":
        raise ValueError(f"unknown split mode {mode!r}")
    nb = max(1, min(B - 1, int(round(frac * B))))
    b = np.arange(B)
    grid = np.arange(S * B).reshape(S, B)
    return grid[:, b < nb].ravel(), grid[:, b >= nb].ravel()


class _Probe:
    """A fitted design: standardized features, a fit/score split, and one shared ridge solve.

    The split is an index pair rather than a scalar cut, so a document-disjoint split is expressible.
    """

    def __init__(self, X, tr, te):
        self.tr, self.te = tr, te
        self.Xtr = X[tr]
        self.Xte = X[te]
        H = X.shape[1]
        self.Ainv = np.linalg.inv(self.Xtr.T @ self.Xtr + RIDGE * np.eye(H))

    def aucs(self, Y):
        """Y [ntok, E] float 0/1 -> [E] held-out AUC, NaN where the expert is unprobeable."""
        Ytr, Yte = Y[self.tr], Y[self.te]
        W = self.Ainv @ (self.Xtr.T @ (Ytr - Ytr.mean(0)))     # [H, E], one GEMM for all experts
        a = auc_batch(self.Xte @ W, Yte)
        bad = (Ytr.sum(0) < MIN_TRAIN_POS) | (Yte.sum(0) < MIN_TEST_POS) | (Yte.sum(0) == len(Yte))
        a[bad] = np.nan
        return a


def analyze(cap_path, run, label=None, variants=("kwin", "kfull", "base"), verify=False,
            split="sequence"):
    """Probe one capture at every MoE layer it contains and every requested window.

    split: 'sequence' holds out whole documents (default); 'position' reproduces the published split.
    See split_index for why the difference matters.

    Returns (rows, floors, coverage, summary):
      rows      per (variant, layer, expert) AUC record
      floors    per (variant, layer, feature, null_type) measured chance floor
      coverage  per (variant, layer) how many experts were probed and how many omitted, and why
      summary   per-variant medians and null medians, for the driver's gate and headline
    """
    import torch
    r = registry.get(run)
    label = label or run
    d = torch.load(cap_path, map_location="cpu", weights_only=False)
    emb = d["emb"].float().numpy()                             # [S,B,H]
    S, B, H = emb.shape
    ntok = S * B
    tr, te = split_index(S, B, split)
    layers = registry.moe_layers(d)

    rows, floors, coverage = [], [], []
    summary = {}
    # Firing sets are layer-dependent but window-independent; unpack once and reuse.
    fires, ks = {}, {}
    for L in layers:
        Ld = d["layers"][L]
        lg = Ld["logits"].float().numpy()
        ks[L] = int(Ld["k"])
        mask = Ld["mask"].numpy() if Ld["mask"] is not None else None
        fires[L] = firing(lg, mask, ks[L]).reshape(ntok, lg.shape[-1]).astype(np.float64)
    kset = sorted(set(ks.values()))
    if len(kset) > 1:
        warnings.warn(f"{run}: top-k varies by layer {ks}; window w is set per layer from its own k")

    tok_design = _Probe(_standardize(emb.reshape(ntok, H).astype(np.float64)), tr, te)
    rng = np.random.default_rng(0)

    for variant in variants:
        # w depends only on k, and every layer here shares one k, so build each design once.
        ctx_designs = {k: _Probe(_standardize(context_mean(emb, WINDOWS[variant](k))
                                              .reshape(ntok, H).astype(np.float64)), tr, te)
                       for k in kset}
        med = {"tok": [], "ctx": [], "ndom": 0, "ntot": 0}
        nulls = {("token", "iid"): [], ("token", "circular"): [],
                 ("context", "iid"): [], ("context", "circular"): []}
        for L in layers:
            Y = fires[L]
            E = Y.shape[1]
            w = WINDOWS[variant](ks[L])
            ctx = ctx_designs[ks[L]]
            a_tok = tok_design.aucs(Y)
            a_ctx = ctx.aucs(Y)
            usage = Y.sum(0).astype(int)
            for e in range(E):
                dif = (a_ctx[e] - a_tok[e]) if np.isfinite(a_ctx[e]) and np.isfinite(a_tok[e]) else np.nan
                rows.append([label, run, r.budget, r.regime, L, e, int(usage[e]),
                             _r(a_tok[e]), _r(a_ctx[e]), _r(dif), variant, w, split])
            fin = np.isfinite(a_tok) & np.isfinite(a_ctx)
            med["tok"] += list(a_tok[fin])
            med["ctx"] += list(a_ctx[fin])
            med["ntot"] += int(fin.sum())
            med["ndom"] += int((a_ctx[fin] > a_tok[fin]).sum())
            omitted = int((~fin).sum())
            coverage.append([label, run, r.budget, L, variant, split, E, int(fin.sum()), omitted,
                             f"usage<{MIN_TRAIN_POS} in fit split or <{MIN_TEST_POS} in score split"
                             if omitted else ""])
            if omitted:
                warnings.warn(f"{run} L{L} {variant}: {omitted}/{E} experts unprobeable "
                              f"(too few firings); recorded in the coverage output, not dropped")
            # null floors, every probed expert, both null types
            perm = rng.permutation(ntok)
            for feat, design, a_real in (("token", tok_design, a_tok), ("context", ctx, a_ctx)):
                fi = np.isfinite(a_real)
                ai = design.aucs(Y[perm])
                floors.append([label, run, r.budget, L, variant, split, feat, "iid",
                               _r(np.nanmedian(ai[fi]), 4), int(fi.sum())])
                nulls[(feat, "iid")] += list(ai[fi][np.isfinite(ai[fi])])
                sh = np.nanmean([design.aucs(np.roll(Y, s, axis=0)) for s in SHIFTS], axis=0)
                floors.append([label, run, r.budget, L, variant, split, feat, "circular",
                               _r(np.nanmedian(sh[fi]), 4), int(fi.sum())])
                nulls[(feat, "circular")] += list(sh[fi][np.isfinite(sh[fi])])
        summary[variant] = {
            "tok": float(np.median(med["tok"])) if med["tok"] else float("nan"),
            "ctx": float(np.median(med["ctx"])) if med["ctx"] else float("nan"),
            "ctx_dom": med["ndom"] / max(1, med["ntot"]), "n": med["ntot"],
            "nulls": {f"{f}/{t}": float(np.median(v)) if v else float("nan")
                      for (f, t), v in nulls.items()},
            "layers": layers,
        }
        if verify:
            _verify(tok_design, fires[layers[0]])
            verify = False        # once is enough; it is a check on the implementation, not the data
    return rows, floors, coverage, summary


def _r(x, nd=4):
    return "" if x is None or not np.isfinite(x) else round(float(x), nd)


def _verify(design, Y):
    """Check the fast paths against references. Two separate claims, so two separate checks.

    1. auc_batch is exactly sklearn's roc_auc_score. Fed the *same* score vector, it must agree to
       floating-point equality -- this is a pure reimplementation and any gap is a bug.
    2. The batched ridge solve matches a per-expert solve, and the ridge probe tracks logistic.
       Here an exact match is not available: solving for all E experts in one GEMM accumulates in a
       different order than solving for one column, so the scores differ in the last few ulps and
       the AUC moves by ~1e-6. The bar is that it stays far below the 1e-4 we report at.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    picks = [e for e in range(Y.shape[1]) if Y[:, e].sum() > 50][:4]
    ours = design.aucs(Y)
    Ytr_all = Y[design.tr]
    Wall = design.Ainv @ (design.Xtr.T @ (Ytr_all - Ytr_all.mean(0)))
    Sall = design.Xte @ Wall
    print("  [verify] auc_batch vs sklearn on identical scores; then batched vs per-expert solve:",
          file=sys.stderr)
    for e in picks:
        y = Y[design.te, e]
        exact = roc_auc_score(y, Sall[:, e])                     # claim 1: same scores in
        Ytr = Ytr_all[:, e:e + 1]
        w1 = design.Ainv @ (design.Xtr.T @ (Ytr - Ytr.mean(0)))
        solo = roc_auc_score(y, (design.Xte @ w1).ravel())        # claim 2: one-column solve
        lr = LogisticRegression(max_iter=200, C=1.0).fit(design.Xtr, Ytr_all[:, e])
        lg = roc_auc_score(y, lr.decision_function(design.Xte))
        print(f"    expert {e}: auc_batch {ours[e]:.9f} vs sklearn {exact:.9f} "
              f"(delta {abs(ours[e]-exact):.1e}) | per-expert solve {solo:.6f} "
              f"(delta {abs(ours[e]-solo):.1e}) | logistic {lg:.3f}", file=sys.stderr)
        assert abs(ours[e] - exact) < 1e-12, "auc_batch disagrees with sklearn on identical scores"
        assert abs(ours[e] - solo) < 1e-4, "batched ridge solve differs beyond reporting precision"


ROW_HEADER = ["label", "run", "budget", "regime", "layer", "expert", "usage_count",
              "token_AUC", "context_AUC", "context_minus_token", "variant", "window", "split"]
FLOOR_HEADER = ["label", "run", "budget", "layer", "variant", "split", "feature", "null_type",
                "median_AUC_floor", "n_experts"]
COVERAGE_HEADER = ["label", "run", "budget", "layer", "variant", "split", "n_experts",
                   "n_probed", "n_omitted", "omission_reason"]
