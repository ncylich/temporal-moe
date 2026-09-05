#!/usr/bin/env python3
"""C9 -- frequency-stratified token AUC: does the lexical shortcut live on rare or common tokens?

The locus probes report one token AUC per (layer, expert), pooled over the whole stream. If the
unconstrained router's lexical binding is concentrated on rare tokens, that changes which layers
deserve the constraint and reframes H2 -- a shortcut that only exists for words the model sees a
handful of times is a different thing from one that governs routing everywhere.

Method. Fit the token probe exactly as delex_locus does, on held-out documents, then split the score
rows by the corpus frequency of their token and report AUC within each stratum. Strata are frequency
*quintiles of the token stream*, so each holds a comparable number of scored rows -- quintiles of the
vocabulary would put almost every row in the top bin, since token frequency is Zipfian.

One subtlety worth stating: AUC within a stratum is not comparable to AUC over the whole stream, and
strata are not comparable to each other in difficulty either, because restricting to a frequency band
also restricts how much the label varies. The comparison that carries the argument is *between
regimes* within the same stratum and layer, and between strata within one model, not the absolute
level.

Token frequency comes from the capture itself (the count of each recovered token id in the 131k-token
batch), which is a sample of corpus frequency, not corpus frequency itself. It is adequate for
ranking tokens into wide bands and is what is available without the corpus.

Output: results/ablations/mechinterp_freqstrat.csv, one row per (run, layer, expert, stratum).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delex_locus
import delex_oracle
import registry
import safe_csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

OUT = os.path.join(ABLATIONS, "mechinterp_freqstrat.csv")
NSTRAT = 5
HEADER = ["label", "run", "budget", "regime", "layer", "expert", "split", "stratum",
          "stratum_lo_count", "stratum_hi_count", "n_score_rows", "usage_in_stratum",
          "token_AUC_in_stratum"]


def strata(ids, nid, nstrat=NSTRAT):
    """-> (stratum index per token position, [(lo,hi) occurrence count per stratum]).

    Quintiles of the token *stream*: sort distinct ids by their count in this batch, then cut so each
    stratum holds about the same number of token positions.
    """
    cnt = np.bincount(ids, minlength=nid).astype(np.int64)
    order = np.argsort(cnt, kind="stable")                 # rarest first
    cum = np.cumsum(cnt[order])
    edges = np.searchsorted(cum, np.linspace(0, cum[-1], nstrat + 1)[1:])
    strat_of_id = np.empty(nid, np.int64)
    prev = 0
    bounds = []
    for s, e in enumerate(edges):
        e = max(e, prev)
        sel = order[prev:e + 1] if s == nstrat - 1 else order[prev:e]
        strat_of_id[sel] = s
        c = cnt[sel]
        bounds.append((int(c.min()) if c.size else 0, int(c.max()) if c.size else 0))
        prev = e
    return strat_of_id[ids], bounds


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
        ids = delex_oracle.token_ids(emb)
        nid = int(ids.max()) + 1
        strat, bounds = strata(ids, nid)
        tr, te = delex_locus.split_index(S, B, split)
        design = delex_locus._Probe(
            delex_locus._standardize(emb.reshape(ntok, H).astype(np.float64)), tr, te)
        te_strat = strat[te]
        layers = registry.moe_layers(d)
        if not layers:
            print(f"[warn] {r.name}: capture holds no MoE layers, skipping (rerun its capture)")
            continue
        print(f"[run] {r.name} ({r.regime}, {r.grain_label}, {r.budget}) "
              f"{nid} token ids, strata by occurrence count {bounds}", flush=True)
        for L in layers:
            Ld = d["layers"][L]
            lg = Ld["logits"].float().numpy()
            mask = Ld["mask"].numpy() if Ld["mask"] is not None else None
            Y = delex_locus.firing(lg, mask, int(Ld["k"])).reshape(ntok, lg.shape[-1]) \
                .astype(np.float64)
            # One shared solve on the full fit split, then score each stratum separately: the probe is
            # the same probe the locus numbers use, only the evaluation set is restricted.
            Ytr = Y[tr]
            W = design.Ainv @ (design.Xtr.T @ (Ytr - Ytr.mean(0)))
            scores_all = design.Xte @ W
            Yte = Y[te]
            per = []
            for s in range(NSTRAT):
                m = te_strat == s
                a = delex_locus.auc_batch(scores_all[m], Yte[m])
                usage = Yte[m].sum(0).astype(int)
                for e in range(Y.shape[1]):
                    rows.append([r.name, r.name, r.budget, r.regime, L, e, split, s,
                                 bounds[s][0], bounds[s][1], int(m.sum()), int(usage[e]),
                                 delex_locus._r(a[e])])
                per.append(float(np.nanmedian(a)))
            summary.append((r.name, L, per))
            print(f"    L{L:<3} median token AUC by frequency stratum (rarest first): "
                  + "  ".join(f"{v:.3f}" for v in per), flush=True)

    os.makedirs(ABLATIONS, exist_ok=True)
    safe_csv.guard(OUT, rows, key_index=HEADER.index("run") if "run" in HEADER else None)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {OUT}: {len(rows)} rows")
    print("\nmedian over layers, per run (rarest stratum first):")
    for run in sorted({s[0] for s in summary}):
        per = np.array([s[2] for s in summary if s[0] == run])
        print(f"  {run:24} " + "  ".join(f"{v:.3f}" for v in np.median(per, axis=0)))


if __name__ == "__main__":
    main()
