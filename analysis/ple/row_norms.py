#!/usr/bin/env python3
"""Per-cell diagnostic: mean PLE row norm bucketed by training-corpus occurrence count.

No GPU. Runs on a trained table saved by train_ple.py.

WHY THIS EXISTS. The weight-decay coefficient on the table is an open decision, so this diagnostic
is what any chosen value should be set against rather than guessed. It matters most at low decay:
Adam normalizes each row's update by that row's own second moment, so after a single observation v
is approximately g^2 and the step is approximately lr*sign(g) -- a row seen once moves nearly as
far as a row seen ten thousand times. With little or no decay, full rank is therefore exposed to
memorizing noise in rare rows, which would be a pre-registered reason for full rank to lose. Losing
that way is a result, not a bug; the point of the diagnostic is to be able to tell.

This diagnostic is what tells the two apart. Reported per occurrence bucket:

  row_norm      ||U[t]|| (factored) or ||P[t]|| (full) -- the raw stored magnitude
  contrib_rms   RMS per element of the vector actually added to the residual stream,
                g_l * U[t] @ V[:,l,:], averaged over layers. This is the comparable quantity
                across ranks; row_norm is not, because U and P live in different spaces.

TRIGGER (stop and report, per the orchestrator's instruction): if at full rank the rare-row norms
grow to match or exceed the frequent-row norms WHILE eval BPB diverges from train loss, that is
the signal to pick a decay coefficient against this diagnostic rather than guess one a priori.
Neither half alone is the trigger -- rare rows may legitimately carry large corrections.
"""

import argparse, csv, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                 # noqa: E402

BUCKETS = [(0, 0), (1, 1), (2, 9), (10, 99), (100, 999),
           (1000, 9999), (10000, 99999), (100000, None)]


def consumed_counts(train_tokens, mb=16, seq=4096, vocab=50304):
    """Occurrence counts over the prefix the cell actually consumed, not the whole 1B corpus.

    A row is only updated when its token appears in a batch the cell saw, so the counts that
    explain a row's norm are the consumed ones. Using full-corpus counts would put rows in
    buckets they never earned.
    """
    corpus = torch.load(os.path.join(DATA_DIR, "finetune_ids.pt"))
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))
    n_steps = -(-train_tokens // (mb * seq))
    used = order[: n_steps * mb]
    return torch.bincount(corpus[used].reshape(-1).long(), minlength=vocab)


def row_stats(sd, rank, chunk=4096):
    """Returns (row_norm [vocab], contrib_rms [vocab])."""
    full = (rank == "full")
    g = sd["g"].float()
    if full:
        P = sd["P"]
        V_ = None
    else:
        U = sd["U"]
        V_ = sd["V"].float()          # [r, L, H]
    n = (sd["P"] if full else sd["U"]).shape[0]
    norms = torch.empty(n)
    rms = torch.empty(n)
    for i in range(0, n, chunk):
        if full:
            blk = P[i:i + chunk].float()                       # [c, L, H]
            norms[i:i + chunk] = blk.reshape(blk.shape[0], -1).norm(dim=-1)
            contrib = blk * g[None, :, None]
        else:
            blk = U[i:i + chunk].float()                       # [c, r]
            norms[i:i + chunk] = blk.norm(dim=-1)
            contrib = torch.einsum("cr,rlh->clh", blk, V_) * g[None, :, None]
        rms[i:i + chunk] = contrib.pow(2).mean(dim=(1, 2)).sqrt()
    return norms, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="ple_table_<tag>.pt from train_ple.py")
    ap.add_argument("--train-tokens", type=int, required=True)
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--tag", default=None)
    A = ap.parse_args()

    sd = torch.load(A.table, map_location="cpu")
    rank = sd.pop("rank")
    tag = A.tag or os.path.basename(A.table).replace("ple_table_", "").replace(".pt", "")
    counts = consumed_counts(A.train_tokens, mb=A.mb,
                             vocab=(sd["P"] if "P" in sd else sd["U"]).shape[0])
    norms, rms = row_stats(sd, rank)

    rows = []
    for lo, hi in BUCKETS:
        m = (counts >= lo) if hi is None else ((counts >= lo) & (counts <= hi))
        k = int(m.sum())
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        rows.append({
            "tag": tag, "rank": str(rank), "train_tokens": A.train_tokens,
            "occurrence_bucket": label, "n_rows": k,
            "mean_row_norm": round(float(norms[m].mean()), 8) if k else "",
            "mean_contrib_rms": round(float(rms[m].mean()), 10) if k else "",
            "max_row_norm": round(float(norms[m].max()), 8) if k else "",
            "frac_rows_exactly_zero": round(float((norms[m] == 0).float().mean()), 6) if k else "",
        })

    path = os.path.join(ABLATIONS, "ple_row_norms.csv")
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


if __name__ == "__main__":
    main()
