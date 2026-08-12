#!/usr/bin/env python3
"""Construct the deliberately held-out token set for the zero-property check (PLE_PLAN.md §4 item 5).

WHY A HELD-OUT SET IS NEEDED. Coverage is 99.99998% by token occurrence and the median audited-slice
token appears 429,893 times in the corpus, so the rows that are naturally uncovered are unused vocab
padding slots -- 1,113 of them at a 50M-token cell. Verifying that those are bit-zero proves almost
nothing: they are rows the model had no opportunity to touch for trivial reasons. The claim worth
testing is that a row the model COULD have trained, and which is simply withheld, stays exactly
zero and leaves the forward pass bit-identical to the no-PLE model.

CONSTRUCTION. Stratified across occurrence deciles rather than sampled uniformly, because a
uniformly sampled set is almost entirely rare tokens and would test only the easy case. Frequent
tokens are the strong case: they receive gradient on nearly every step, so a frequent held-out row
staying bit-zero is real evidence the exclusion is airtight.

The constraint that makes this affordable is loss mass. Holding out a token means PLE can never
help it, so the set must carry a negligible share of eval loss or the ladder's headline number is
quietly penalised. The set is therefore grown decile by decile under a hard cap on the fraction of
eval cross-entropy it accounts for (default 0.1%), and the achieved share is REPORTED rather than
assumed -- if the cap cannot be met while including the top decile, the top decile is dropped and
that is stated, rather than silently eating the cost.

Loss mass is measured on the C-adapted model, the surface PLE cells train on.

Output: results/ablations/ple_heldout.csv (one row per held-out token) plus a summary line.
"""

import argparse, csv, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-decile", type=int, default=20)
    ap.add_argument("--max-loss-share", type=float, default=0.001,
                    help="hard cap on the fraction of eval CE the held-out set may carry")
    ap.add_argument("--train-tokens", type=int, default=50_000_000)
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--eval-n", type=int, default=256)
    A = ap.parse_args()

    import residency as RES
    corpus = torch.load(os.path.join(DATA_DIR, "finetune_ids.pt"))
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))
    n_steps = -(-A.train_tokens // (A.mb * 4096))
    counts = torch.bincount(corpus[order[: n_steps * A.mb]].reshape(-1).long(), minlength=50304)

    slice_ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
    sub = slice_ids[torch.linspace(0, slice_ids.shape[0] - 1, A.eval_n).long()].long()
    tgt = sub[:, 1:].reshape(-1)

    # per-token-id eval CE on the C-adapted model
    meta = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))
    model, _ = RES.load_model()
    RES.load_c_adapted(model)
    RES.enable_residency(R=8)
    model.eval()
    ce_by_id = torch.zeros(50304, dtype=torch.float64)
    total = 0.0
    with torch.no_grad():
        for i in range(sub.shape[0]):
            x = sub[i:i + 1].to("cuda")
            lg = model(x).logits.float()
            ce = torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none").cpu()
            ce_by_id.index_add_(0, x[0, 1:].cpu().long(), ce.double())
            total += float(ce.sum())

    # deciles over tokens the cell actually sees, most frequent decile first so the strong case is
    # included whenever the loss budget allows it
    seen = torch.nonzero(counts > 0).flatten()
    seen = seen[counts[seen].argsort(descending=True)]
    deciles = torch.chunk(seen, 10)
    g = torch.Generator().manual_seed(A.seed)

    chosen, share = [], 0.0
    dropped = []
    for d, band in enumerate(deciles):
        pick = band[torch.randperm(band.numel(), generator=g)[: A.n_per_decile]]
        add = float(ce_by_id[pick].sum()) / total
        if share + add > A.max_loss_share:
            dropped.append(d)
            continue
        chosen.append(pick); share += add
    ids = torch.cat(chosen) if chosen else torch.tensor([], dtype=torch.long)

    rows = [{"token_id": int(t), "corpus_count_in_cell": int(counts[t]),
             "eval_ce_nats": round(float(ce_by_id[t]), 6),
             "eval_loss_share": round(float(ce_by_id[t]) / total, 10)}
            for t in ids.sort().values]
    path = os.path.join(ABLATIONS, "ple_heldout.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    torch.save({"ids": ids, "loss_share": share, "train_tokens": A.train_tokens},
               os.path.join(DATA_DIR, "ple_heldout.pt"))

    print(f"held-out tokens      : {ids.numel()}")
    print(f"eval loss share      : {share*100:.4f}%  (cap {A.max_loss_share*100:.4f}%)")
    print(f"deciles dropped      : {dropped if dropped else 'none'} (0 = most frequent)")
    print(f"count range in cell  : {int(counts[ids].min())} .. {int(counts[ids].max())}")
    print("wrote", path)


if __name__ == "__main__":
    main()
