#!/usr/bin/env python3
"""The zero-property check (PLE_PLAN.md §4 item 5), plus the init/gradient sanity it depends on.

Two claims are tested, in order of when they can run:

  --init   runnable now, before any cell trains. On a freshly constructed table:
             1. the PLE contribution is exactly 0.0 (bitwise), so the model is bit-identical to
                the no-PLE model and parity is structural rather than lucky;
             2. dL/dU is nonzero, so the branch can actually leave zero. This is the check that
                catches the dead-branch init: with a zero gate as well as a zero table, every
                gradient here is 0 and the cell would silently train nothing.

  --trained PATH  runs after the first cell trains (§4 item 5 says so explicitly). On a trained
                  table:
             3. every row whose token never appeared in the training corpus is bit-zero;
             4. a forward pass on an uncovered token matches the no-PLE model EXACTLY, compared
                bitwise on the logits, not within a tolerance.

Claims 3 and 4 are the deliverable; 1 and 2 are what make them worth expecting.
"""

import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ple as PLE                           # noqa: E402
from olmoe_paths import DATA_DIR            # noqa: E402


def check_init(vocab=50304, layers=16, hidden=2048, rank=32, device="cpu"):
    t = PLE.FactoredPLE(vocab, layers, hidden, rank, device=device)
    ids = torch.tensor([[1, 2, 3]], device=device)
    out = t(ids, 0, torch.float32)
    exact_zero = bool((out == 0).all())

    # gradient reachability: a loss that depends on the contribution must move the table
    t.zero_grad()
    loss = t(ids, 0, torch.float32).sum() + t(ids, 3, torch.float32).pow(2).sum()
    # .sum() of an exactly-zero tensor still has a nonzero derivative wrt U; that is the point
    loss = (t(ids, 0, torch.float32) * torch.randn(1, 3, hidden, device=device)).sum()
    loss.backward()
    tab = t.table_params()[0]
    g_tab = tab.grad
    grad_nonzero = g_tab is not None and bool((g_tab != 0).any())
    touched = torch.nonzero((g_tab != 0).any(-1)).flatten().tolist() if grad_nonzero else []
    return {
        "rank": rank,
        "contribution_bitwise_zero": exact_zero,
        "table_grad_nonzero": grad_nonzero,
        "rows_receiving_grad": touched,
        "gate_init": float(t.g[0]),
        "n_params": t.n_params(),
    }


def consumed_covered(train_tokens, mb=16, seq=4096, vocab=50304):
    """Token ids the cell ACTUALLY saw, not the ids present anywhere in the 1B corpus.

    A cell trains on a prefix of the shuffled order, `order[0 : n_steps*mb]`, so its covered set is
    a strict subset of the corpus vocabulary and the uncovered set is correspondingly larger. Using
    the whole corpus here would test a far weaker claim -- and would wrongly count a row as
    "covered" that the cell never had a chance to update.
    """
    corpus = torch.load(f"{DATA_DIR}/finetune_ids.pt")
    order = torch.randperm(corpus.shape[0], generator=torch.Generator().manual_seed(0))
    n_steps = -(-train_tokens // (mb * seq))          # ceil, matching `while seen < tokens`
    used = order[: n_steps * mb]
    ids = torch.unique(corpus[used].reshape(-1))
    covered = torch.zeros(vocab, dtype=torch.bool)
    covered[ids.long()] = True
    return covered, int(used.numel())


def check_trained(path, train_tokens=None, mb=16, model=None, device="cuda"):
    """Claims 3 and 4 on a trained table saved by train_ple.py."""
    sd = torch.load(path, map_location="cpu")
    rank = sd.pop("rank")
    tab = sd["P"] if "P" in sd else sd["U"]                 # [vocab, ...]
    row_nonzero = (tab.reshape(tab.shape[0], -1) != 0).any(-1)

    if train_tokens is None:
        raise SystemExit("--train-tokens is required: the covered set is defined by what the cell saw")
    covered, n_seq = consumed_covered(train_tokens, mb=mb, vocab=tab.shape[0])

    uncovered_nonzero = int((row_nonzero & ~covered).sum())
    out = {
        "rank": rank,
        "n_rows": int(tab.shape[0]),
        "train_tokens": train_tokens,
        "train_seqs_consumed": n_seq,
        "n_covered": int(covered.sum()),
        "n_uncovered": int((~covered).sum()),
        "uncovered_rows_bit_zero": uncovered_nonzero == 0,
        "uncovered_rows_violating": uncovered_nonzero,
        "covered_rows_that_moved": int((row_nonzero & covered).sum()),
    }

    # claim 4: bitwise-equal logits on an uncovered token
    if model is not None:
        unc = torch.nonzero(~covered).flatten()
        if unc.numel():
            ids = unc[:8].unsqueeze(0).to(device)
            PLE.uninstall()
            with torch.no_grad():
                a = model(ids).logits.clone()
            PLE._STATE["ple"] = _reload(sd, rank, device)
            with torch.no_grad():
                b = model(ids).logits.clone()
            out["forward_bitwise_equal"] = bool((a == b).all())
            out["max_abs_logit_diff"] = float((a - b).abs().max())
    return out


def _reload(sd, rank, device):
    r = rank if rank == "full" else int(rank)
    tab = sd["P"] if "P" in sd else sd["U"]
    vocab = tab.shape[0]
    layers, hidden = (tab.shape[1], tab.shape[2]) if r == "full" else (sd["V"].shape[1], sd["V"].shape[2])
    t = PLE.FactoredPLE(vocab, layers, hidden, r, device=device)
    t.load_state_dict({k: v.to(device) for k, v in sd.items()})
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--trained", default=None, help="path to ple_table_<tag>.pt")
    ap.add_argument("--train-tokens", type=int, default=None,
                    help="token budget the cell ran, which defines the covered set")
    ap.add_argument("--mb", type=int, default=16)
    A = ap.parse_args()
    if A.init or not A.trained:
        for r in (32, 128, "full"):
            print(json.dumps(check_init(rank=r), indent=1))
    if A.trained:
        print(json.dumps(check_trained(A.trained, train_tokens=A.train_tokens, mb=A.mb), indent=1))
