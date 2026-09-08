#!/usr/bin/env python3
"""PLE_PLAN.md §4 item 4: parameter / bandwidth accounting and corpus coverage.

Two independent parts, written to two CSVs under results/ablations/.

ACCOUNTING (--accounting, CPU, seconds)
    Per rung of the ladder (32 / 128 / 512 / full): total parameters, per-token flash fetch, and
    the RAM-resident basis. The comparator is one expert swap, which is the traffic the residency
    schedule already pays -- the point of §2 is that the PLE fetch is orders of magnitude below it.
    Every figure is derived from config.json, not from the plan's prose.

COVERAGE (--coverage, needs the corpus; loss attribution needs a GPU)
    "Coverage" is two different fractions and the plan asks for both:
      by-type    fraction of the vocabulary that appears in the training corpus
      by-token   fraction of audited-slice token OCCURRENCES whose id appears in the corpus
      by-loss    fraction of total eval cross-entropy carried by those occurrences
    by-token and by-loss are the ones that make a headline gain interpretable: a PLE row only
    exists for a token the corpus covered, so uncovered occurrences are exactly where PLE can do
    nothing. by-loss can differ sharply from by-token because rare tokens are the expensive ones.

    --model-for-loss PATH runs the forward pass needed for by-loss. Without it, by-loss is left
    empty rather than estimated.
"""

import argparse, csv, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR, MODEL_DIR   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                   # noqa: E402

BYTES = 2          # bf16 serving dtype; §2's "64 B at r=32" is 32 * 2


def arch():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    return {
        "vocab": cfg["vocab_size"],
        "hidden": cfg["hidden_size"],
        "layers": cfg["num_hidden_layers"],
        "experts": cfg["num_experts"],
        "top_k": cfg["num_experts_per_tok"],
        "inter": cfg["intermediate_size"],
        "tied": cfg["tie_word_embeddings"],
    }


def accounting_rows():
    a = arch()
    V, H, L, I = a["vocab"], a["hidden"], a["layers"], a["inter"]
    # one expert = gate_proj + up_proj + down_proj, each H x I
    expert_params = 3 * H * I
    rows = []
    for r in (32, 128, 512, "full"):
        if r == "full":
            params = V * L * H
            fetch = L * H * BYTES
            basis = 0
            note = "unfactored table [vocab, layers, hidden]"
        else:
            params = V * r + r * L * H
            fetch = r * BYTES
            basis = r * L * H * BYTES
            note = f"U[{V},{r}] + V[{r},{L},{H}]"
        # Training-time cost of the table, which is NOT the serving cost above. The table is an
        # fp32 Parameter indexed with advanced indexing, so autograd materialises a DENSE gradient
        # the full size of the table -- there is no sparse path unless nn.Embedding(sparse=True) is
        # used. At full rank that dense gradient alone is larger than the free memory measured on
        # this box, which is why §11's "optimizer state exhausts memory at full rank" bites earlier
        # than the optimizer.
        table_n = (V * L * H) if r == "full" else (V * r)
        basis_n = 0 if r == "full" else (r * L * H)
        train_bytes = (table_n + basis_n) * 4 * 2            # fp32 param + fp32 dense grad
        adam8 = (table_n + basis_n) * 2                      # 8-bit Adam: 2 states x 1 byte
        adam32 = (table_n + basis_n) * 8                     # fp32 Adam: 2 states x 4 bytes
        rows.append({
            "rank": r, "total_params": params, "params_M": round(params / 1e6, 3),
            "flash_fetch_bytes_per_token": fetch,
            "resident_basis_bytes": basis,
            "expert_swap_bytes": expert_params * BYTES,
            "fetch_vs_expert_swap": round(fetch / (expert_params * BYTES), 6),
            "train_param_plus_grad_GiB": round(train_bytes / 2**30, 3),
            "train_adam8bit_GiB": round(adam8 / 2**30, 3),
            "train_adam_fp32_GiB": round(adam32 / 2**30, 3),
            "train_total_adam8bit_GiB": round((train_bytes + adam8) / 2**30, 3),
            "note": note,
        })
    return a, expert_params, rows


def coverage(model_path=None, eval_n=256, device="cuda"):
    a = arch()
    t0 = time.time()
    corpus = torch.load(os.path.join(DATA_DIR, "finetune_ids.pt"))
    slice_ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
    print(f"  loaded corpus {tuple(corpus.shape)} slice {tuple(slice_ids.shape)} "
          f"in {time.time()-t0:.0f}s", flush=True)

    counts = torch.bincount(corpus.reshape(-1).long(), minlength=a["vocab"])
    covered = counts > 0
    n_types = int(covered.sum())

    # by-token over the WHOLE audited slice
    flat = slice_ids.reshape(-1).long()
    hit = covered[flat]
    by_token_all = float(hit.float().mean())

    # the eval subset the trainer actually scores, so by-loss and by-token refer to the same tokens
    sub = slice_ids[torch.linspace(0, slice_ids.shape[0] - 1, eval_n).long()].long()
    tgt = sub[:, 1:].reshape(-1)                       # next-token targets are what CE is over
    hit_sub = covered[tgt]
    by_token_sub = float(hit_sub.float().mean())

    out = {
        "vocab": a["vocab"],
        "corpus_tokens": int(corpus.numel()),
        "slice_tokens": int(slice_ids.numel()),
        "eval_subset_seqs": eval_n,
        "eval_subset_target_tokens": int(tgt.numel()),
        "coverage_by_type": round(n_types / a["vocab"], 6),
        "n_types_covered": n_types,
        "n_types_uncovered": a["vocab"] - n_types,
        "coverage_by_token_full_slice": round(by_token_all, 8),
        "coverage_by_token_eval_subset": round(by_token_sub, 8),
        "coverage_by_loss": "",
        "uncovered_target_tokens_eval_subset": int((~hit_sub).sum()),
    }

    # "Has a row" is not "has a row that learned anything". A row only moves when its token appears,
    # so the useful question is what fraction of eval tokens are backed by a row that saw enough
    # training occurrences. Reported at thresholds over the FULL corpus; a 50M-token cell sees ~1/20
    # of these counts, so divide by the cell's share of the corpus to read it for that cell.
    ctarget = counts[tgt]
    for thr in (1, 10, 100, 1000, 10000):
        out[f"eval_tokens_with_corpus_count_ge_{thr}"] = round(float((ctarget >= thr).float().mean()), 6)
    out["median_corpus_count_of_eval_token"] = int(ctarget.median())

    if model_path is not None:
        import residency as RES
        meta = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))
        D = meta["divisor_D"]
        model, _ = RES.load_model()
        if model_path == "C":
            nr, nn_, p = RES.load_c_adapted(model)
            print(f"  applied arm-C delta ({nr} routers, {nn_} norms) from {p}", flush=True)
            out["loss_model"] = "C (router + norm gains)"
        else:
            out["loss_model"] = "base, residency imposed untrained"
        RES.enable_residency(R=8)
        model.eval()
        tot = cov_loss = 0.0
        n = 0
        with torch.no_grad():
            for i in range(sub.shape[0]):
                x = sub[i:i + 1].to(device)
                lg = model(x).logits.float()
                ce = torch.nn.functional.cross_entropy(
                    lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
                m = covered[x[0, 1:].cpu()].to(device)
                tot += float(ce.sum()); cov_loss += float(ce[m].sum()); n += ce.numel()
        out["coverage_by_loss"] = round(cov_loss / tot, 8)
        out["eval_bpb_of_scored_model"] = round((tot / n) / D, 6)
        out["divisor"] = D
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounting", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--model-for-loss", default=None,
                    help="'base' for the released checkpoint, or a dir; omit to skip by-loss")
    A = ap.parse_args()
    os.makedirs(ABLATIONS, exist_ok=True)

    if A.accounting or not (A.accounting or A.coverage):
        a, ep, rows = accounting_rows()
        print(json.dumps(a, indent=1))
        print(f"one expert = 3 * {a['hidden']} * {a['inter']} = {ep} params = {ep*BYTES} bytes bf16")
        path = os.path.join(ABLATIONS, "ple_accounting.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        for r in rows:
            print(r)
        print("wrote", path)

    if A.coverage:
        out = coverage(A.model_for_loss)
        path = os.path.join(ABLATIONS, "ple_coverage.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out.keys()))
            w.writeheader(); w.writerow(out)
        print(json.dumps(out, indent=1))
        print("wrote", path)
