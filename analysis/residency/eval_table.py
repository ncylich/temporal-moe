#!/usr/bin/env python3
"""Evaluate a PLE table WITHOUT training it (PLE_PLAN.md §9), and bucket recovery by frequency (§8.3).

Two uses, both cheap because neither trains anything:

  --table calib_table_r512.pt
      §9's training-free recovery number: install the calibrated table on the model it was
      captured against and score the audited slice. §9 calls this a LOWER BOUND on PLE's value,
      because it measures the residual conditional on an allocation the model has already made,
      whereas a trained cell is free to allocate differently. A null here therefore cannot kill the
      trained cells, and must not be reported as if it could.

  --buckets
      §8 item 3: recovery bucketed by training-corpus occurrence count. If gains concentrate on
      frequent tokens that is an honest scaling caveat, not something to hide behind an aggregate.

--csurf applies a C-surface checkpoint first, so a table captured against the 50M surface is scored
against that same surface rather than the untrained base. Scoring a table against a surface it was
not captured on is the mismatch this flag exists to prevent.
"""

import argparse, csv, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ple"))  # sibling dir (2026-08 split)
import residency as RES               # noqa: E402
import ple as PLE                     # noqa: E402
from olmoe_paths import DATA_DIR      # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS           # noqa: E402

BASE, IMPOSE = 0.6727, 2.7507
BUCKETS = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, 9999), (10000, 99999), (100000, None)]


def recovery(b):
    return 1.0 - (b - BASE) / (IMPOSE - BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None, help="calib_table_*.pt or ple_table_*.pt; omit for none")
    ap.add_argument("--csurf", default=None, help="C-surface checkpoint to apply first")
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--buckets", action="store_true")
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    D = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))["divisor_D"]
    model, _ = RES.load_model()
    if A.lora:
        LORA_PS = RES.add_lora(model, r=A.lora, alpha=2 * A.lora)
    if A.csurf:
        ck = torch.load(A.csurf, map_location="cuda")
        # The trainer's parameter list is router + norms + LoRA, in that order, so the
        # checkpoint's masters must be zipped against the SAME list. Zipping against only
        # router+norms truncates silently -- zip stops at the shorter sequence -- discarding every
        # trained LoRA tensor and leaving LoRA at its zero-init no-op. That reconstructed
        # ce_ple_128 at 1.1652 BPB when the cell had trained to 0.8327. The assert makes the
        # mismatch loud instead of silent.
        tp = RES.router_params(model) + RES.norm_params(model) + (LORA_PS if A.lora else [])
        assert len(tp) == len(ck["masters"]), \
            f"param/checkpoint mismatch: {len(tp)} params vs {len(ck['masters'])} masters"
        with torch.no_grad():
            for p, m in zip(tp, ck["masters"]):
                p.data.copy_(m.to("cuda").to(p.dtype))
        print(f"[eval] C surface from {os.path.basename(A.csurf)} at {ck['seen']/1e6:.0f}M tokens",
              flush=True)

    if A.table:
        sd = torch.load(A.table, map_location="cuda")
        rank = sd.pop("rank")
        r = rank if rank == "full" else int(rank)
        t = PLE.install(model, r, device="cuda")
        with torch.no_grad():
            for k, v in sd.items():
                getattr(t, k).copy_(v.to("cuda"))
        tp0 = t.table_params()[0]
        print(f"[eval] table {os.path.basename(A.table)} rank={rank} ||T||={float(tp0.norm()):.4f} "
              f"rows nonzero {int((tp0.reshape(tp0.shape[0],-1)!=0).any(-1).sum())}", flush=True)
    else:
        PLE.uninstall()

    RES.enable_residency(R=8)
    RES.reset_telem(); RES._CFG["collect_telem"] = True
    model.eval()

    ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
    sub = ids[torch.linspace(0, ids.shape[0] - 1, A.eval_n).long()].long()
    tot = n = 0
    ce_by_id = torch.zeros(model.config.vocab_size, dtype=torch.float64)
    cnt_by_id = torch.zeros(model.config.vocab_size, dtype=torch.float64)
    with torch.no_grad():
        for i in range(sub.shape[0]):
            x = sub[i:i + 1].to("cuda")
            lg = model(x).logits.float()
            ce = torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
            tot += float(ce.sum()); n += ce.numel()
            if A.buckets:
                tid = x[0, 1:].cpu().long()
                ce_by_id.index_add_(0, tid, ce.cpu().double())
                cnt_by_id.index_add_(0, tid, torch.ones_like(tid, dtype=torch.float64))
    RES._CFG["collect_telem"] = False
    swap, ent = RES.telem_summary(model.config.num_experts)
    bpb = (tot / n) / D
    print(f"[eval] {A.tag}: BPB={bpb:.6f} recovery={recovery(bpb)*100:.2f}% swap={swap:.4f} "
          f"ent={ent:.4f} (NO TRAINING)", flush=True)

    rows = [{"tag": A.tag, "table": os.path.basename(A.table) if A.table else "none",
             "csurf": os.path.basename(A.csurf) if A.csurf else "none", "lora": A.lora,
             "bucket": "ALL", "n_eval_tokens": n, "bpb": round(bpb, 6),
             "recovery_pct": round(recovery(bpb) * 100, 2),
             "swap_rate": round(swap, 6), "usage_entropy": round(ent, 6), "divisor": D}]
    if A.buckets:
        corpus = torch.load(os.path.join(DATA_DIR, "finetune_ids.pt"))
        occ = torch.bincount(corpus.reshape(-1).long(), minlength=model.config.vocab_size).double()
        for lo, hi in BUCKETS:
            m = (occ >= lo) if hi is None else ((occ >= lo) & (occ <= hi))
            c, k = float(ce_by_id[m].sum()), float(cnt_by_id[m].sum())
            if k == 0:
                continue
            b = (c / k) / D
            rows.append({"tag": A.tag, "table": rows[0]["table"], "csurf": rows[0]["csurf"],
                         "lora": A.lora,
                         "bucket": f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}"),
                         "n_eval_tokens": int(k), "bpb": round(b, 6),
                         "recovery_pct": round(recovery(b) * 100, 2),
                         "swap_rate": "", "usage_entropy": "", "divisor": D})

    path = A.out or os.path.join(ABLATIONS, "ple_trainfree_and_buckets.csv")
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("   ", r)
    print("wrote", path)


if __name__ == "__main__":
    main()
