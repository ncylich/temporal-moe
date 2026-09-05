#!/usr/bin/env python3
"""Is `grouped_mm` 2.69x faster and correct, or 2.69x faster and wrong?

The kernel benchmark reported max|dlogit| = 4.562 against eager. bf16 reassociation in a grouped GEMM
is worth ~1e-3, so 4.56 is either a real defect or an artefact of how that benchmark compared. It
never validated its own harness: the eager result was taken as the reference and assigned zero
difference by construction, so a non-deterministic model or a stale config would look like a
difference in the candidate.

Three checks, in the order that isolates cause from symptom:

    eager vs eager      run the same configuration twice. Any non-zero here means the harness is
                        measuring noise and the candidate's number means nothing.
    dlogit + BPB        max|dlogit| is one outlier position and is a poor summary. BPB over a real
                        held-out slice is the quantity we actually report, so a kernel that leaves
                        BPB unchanged to ~1e-4 is computing the same model whatever a single logit
                        does; one that moves BPB is broken regardless of how fast it is.
    argmax agreement    fraction of positions whose top-1 token is unchanged. Downstream tasks are
                        scored by ranking continuations, so this is closer to what a task cares
                        about than either of the above.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402


def set_impl(model, name):
    seen = []
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and id(cfg) not in [id(x) for x in seen]:
            cfg._experts_implementation = name
            seen.append(cfg)
    for m in model.modules():
        if getattr(m, "config", None) is not None:
            try:
                m.config._experts_implementation = name
            except Exception:
                pass


@torch.no_grad()
def logits_and_bpb(model, ids_bench, slice_ids, divisor):
    out = model(ids_bench).logits.float().clone()
    tot = ntok = 0
    for i in range(len(slice_ids)):
        b = slice_ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return out, (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/qwen3-30b")
    ap.add_argument("--family", default="qwen3")
    ap.add_argument("--data", default="/workspace/qwen3moe-adapt/data")
    ap.add_argument("--slice-name", default="qwen3")
    ap.add_argument("--n-seq", type=int, default=4)
    A = ap.parse_args()

    meta = json.load(open(f"{A.data}/bpb_slice_meta_{A.slice_name}.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model(path=A.model, family=A.family)
    blk, rtr = RQ.FAMILIES[A.family]
    blk.forward, rtr.forward = RQ._ORIG[A.family]                  # stock: this is about the kernel
    sl = torch.load(f"{A.data}/bpb_slice_ids_{A.slice_name}.pt", weights_only=False)[: A.n_seq]
    ids = torch.randint(0, model.config.vocab_size, (4, 512), device="cuda")

    set_impl(model, None)
    e1, bpb_e1 = logits_and_bpb(model, ids, sl, D)
    set_impl(model, None)
    e2, bpb_e2 = logits_and_bpb(model, ids, sl, D)
    d_ee = float((e2 - e1).abs().max())
    print(f"  eager vs eager      max|dlogit| {d_ee:.3e}   BPB {bpb_e1:.6f} vs {bpb_e2:.6f}", flush=True)
    if d_ee > 0:
        print("  [!] the harness is not deterministic; any candidate difference below is partly this",
              flush=True)

    set_impl(model, "grouped_mm")
    g, bpb_g = logits_and_bpb(model, ids, sl, D)
    d_eg = float((g - e1).abs().max())
    agree = float((g.argmax(-1) == e1.argmax(-1)).float().mean())
    print(f"  eager vs grouped_mm max|dlogit| {d_eg:.3e}   BPB {bpb_e1:.6f} vs {bpb_g:.6f} "
          f"(delta {bpb_g - bpb_e1:+.6f})", flush=True)
    print(f"  top-1 token agreement: {100*agree:.2f}%", flush=True)

    ok = abs(bpb_g - bpb_e1) < 1e-3 and agree > 0.999
    print(f"\n  VERDICT: grouped_mm is {'USABLE' if ok else 'NOT usable'} -- "
          f"{'BPB and ranking are preserved; the max-logit outlier is not material' if ok else 'it changes what the model computes'}",
          flush=True)
    print("=== GROUPED_MM CHECK COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
