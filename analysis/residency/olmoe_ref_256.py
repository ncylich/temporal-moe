#!/usr/bin/env python3
"""OLMoE reference BPB on the 256-sequence subset the trained runs are actually scored on.

train_ple.py:204 evaluates on `bpb_ids[torch.linspace(0, n-1, 256)]` -- 256 sequences spread evenly
across the whole 24,414-sequence held-out slice. olmoe_remeasure.py measured the untrained free and
constrained references on `ids[:16]`, the first 16 sequences. Those are different evaluation sets, so
subtracting one from the other is not a recovery measurement: the free baseline alone varies 0.6359
to 0.6822 across disjoint blocks of this slice, roughly 40x the effects being claimed.

This re-measures both references on the SAME subset the trained cell used, so the trained number
0.804438 can be differenced against them.

Both arms use gate_mass="preserve", matching how the trained cell was run.

    olmoe_ref_256.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
from olmoe_paths import DATA_DIR                                   # noqa: E402


@torch.no_grad()
def bpb(model, ids, divisor):
    tot = ntok = 0
    for i in range(ids.shape[0]):
        b = ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor


def main():
    D = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))["divisor_D"]
    allids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt", weights_only=False)
    # Exactly train_ple.py's selection, so the trained number is differenced against like for like.
    sub = allids[torch.linspace(0, allids.shape[0] - 1, 256).long()]
    model, _ = RES.load_model()
    L = model.config.num_hidden_layers
    print(f"  OLMoE, {sub.shape[0]} seqs (train_ple.py's eval subset), gate_mass=preserve", flush=True)

    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False, gate_mass="preserve")
    RES.set_free_layers(list(range(L)))                 # residency inert
    free = bpb(model, sub, D)
    print(f"  untrained, unconstrained   BPB {free:.6f}", flush=True)

    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False, gate_mass="preserve")
    RES.set_free_layers([])                             # constrained on all 16 layers
    con = bpb(model, sub, D)
    print(f"  untrained, R=8 of 64       BPB {con:.6f}", flush=True)

    trained = 0.8044375975325939                        # ple_ce_attn_nofree_50M.json
    print(f"\n  trained 50M (CE + attn LoRA) BPB {trained:.6f}")
    print(f"  constraint cost untrained    {con-free:+.6f}")
    print(f"  constraint cost after training{trained-free:+.6f}")
    denom = con - free
    if denom > 0:
        print(f"  recovery vs untrained ceiling {(con-trained)/denom*100:.1f}%")
    print(f"\n  NOTE: measured against the UNTRAINED unconstrained model. The matched trained null\n"
          f"  (ce_freeall_50M, 0.695064 on its own eval subset) is the achievable ceiling and would\n"
          f"  raise this figure; it is not differenced here because it was scored on the 256-subset\n"
          f"  too and belongs in the same table rather than in this script's arithmetic.")
    print("=== OLMOE REF COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
