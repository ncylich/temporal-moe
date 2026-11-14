#!/usr/bin/env python3
"""Does the fused library compute the same model as stock transformers?

`grouped_mm` was rejected by exactly this test: 1.004x speed, 7% top-1 disagreement and a BPB shift
of -4.93e-04 (results/ablations/qwen_expert_kernels.csv). The fused library was then adopted for its
claimed 22.7x speedup (since withdrawn -- see results/ablations/crossmodel_RESULTS.md S9) without the test being re-applied,
which is the same mistake with a bigger
blast radius -- every Qwen retrain number would flow through its Triton grouped-GEMM kernels.

The bar is set by the effects being measured, not by "close": the aux-loss correction was 4.85e-04
BPB and free-set differences run ~2.5e-03. A kernel that shifts BPB by 5e-04 would be
indistinguishable from a real free-set effect, so ACCEPT requires the shift to be well under that.

Both arms run with residency OFF (every layer freed), because this compares the expert kernels and
not the constraint. Sequential in one process with an explicit free between: two 57 GB bf16 copies do
not co-reside on an 80 GB card.

    check_fused_kernels.py --n-seq 16
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402

DATA = "/workspace/qwen3moe-adapt/data"
STOCK = "/dev/shm/qwen3-30b"
FUSED = "/root/models/qwen3-30b-fused"


@torch.no_grad()
def measure(model, ids, divisor, n_seq, bs=1):
    """CE in nats/token and the argmax prediction per position, chunked to survive a 152k vocab.

    `bs` exists to establish a noise floor. Scoring the same model on the same tokens at a different
    batch size changes GEMM shapes and therefore bf16 reduction order, so top-1 disagreement between
    bs=1 and bs=2 on ONE kernel is the amount of argmax churn that near-ties produce for free. Without
    it, a top-1 agreement of 0.978 between two kernels is uninterpretable -- it could be drift or it
    could be the floor.
    """
    tot = ntok = 0
    # Keyed by sequence, not appended flat: at bs>1 a chunk's argmax is [bs, chunk] and flattening it
    # interleaves sequences, so a flat list compares position p of seq 0 against position p of seq 1.
    # That is what produced a "noise floor" of 0.1385 -- an indexing artifact of this harness, not a
    # property of the kernels. Reassembling per sequence makes the two orderings comparable.
    per_seq = [[] for _ in range(n_seq)]
    for i in range(0, n_seq, bs):
        b = ids[i:i + bs].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            am = sl.argmax(-1).cpu()                        # [bs, chunk]
            for j in range(am.shape[0]):
                per_seq[i + j].append(am[j])
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor, torch.cat([torch.cat(s) for s in per_seq])


def load_stock(path):
    """transformers' expert math under our block wrapper -- which is what 'stock' has always meant.

    transformers 5.12's own Qwen3MoeSparseMoeBlock.forward returns (hidden_states, router_logits)
    while its decoder layer at modeling_qwen3_moe.py:352 does `residual + hidden_states`, so the
    shipped model cannot run a forward pass unpatched (TypeError: Tensor + tuple). Every Qwen number
    in this programme, including the 0.582025 free-arm BPB this check compares against, was produced
    with residency_qwen installed and residency off. Installing it here reproduces that baseline
    rather than inventing a third configuration.
    """
    import residency_qwen as RQ
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
    RQ.install("qwen3")
    RQ.tag_layers(m)
    return m.eval()


def load_fused(path):
    sys.path.insert(0, "/workspace/qwen3-moe-fused")
    import residency_fused as RF
    from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
    m = Qwen3MoeFusedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
    RF.install(m)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=16)
    A = ap.parse_args()

    D = json.load(open(f"{DATA}/bpb_slice_meta_qwen3.json"))["divisor_D"]
    ids = torch.load(f"{DATA}/bpb_slice_ids_qwen3.pt", weights_only=False)[: A.n_seq]
    RES._CFG.update(on=False, collect_telem=False)          # residency off in both arms

    out = {}
    torch.cuda.empty_cache()
    model = load_stock(STOCK)
    for tag, bs in (("stock", 1), ("stock_bs2", 2)):        # same kernel, different reduction order
        b, t1 = measure(model, ids, D, A.n_seq, bs=bs)
        out[tag] = (b, t1)
        print(f"  {tag:10} BPB {b:.6f}   ({t1.numel():,} positions scored)", flush=True)
    del model
    torch.cuda.empty_cache()
    model = load_fused(FUSED)
    b, t1 = measure(model, ids, D, A.n_seq, bs=1)
    out["fused"] = (b, t1)
    print(f"  {'fused':10} BPB {b:.6f}   ({t1.numel():,} positions scored)", flush=True)
    del model
    torch.cuda.empty_cache()

    def cmp(a, c):
        return out[c][0] - out[a][0], float((out[c][1] == out[a][1]).float().mean())

    d_fl, ag_fl = cmp("stock", "stock_bs2")                 # noise floor: one kernel, two orderings
    d, agree = cmp("stock", "fused")
    print(f"\n  === bf16 noise floor (stock kernel, bs=1 vs bs=2) ===")
    print(f"  BPB delta    {d_fl:+.6e}     top-1 agree  {ag_fl:.4f}")
    print(f"\n  === fused Triton kernels vs stock ===")
    print(f"  BPB          {out['stock'][0]:.6f} -> {out['fused'][0]:.6f}   delta {d:+.6e}")
    print(f"  top-1 agree  {agree:.4f}")
    print(f"\n  Reference: grouped_mm was REJECTED at bpb_delta -4.93e-04, top1 0.9316.")
    print(f"  Effects being measured: aux correction 4.85e-04 BPB, free-set spread ~2.5e-03 BPB.")
    # BPB is the acceptance criterion: it is the quantity every result in this programme is stated in,
    # and it is what grouped_mm failed. Top-1 agreement is interpreted RELATIVE to the floor -- argmax
    # churn on near-ties is free under any change of reduction order, so an absolute threshold on it
    # (the 0.99 used in the first version of this script) has no evidential basis.
    ok = abs(d) < 2e-4 and agree >= ag_fl - 0.01
    print(f"\n  VERDICT: {'ACCEPT' if ok else 'REJECT'} -- BPB shift {abs(d):.2e} is "
          f"{'well under' if abs(d) < 2e-4 else 'comparable to'} the effects measured; top-1 "
          f"{agree:.4f} vs a same-kernel floor of {ag_fl:.4f}")
    print("=== FUSED KERNEL CHECK COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
