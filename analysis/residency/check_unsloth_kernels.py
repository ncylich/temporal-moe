#!/usr/bin/env python3
"""Does Unsloth's grouped_mm expert path compute the same Qwen3-30B as our stock path?

TRAINING_OPTIM_PLAN.md step (a): kernel equivalence at step 0, residency OFF in both arms.
Same protocol as check_fused_kernels.py, which accepted the fused library at +9.60e-06 BPB
against a same-kernel noise floor of +6.26e-05, and rejected grouped_mm-via-our-wrapper at
-4.93e-04. Unsloth's path is torch._grouped_mm again, but their wrapper, on torch 2.13 --
so the old rejection does not transfer in either direction and the number must be re-measured.
Per explicit decision the BPB delta here is recorded, not vetoed at 1e-4.

Three configurations, sequential processes (unsloth patches transformers at import, so the
stock arm must never see it; and two 61 GB models cannot co-reside):

    --arm stock      residency_qwen installed; bs=1 and bs=2 (noise floor)
    --arm unsloth    zoo-patched model, base; then wrapped in zero-init LoRA r32 -- the
                     wrapper must be an exact numerical no-op, measured rather than assumed
    --arm compare    read both dumps, print deltas and the verdict

`--residency on` is plan step (c): the same three configurations with the constraint active
(R=k=8, every layer, min_logit evict; ours via residency_qwen, theirs via residency_unsloth).
Step (a) must pass first -- without it a mismatch here is ambiguous between bad kernel and
bad patch.

    for a in stock unsloth compare; do check_unsloth_kernels.py --arm $a [--residency on]; done
"""
import argparse
import json
import os
import sys

import torch

# Per-family paths; keyed like train_qwen.FAMILY. DUMP dirs are separate so a qwen3_5 run
# can never be compared against a qwen3 dump by accident.
PATHS = {
    "qwen3": {"data": "/workspace/qwen3moe-adapt/data", "sfx": "qwen3",
              "model": "/dev/shm/qwen3-30b",
              "dump": "/workspace/qwen3moe-adapt/results/unsloth_check"},
    "qwen3_5": {"data": "/workspace/qwen35-adapt/data", "sfx": "qwen",
                "model": "/workspace/qwen35-adapt/model",
                "dump": "/workspace/qwen35-adapt/results/unsloth_check"},
}
DATA = STOCK = DUMP = None                                # set from --model in main()


@torch.no_grad()
def measure(model, ids, divisor, n_seq, bs=1):
    """CE in nats/token -> BPB, plus per-position argmax, chunked to survive a 152k vocab.

    Argmax is reassembled per sequence, not appended flat -- at bs>1 a chunk's argmax is
    [bs, chunk] and flattening interleaves sequences (the bug that once produced a fake
    noise floor of 0.1385).
    """
    tot = ntok = 0
    per_seq = [[] for _ in range(n_seq)]
    for i in range(0, n_seq, bs):
        b = ids[i:i + bs].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            am = sl.argmax(-1).cpu()
            for j in range(am.shape[0]):
                per_seq[i + j].append(am[j])
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor, torch.cat([torch.cat(s) for s in per_seq])


def _set_residency(RES, on):
    if on:
        RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
        RES.set_free_layers(None)                       # every layer constrained
    else:
        RES._CFG.update(on=False, collect_telem=False)


def run_stock(ids, D, n_seq, res, sfx, family):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import residency as RES
    import residency_qwen as RQ
    from transformers import AutoModelForCausalLM
    _set_residency(RES, res)
    m = AutoModelForCausalLM.from_pretrained(STOCK, dtype=torch.bfloat16).to("cuda")
    RQ.install(family)
    RQ.tag_layers(m)
    m.eval()
    out = {}
    for tag, bs in (("stock", 1), ("stock_bs2", 2)):
        b, t1 = measure(m, ids, D, n_seq, bs=bs)
        out[tag] = {"bpb": b, "top1": t1}
        print(f"  {tag:14} BPB {b:.6f}   ({t1.numel():,} positions)", flush=True)
    torch.save(out, f"{DUMP}/stock{sfx}.pt")


def run_unsloth(ids, D, n_seq, res, sfx, family):
    import unsloth  # noqa: F401  must precede any transformers import
    from unsloth import FastModel
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import residency as RES
    m, _ = FastModel.from_pretrained(
        STOCK, max_seq_length=2048, dtype=torch.bfloat16,
        load_in_4bit=False, full_finetuning=False)
    if res:
        import residency_unsloth as RU
        n = RU.install(m)
        print(f"  residency_unsloth installed on {n} blocks", flush=True)
    _set_residency(RES, res)
    m.eval()
    out = {}
    b, t1 = measure(m, ids, D, n_seq, bs=1)
    out["unsloth"] = {"bpb": b, "top1": t1}
    print(f"  {'unsloth':14} BPB {b:.6f}   ({t1.numel():,} positions)", flush=True)
    # The training path adds a peft wrapper. LoRA B is zero-init so it is algebraically a
    # no-op; this measures whether the wrapper's plumbing (casts, hooks) preserves that.
    m = FastModel.get_peft_model(m, r=32, lora_alpha=64, lora_dropout=0.0)
    m.eval()
    b, t1 = measure(m, ids, D, n_seq, bs=1)
    out["unsloth_lora"] = {"bpb": b, "top1": t1}
    print(f"  {'unsloth_lora':14} BPB {b:.6f}   ({t1.numel():,} positions)", flush=True)
    torch.save(out, f"{DUMP}/unsloth{sfx}.pt")


def compare(sfx):
    s = torch.load(f"{DUMP}/stock{sfx}.pt", weights_only=False)
    u = torch.load(f"{DUMP}/unsloth{sfx}.pt", weights_only=False)
    all_ = {**s, **u}

    def cmp(a, c):
        return all_[c]["bpb"] - all_[a]["bpb"], \
            float((all_[c]["top1"] == all_[a]["top1"]).float().mean())

    d_fl, ag_fl = cmp("stock", "stock_bs2")
    print("\n  === bf16 noise floor (stock kernel, bs=1 vs bs=2) ===")
    print(f"  BPB delta    {d_fl:+.6e}     top-1 agree  {ag_fl:.4f}")
    # On torch 2.13 the stock path is batch-size invariant, so the bs floor saturates at
    # 1.0000 and stops measuring anything. The operative floor is then cross-environment:
    # the same stock code under torch 2.4 vs 2.13 (measured 2026-08-05, residency off:
    # BPB +1.46e-04, top-1 0.9776 -- same math, different reduction orders). Produced by
    # running --arm stock under the torch 2.4 venv and renaming the dump. The suffix keeps
    # residency states matched: under the constraint, logit-level noise flips swap decisions
    # and cascades, so the on-floor is legitimately far looser than the off-floor and the
    # two must never be conflated.
    p24 = f"{DUMP}/stock_torch24{sfx}.pt"
    t24 = None
    if ag_fl > 0.999 and os.path.exists(p24):
        t24 = torch.load(p24, weights_only=False)
        all_["stock_torch24"] = t24["stock"]
        d_fl, ag_fl = cmp("stock_torch24", "stock")
        print("\n  === operative floor: same stock code, torch 2.4 vs 2.13 ===")
        print(f"  BPB delta    {d_fl:+.6e}     top-1 agree  {ag_fl:.4f}")
    for tag in ("unsloth", "unsloth_lora"):
        d, ag = cmp("stock", tag)
        print(f"\n  === {tag} vs stock ===")
        print(f"  BPB          {all_['stock']['bpb']:.6f} -> {all_[tag]['bpb']:.6f}   "
              f"delta {d:+.6e}")
        print(f"  top-1 agree  {ag:.4f}")
    d_lw, ag_lw = cmp("unsloth", "unsloth_lora")
    print(f"\n  === peft wrapper alone (unsloth base vs +zero-init LoRA, same kernels) ===")
    print(f"  BPB delta    {d_lw:+.6e}     top-1 agree  {ag_lw:.4f}")
    print("\n  Reference: fused lib ACCEPTED at +9.60e-06; grouped_mm-via-our-wrapper "
          "REJECTED at -4.93e-04.")
    print("  Effects measured downstream: aux correction 4.85e-04 BPB, free-set spread ~2.5e-03.")
    # Under the constraint the system is chaotic: any epsilon perturbation (a torch version,
    # a batch size, the peft wrapper's casts) flips swap decisions and ~21% of argmaxes. The
    # BPB tolerance must then be the measured trajectory noise of the SAME kernel under a
    # reduction-order perturbation -- bs=1 vs bs=2 in the batch-variant torch 2.4 env -- not
    # the residency-off threshold. Consequence for use: implementations carry O(noise) BPB
    # offsets, so every downstream comparison (trained vs null vs baseline) must be measured
    # within ONE implementation; cross-arm numbers must never be differenced.
    tol = 2e-4
    if t24 is not None and "stock_bs2" in t24:
        d_tn = t24["stock_bs2"]["bpb"] - t24["stock"]["bpb"]
        print(f"  Same-kernel trajectory noise (torch 2.4, bs1 vs bs2): {d_tn:+.6e}")
        tol = max(tol, abs(d_tn))
    d, ag = cmp("stock", "unsloth_lora")
    if ag_fl > 0.999 and t24 is None:
        # No same-math floor exists for this model (the torch 2.4 venv cannot run it), and
        # the bs floor is saturated. A binary verdict here would be judged against nothing;
        # print the numbers and defer to the qwen3-measured floors (top-1 0.9776,
        # BPB +/-1.5e-04 off / trajectory 1.4e-03 on) recorded in unsloth_parity.md.
        print(f"\n  VERDICT: NO-FLOOR -- unsloth+LoRA shifts BPB {d:+.2e}, top-1 {ag:.4f}; "
              f"no same-math floor for this model, judge against the qwen3 floors "
              f"(results/ablations/unsloth_parity.md)")
    else:
        ok = abs(d) < tol and ag >= ag_fl - 0.01
        print(f"\n  VERDICT: {'ACCEPT' if ok else 'REJECT'} -- full training config "
              f"(unsloth+LoRA) shifts BPB {d:+.2e} vs tolerance {tol:.2e}; "
              f"top-1 {ag:.4f} vs floor {ag_fl:.4f}")
    print("=== UNSLOTH KERNEL CHECK COMPLETE ===", flush=True)


def main():
    global DATA, STOCK, DUMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("stock", "unsloth", "compare"))
    ap.add_argument("--model", default="qwen3", choices=("qwen3", "qwen3_5"))
    ap.add_argument("--residency", default="off", choices=("off", "on"))
    ap.add_argument("--n-seq", type=int, default=16)
    A = ap.parse_args()
    P = PATHS[A.model]
    DATA, STOCK, DUMP = P["data"], P["model"], P["dump"]
    os.makedirs(DUMP, exist_ok=True)
    res = A.residency == "on"
    sfx = "_res" if res else ""
    if A.arm == "compare":
        compare(sfx)
        return
    D = json.load(open(f"{DATA}/bpb_slice_meta_{P['sfx']}.json"))["divisor_D"]
    ids = torch.load(f"{DATA}/bpb_slice_ids_{P['sfx']}.pt", weights_only=False)[: A.n_seq]
    (run_stock if A.arm == "stock" else run_unsloth)(ids, D, A.n_seq, res, sfx, A.model)


if __name__ == "__main__":
    main()
