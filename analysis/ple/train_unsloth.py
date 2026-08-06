#!/usr/bin/env python3
"""Residency adaptation on the Unsloth grouped_mm path -- the sweep trainer for both Qwen models.

The accepted configuration (results/ablations/unsloth_parity.md): FastModel bf16 + peft LoRA
(experts auto-detected + attention) cast to bf16, router gates + RMSNorm gains unfrozen,
residency via residency_unsloth (step-0 parity accepted at both residency states), AdamW8bit
+ cut_cross_entropy. 10.5x / ~15.4x over the stock expert loop at matched config.

Everything that decides a number is shared with train_qwen.py: FAMILY paths, the corpus
reshape, the eval protocol (`train_qwen.evaluate` on the audited slice), and the aux/z
per-layer formula (residency.aux_z_from_router_logits verbatim, computed inside the block
forward by residency_unsloth and injected Megatron-style -- see _AuxInject there for why:
every no-grad-outer-forward checkpointing mode detaches post-forward aux, and unsloth's
offloaded checkpointing, whose memory profile the probes validated, is such a mode).
Numbers from this trainer must only ever be differenced against numbers from this trainer
-- implementations carry O(1e-03) BPB offsets under the constraint (unsloth_parity.md).
Step 0 fatally checks that the router gate weight receives a non-zero gradient: the LM
loss cannot reach it (top-k carries no gradient), so a live gate grad proves the injection.

Env (set by the driver): UNSLOTH_MOE_DISABLE_AUTOTUNE=1 always (autotuners bench at peak
memory and OOM a full card; fla cache is warmed separately); UNSLOTH_COMPILE_DISABLE=1 for
qwen3_5 (CUDA-graph pools hold ~7.4 GB); PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

    train_unsloth.py --tag sweep_lr3e-4 --family qwen3 --lr 3e-4 --lora 32 --tokens 15000000
"""
import argparse
import json
import os
import sys
import time

import unsloth  # noqa: F401  must precede any transformers import
from unsloth import FastModel
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_unsloth as RU                                     # noqa: E402
import train_qwen as TQ                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--family", default="qwen3", choices=("qwen3", "qwen3_5"))
    ap.add_argument("--aux-c", type=float, default=None,
                    help="None = the model's shipped router_aux_loss_coef")
    ap.add_argument("--z-c", type=float, default=0.001)
    ap.add_argument("--tokens", type=int, default=15_000_000)
    ap.add_argument("--free-set", default="", help="comma list; 'all' = unconstrained null")
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mb", type=int, default=0, help="0 = family default (qwen3:2, qwen3_5:1)")
    ap.add_argument("--eval-every", type=int, default=5_000_000)
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    A = ap.parse_args()
    if A.mb == 0:
        A.mb = 2 if A.family == "qwen3" else 1
    A.accum = max(1, 16_384 // (A.mb * A.seq))          # 16,384 tok/step, matched across models

    FAM = TQ.resolve(A.family)
    DATA, OUT, SFX = FAM["data"], FAM["out"], FAM["suffix"]
    os.makedirs(OUT, exist_ok=True)
    D = json.load(open(f"{DATA}/bpb_slice_meta_{SFX}.json"))["divisor_D"]

    model, _ = FastModel.from_pretrained(
        FAM["model"], max_seq_length=A.seq, dtype=torch.bfloat16,
        load_in_4bit=False, full_finetuning=False)
    for mod in model.modules():                          # Qwen3.5's unused ~8 GB vision tower
        if getattr(mod, "visual", None) is not None and "Vision" in type(mod.visual).__name__:
            mod.visual = None
            torch.cuda.empty_cache()
            break
    # unsloth's offloaded checkpointing: the probe-validated memory profile. Its outer
    # forward runs under no-grad, which kills any post-forward aux computed from captured
    # logits -- so the aux/z losses are instead injected INSIDE the block forward
    # (residency_unsloth._AuxInject, Megatron-style), which survives every checkpointing
    # mode. HF non-reentrant checkpointing was tried and OOMs: unsloth's replaced forwards
    # do not respect the HF flag, so activations for all 48 layers stayed live.
    model = FastModel.get_peft_model(
        model, r=A.lora, lora_alpha=2 * A.lora, lora_dropout=0.0,
        use_gradient_checkpointing="unsloth")

    nblk = RU.install(model)
    if A.aux_c is None:
        _tc = getattr(model.config, "text_config", model.config)
        A.aux_c = float(getattr(_tc, "router_aux_loss_coef",
                                getattr(model.config, "router_aux_loss_coef", 0.01)))
        print(f"  [aux] using the model's shipped router_aux_loss_coef = {A.aux_c}", flush=True)

    L = getattr(model.config, "text_config", model.config).num_hidden_layers
    if A.free_set == "all":
        free = list(range(L))
    elif A.free_set:
        free = [int(x) for x in A.free_set.split(",")]
    else:
        free = None
    RES._CFG.update(on=True, R=A.R, evict="min_logit", collect_telem=True)
    RES.set_free_layers(free)

    # peft froze everything outside the adapters; the sweep's fixed surface also trains
    # router gates and RMSNorm gains. bf16 adapters throughout (fp32 OOMs this surface).
    n_extra = 0
    for name, p in model.named_parameters():
        if name.endswith(".gate.weight") or "norm" in name.split(".")[-2]:
            p.requires_grad_(True)
            n_extra += 1
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    n_tr = sum(p.numel() for p in params)
    print(f"  [unsloth-train] tag={A.tag} family={A.family} blocks={nblk} "
          f"free={A.free_set or 'none'} R={A.R} lora=r{A.lora} trainable={n_tr/1e6:.1f}M "
          f"(+{n_extra} router/norm tensors) mb={A.mb} accum={A.accum}", flush=True)

    causal = model.base_model.model

    corpus = torch.load(f"{DATA}/finetune_ids_{SFX}.pt", weights_only=False)
    if A.seq < corpus.shape[1]:                          # reshape: every token reachable
        nper = corpus.shape[1] // A.seq
        corpus = corpus[:, :nper * A.seq].reshape(-1, A.seq)
    epochs = A.tokens / corpus.numel()
    print(f"  [data] {corpus.shape[0]} x {A.seq} rows; {A.tokens/1e6:.0f}M tokens = "
          f"{epochs:.2f} epochs{' WARNING: >1 epoch' if epochs > 1 else ''}", flush=True)
    bpb_ids = torch.load(f"{DATA}/bpb_slice_ids_{SFX}.pt", weights_only=False)[: A.eval_seq]

    import bitsandbytes as bnb
    from cut_cross_entropy import linear_cross_entropy
    opt = bnb.optim.AdamW8bit(params, lr=A.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train()
    # Aux/z scales for the in-forward injection: residency.aux_z_from_router_logits averages
    # per-layer values over the L MoE layers and the trainer divides the total loss by accum,
    # so each layer's injected gradient carries coeff / (L * accum).
    RES._CFG["aux_inject"] = {"aux": A.aux_c / (nblk * A.accum), "z": A.z_c / (nblk * A.accum)}
    steps = A.tokens // (A.mb * A.accum * A.seq)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=max(1, steps),
                                                pct_start=0.02, anneal_strategy="cos")


    hist, seen, t0, nxt, ptr = [], 0, time.time(), A.eval_every, 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        aux_vals = []
        for _ in range(A.accum):
            b = corpus[ptr:ptr + A.mb, :A.seq].to("cuda").long()
            ptr = 0 if ptr + 2 * A.mb > len(corpus) else ptr + A.mb
            RU.AUX_LOG.clear()
            h = causal.model(b)[0]                       # logits never materialised (CCE)
            lm = linear_cross_entropy(h, causal.lm_head.weight, b, shift=True)
            lm.div(A.accum).backward()                   # aux/z arrive via _AuxInject
            aux_vals.extend(RU.AUX_LOG)
            seen += b.numel()
            del h
        RU.AUX_LOG.clear()
        if step == 0:
            # A gate-grad check cannot prove the injection (the LM loss also reaches the
            # gate through the softmax routing weights); _INJ_FIRED is set inside
            # _AuxInject.backward itself, so it proves the aux backward path ran.
            if not RU._INJ_FIRED:
                sys.exit("FATAL: _AuxInject.backward never fired -- the aux/z injection "
                         "is dead. Do not train.")
            print(f"  [guard] aux injection alive ({len(aux_vals)} layer-values logged)",
                  flush=True)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            a_m = sum(v[0] for v in aux_vals) / max(1, len(aux_vals))
            print(f"  step {step}/{steps} tok={seen/1e6:.1f}M lm={float(lm.detach()):.4f} "
                  f"aux={a_m:.3f} {(time.time()-t0)/60:.1f}min", flush=True)
        if seen >= nxt or step == steps - 1:
            # Free last step's gradients (~3.8 GB, already consumed by opt.step) and
            # defragment before eval: evaluate() materialises full logits, 1.0 GB bf16 +
            # a 0.5 GB fp32 chunk on Qwen3.5's 248k vocab -- exactly what OOM'd the first
            # 5M eval. The eval protocol itself is unchanged.
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            model.eval()
            bpb, swap, ent = TQ.evaluate(model, bpb_ids, D, A.mb)
            model.train()
            hist.append({"tok": seen, "bpb": bpb, "swap": swap})
            print(f"[eval] {A.tag} tok={seen/1e6:.0f}M BPB={bpb:.6f} swap={swap:.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
            nxt += A.eval_every
            if A.max_minutes and (time.time() - t0) / 60 >= A.max_minutes:
                print(f"  [cap] stopping at an eval boundary", flush=True)
                break

    opt.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    model.eval()
    bpb, swap, ent = TQ.evaluate(model, bpb_ids, D, A.mb)
    res = {"tag": A.tag, "family": A.family, "harness": "unsloth", "final_bpb": bpb,
           "final_swap": swap, "free_set": A.free_set, "R": A.R, "lora": A.lora,
           "tokens": seen, "divisor": D, "curve": hist, "trainable": n_tr,
           "aux_c": A.aux_c, "z_c": A.z_c, "lr": A.lr, "opt": "adamw8bit", "ce": "cce",
           "minutes": (time.time() - t0) / 60}
    try:
        sd = {n: p.detach().to(torch.bfloat16).cpu()
              for n, p in model.named_parameters() if p.requires_grad}
        ck = os.path.join(OUT, f"unsloth_{A.tag}_adapter.pt")
        torch.save({"tensors": sd, "family": A.family, "lora": A.lora,
                    "free_set": A.free_set, "R": A.R, "tokens": seen}, ck)
        res["adapter"] = ck
        print(f"  [save] {ck} ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)", flush=True)
    except Exception as e:
        print(f"  [save] FAILED {type(e).__name__}: {e}", flush=True)
    with open(os.path.join(OUT, f"unsloth_{A.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[DONE] {A.tag} final BPB={bpb:.6f} swap={swap:.4f} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
