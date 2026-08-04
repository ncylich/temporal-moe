#!/usr/bin/env python3
"""Adapt Qwen3.5-35B-A3B-Base under rolling residency, with the matched unconstrained null.

Three arms, identical in tokens, LoRA budget, optimiser, schedule and data order. Only the residency
configuration differs, which is the whole point: with the corpus and the adapter held fixed, any BPB
difference between arms is attributable to the constraint rather than to continued training.

    ce            residency on every MoE layer, expert LoRA          (the constrained surface)
    ce_free_attn  residency off on {0,1,38,39}, expert + attn LoRA   (the recipe that won on OLMoE)
    null          residency off on EVERY layer, expert LoRA          (continual training, no constraint)

The null is not optional bookkeeping. On OLMoE the same arm moved BPB from 0.6727 to 0.6905 over 30M
tokens -- continual training on this corpus makes the *unconstrained* model worse -- so a recovery
percentage measured against the untrained checkpoint credits the constraint for damage the training
recipe caused. Without this arm the other two cannot be read.

Expert LoRA is applied by patching the experts forward, because Qwen stores expert weights as 3-D
tensors indexed in a Python loop rather than as Linear modules, so a module hook has nothing to
attach to. Attention LoRA does use a hook, and lands on the 10 full-attention layers only: 30 of the
40 layers are Gated DeltaNet, which has no q/k/v/o to adapt. That asymmetry is real and is why the
attention arm is not the same intervention it was on OLMoE.

    train_qwen.py --tag ce --tokens 30000000
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                             # noqa: E402
import residency_qwen as RQ                                         # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (   # noqa: E402
    Qwen3_5MoeExperts, Qwen3_5MoeAttention,
)

DATA = "/workspace/qwen35-adapt/data"
OUT = "/workspace/qwen35-adapt/results"
SEQ, AUX_C, Z_C = 4096, 0.01, 0.001
_LORA = {"scale": 2.0}
_orig_experts_forward = None


def _experts_forward_lora(self, hidden_states, top_k_index, top_k_weights):
    """Stock Qwen expert loop with a zero-initialised LoRA branch on each expert's two projections.

    B is zero at init so the branch contributes exactly nothing at step 0; that keeps flag-off parity
    exact and means any divergence from the base model is learned, not an artefact of attaching the
    adapter.
    """
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    s = _LORA["scale"]
    for e in hit:
        e = e[0]
        if e == self.num_experts:
            continue
        pos, tok = torch.where(expert_mask[e])
        x = hidden_states[tok]
        gu = F.linear(x, self.gate_up_proj[e])
        gu = gu + F.linear(F.linear(x, self.lora_gu_A[e]), self.lora_gu_B[e]) * s
        gate, up = gu.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        y = F.linear(h, self.down_proj[e])
        y = y + F.linear(F.linear(h, self.lora_dn_A[e]), self.lora_dn_B[e]) * s
        y = y * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, y.to(final.dtype))
    return final


def add_lora(model, r=32, alpha=64):
    global _orig_experts_forward
    _LORA["scale"] = alpha / r
    ps = []
    for m in model.modules():
        if not isinstance(m, Qwen3_5MoeExperts):
            continue
        E, twoI, H = m.gate_up_proj.shape
        _, Hd, I = m.down_proj.shape
        dev = m.gate_up_proj.device

        def mk(shape, zero):
            t = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
            if not zero:
                torch.nn.init.normal_(t, std=1.0 / r)
            return torch.nn.Parameter(t)
        m.lora_gu_A = mk((E, r, H), False); m.lora_gu_B = mk((E, twoI, r), True)
        m.lora_dn_A = mk((E, r, I), False); m.lora_dn_B = mk((E, Hd, r), True)
        ps += [m.lora_gu_A, m.lora_gu_B, m.lora_dn_A, m.lora_dn_B]
    if _orig_experts_forward is None:
        _orig_experts_forward = Qwen3_5MoeExperts.forward
    Qwen3_5MoeExperts.forward = _experts_forward_lora
    return ps


def _hook(mod, inp, out):
    x = inp[0]
    return out + F.linear(F.linear(x, mod.lora_A), mod.lora_B) * mod.lora_s


def add_lora_attn(model, r=32, alpha=64, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    """LoRA on the full-attention layers. Gated DeltaNet layers are skipped -- they have no q/k/v/o."""
    ps, n = [], 0
    for m in model.modules():
        if not isinstance(m, Qwen3_5MoeAttention):
            continue
        for name in targets:
            lin = getattr(m, name, None)
            if lin is None:
                continue
            dev, dt = lin.weight.device, lin.weight.dtype
            lin.lora_A = torch.nn.Parameter(torch.zeros(r, lin.in_features, device=dev, dtype=dt))
            torch.nn.init.normal_(lin.lora_A, std=1.0 / r)
            lin.lora_B = torch.nn.Parameter(torch.zeros(lin.out_features, r, device=dev, dtype=dt))
            lin.lora_s = alpha / r
            lin.register_forward_hook(_hook)
            ps += [lin.lora_A, lin.lora_B]; n += 1
    print(f"  [lora-attn] {n} projections across the full-attention layers", flush=True)
    return ps


@torch.no_grad()
def evaluate(model, ids, divisor, mb=1):
    model.eval()
    RES.reset_telem()
    tot = ntok = 0
    for i in range(0, len(ids), mb):
        b = ids[i:i + mb].to("cuda").long()
        lg = model(b).logits[:, :-1].float()
        tg = b[:, 1:]
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="sum"))
        ntok += tg.numel()
        del lg
    model.train()
    swap, ent = RES.telem_summary(model.config.num_experts)
    return (tot / ntok) / divisor, swap, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=30_000_000)
    ap.add_argument("--free-set", default="", help="comma list; 'all' = unconstrained null")
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--lora-attn", type=int, default=0)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=10_000_000)
    ap.add_argument("--eval-seq", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="hard wall-clock cap; the run stops at the next eval boundary and still "
                         "writes its JSON, so arms stay comparable at a common checkpoint")
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    meta = json.load(open(f"{DATA}/bpb_slice_meta_qwen.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model()
    L = model.config.num_hidden_layers

    if A.free_set == "all":
        free = list(range(L))
    elif A.free_set:
        free = [int(x) for x in A.free_set.split(",")]
    else:
        free = None
    RES._CFG.update(on=True, R=A.R, evict="min_logit", collect_telem=True)
    RES.set_free_layers(free)

    for p in model.parameters():
        p.requires_grad_(False)
    params = add_lora(model, r=A.lora)
    if A.lora_attn:
        params += add_lora_attn(model, r=A.lora_attn)
    for p in params:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in params)
    print(f"  [qwen-train] tag={A.tag} free={A.free_set or 'none'} R={A.R} trainable={n_tr/1e6:.1f}M "
          f"tokens={A.tokens/1e6:.0f}M divisor={D:.7f}", flush=True)

    corpus = torch.load(f"{DATA}/finetune_ids_qwen.pt", weights_only=False)
    bpb_ids = torch.load(f"{DATA}/bpb_slice_ids_qwen.pt", weights_only=False)[: A.eval_seq]
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.gradient_checkpointing_enable()
    model.train()

    tok_per_step = A.mb * A.accum * SEQ
    steps = A.tokens // tok_per_step
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=max(1, steps),
                                                pct_start=0.02, anneal_strategy="cos")
    hist, seen, t0, nxt = [], 0, time.time(), A.eval_every
    ptr = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(A.accum):
            b = corpus[ptr:ptr + A.mb].to("cuda").long(); ptr += A.mb
            if ptr + A.mb > len(corpus):
                ptr = 0
            RQ.capture(True)
            out = model(b)
            rl = RQ.captured(); RQ.capture(False)
            lg = out.logits[:, :-1]
            lm = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), b[:, 1:].reshape(-1))
            aux, z = RES.aux_z_from_router_logits(rl, b.shape[0], b.shape[1], A.R)
            (lm + AUX_C * aux + Z_C * z).div(A.accum).backward()
            seen += b.numel()
            del out, lg, rl
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            print(f"  step {step}/{steps} tok={seen/1e6:.1f}M lm={float(lm):.4f} aux={float(aux):.3f} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
        if seen >= nxt or step == steps - 1:
            bpb, swap, ent = evaluate(model, bpb_ids, D, A.mb)
            hist.append({"tok": seen, "bpb": bpb, "swap": swap})
            print(f"[eval] {A.tag} tok={seen/1e6:.0f}M BPB={bpb:.6f} swap={swap:.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
            nxt += A.eval_every
            if A.max_minutes and (time.time() - t0) / 60 >= A.max_minutes:
                print(f"  [cap] {A.max_minutes:.0f}min reached at {seen/1e6:.1f}M tokens; stopping "
                      f"at an eval boundary so this arm stays comparable to the others", flush=True)
                break
    bpb, swap, ent = evaluate(model, bpb_ids, D, A.mb)
    res = {"tag": A.tag, "final_bpb": bpb, "final_swap": swap, "free_set": A.free_set,
           "R": A.R, "lora": A.lora, "lora_attn": A.lora_attn, "tokens": seen,
           "divisor": D, "curve": hist, "trainable": n_tr,
           "minutes": (time.time() - t0) / 60}
    with open(os.path.join(OUT, f"qwen_{A.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[DONE] {A.tag} final BPB={bpb:.6f} swap={swap:.4f} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
