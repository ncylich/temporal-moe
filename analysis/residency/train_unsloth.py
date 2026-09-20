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
    # Recipe ablations (SWEEP follow-up). --distill: replace the CE objective with KL to the
    # pristine unconstrained base (teacher = same weights, adapters disabled, residency off,
    # router/norms restored to their initial values for the teacher pass) -- optimises exactly
    # what recovery measures; chunked-checkpointed KL, so it runs at the family's native mb.
    # --r-anneal F: anneal R from E (free) to the target over the first F of optimiser steps.
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--r-anneal", type=float, default=0.0)
    ap.add_argument("--anneal-style", default="linear", choices=("linear", "damage"),
                    help="linear: R falls linearly E/4->R over the window. damage: transit "
                         "each segment at time proportional to its measured training-free "
                         "damage share (qwen3 R-curve 2026-08-07, 32->8 total 0.0930 BPB: "
                         "32->16 34.2%%, 16->12 22.8%%, 12->10 16.1%%, 10->8 26.9%%)")
    ap.add_argument("--distill-T", type=float, default=1.0,
                    help="distillation temperature; loss scaled by T^2 so gradient magnitude "
                         "is T-invariant and one LR bracket serves all T")
    ap.add_argument("--evict", default="min_logit", choices=("min_logit", "lru"))
    # paged8bit: SAME 8-bit block-quantised Adam states as adamw8bit, held in CUDA unified
    # memory -- frees ~3.8 GB of committed VRAM (the difference that lets Qwen3.5 run mb2)
    # at a small paging cost once per optimiser step, amortised over accum.
    ap.add_argument("--opt", default="adamw8bit", choices=("adamw8bit", "paged8bit"))
    # Rolling cached teacher (user design 2026-08-08): every segment of N tokens, generate
    # the teacher's top-K targets for the NEXT min(N, remaining) tokens in one fat-batched
    # no-grad sweep (weights swapped ONCE per segment, cache held in host RAM), then train
    # the student over that segment at CE-like speed reading cached targets. 0 = inline.
    ap.add_argument("--teacher-cache-tokens", type=int, default=0)
    ap.add_argument("--teacher-topk", type=int, default=256)
    ap.add_argument("--teacher-mb", type=int, default=8)
    A = ap.parse_args()
    # Distillation runs the chunked-checkpointed KL (gradient-identical to full KL,
    # validated), which never materialises the [B,S,V] logits -- so the micro-batch no
    # longer needs to drop to 1. Family-default mb applies, restoring ~1.8x throughput
    # on qwen3 (its forced-mb1 era covered the 15M screens/brackets; recorded).
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
    RES._CFG.update(on=True, R=A.R, evict=A.evict, collect_telem=True)
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
    _Opt = bnb.optim.PagedAdamW8bit if A.opt == "paged8bit" else bnb.optim.AdamW8bit
    opt = _Opt(params, lr=A.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train()
    # Aux/z scales for the in-forward injection: residency.aux_z_from_router_logits averages
    # per-layer values over the L MoE layers and the trainer divides the total loss by accum,
    # so each layer's injected gradient carries coeff / (L * accum).
    RES._CFG["aux_inject"] = {"aux": A.aux_c / (nblk * A.accum), "z": A.z_c / (nblk * A.accum)}
    steps = A.tokens // (A.mb * A.accum * A.seq)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=max(1, steps),
                                                pct_start=0.02, anneal_strategy="cos")


    if A.distill:
        # Pristine router/norm values for the teacher: these train in-place (unlike the
        # adapters, which disable_adapter() bypasses), so without this the teacher would
        # drift with the student. ~30 MB total.
        pristine = {n: p.detach().clone() for n, p in model.named_parameters()
                    if n.endswith(".gate.weight") or "norm" in n.split(".")[-2]}
        trained = {}

        def _teacher_swap(restore=False):
            for n, p in model.named_parameters():
                if n in pristine:
                    if restore:
                        p.data.copy_(trained[n])
                    else:
                        trained[n] = p.detach().clone()
                        p.data.copy_(pristine[n])

        def teacher_hidden(b):
            # Qwen3.5's 248k vocab makes full-logit KL OOM; the teacher hands back HIDDEN
            # states (8 MB, not 2 GB) and each chunk's logits are built inside a
            # checkpointed loss that recomputes in backward. Valid because lm_head is
            # frozen and identical for teacher and student, so teacher chunk logits are
            # recomputable at backward time without the weight swap being active.
            _teacher_swap()
            RES._CFG["on"] = False
            with torch.no_grad(), model.disable_adapter():
                h_t = causal.model(b)[0].detach()
            RES._CFG["on"] = True
            _teacher_swap(restore=True)
            return h_t

        def generate_cache(row0, n_rows):
            """Teacher targets for corpus rows [row0, row0+n_rows): top-K values (bf16) +
            indices (int32) per position, in host RAM. Weights swapped once for the sweep."""
            _teacher_swap()
            was_on = RES._CFG["on"]
            RES._CFG["on"] = False
            K = A.teacher_topk
            vals = torch.empty(n_rows, A.seq, K, dtype=torch.bfloat16)
            idxs = torch.empty(n_rows, A.seq, K, dtype=torch.int32)
            mass_sum, mass_n = 0.0, 0
            with torch.no_grad(), model.disable_adapter():
                for i in range(0, n_rows, A.teacher_mb):
                    rb = corpus[row0 + i: row0 + min(i + A.teacher_mb, n_rows), :A.seq] \
                        .to("cuda").long()
                    h = causal.model(rb)[0]
                    for j0 in range(0, A.seq, 512):          # chunk the vocab projection
                        lg = (h[:, j0:j0 + 512] @ causal.lm_head.weight.T).float()
                        v, ix = lg.topk(K, dim=-1)
                        logZ = torch.logsumexp(lg, -1, keepdim=True)
                        mass_sum += float((v - logZ).exp().sum(-1).mean()) * v.shape[0] * v.shape[1]
                        mass_n += v.shape[0] * v.shape[1]
                        vals[i:i + rb.shape[0], j0:j0 + 512] = v.to(torch.bfloat16).cpu()
                        idxs[i:i + rb.shape[0], j0:j0 + 512] = ix.to(torch.int32).cpu()
                        del lg, v, ix, logZ
                    del h, rb
            RES._CFG["on"] = was_on
            _teacher_swap(restore=True)
            print(f"  [cache] top-{K} mass coverage {mass_sum / max(1, mass_n):.6f}", flush=True)
            return vals, idxs

        def cached_kl(h_s, cv, cix, T, chunk=512):
            """Soft-CE of the student against renormalised cached top-K teacher targets,
            chunk-checkpointed like chunked_kl. Constant teacher-entropy term omitted (no
            gradient); reported value is soft-CE + renormalised entropy - matches KL up to
            the top-K truncation."""
            from torch.utils.checkpoint import checkpoint
            W = causal.lm_head.weight

            def _chunk(hs_c, cv_c, cix_c):
                lg_s = (hs_c @ W.T).float() / T
                logZ = torch.logsumexp(lg_s, -1, keepdim=True)
                sel = lg_s.gather(-1, cix_c.long()) - logZ          # log p_s at cached idx
                with torch.no_grad():
                    p = torch.softmax(cv_c.float() / T, -1)          # renormalised top-K
                    ent = (p * torch.log_softmax(cv_c.float() / T, -1)).sum(-1)
                return (-(p * sel).sum(-1) + ent).sum()

            S = h_s.shape[1]
            tot = None
            for i in range(0, S, chunk):
                c = checkpoint(_chunk, h_s[:, i:i + chunk],
                               cv[:, i:i + chunk].to("cuda"), cix[:, i:i + chunk].to("cuda"),
                               use_reentrant=False)
                tot = c if tot is None else tot + c
            return (T * T) * tot / (S * h_s.shape[0])

        def chunked_kl(h_s, h_t, T, chunk=512):
            from torch.utils.checkpoint import checkpoint
            W = causal.lm_head.weight

            def _kl_chunk(hs_c, ht_c):
                lg_s = (hs_c @ W.T).float() / T
                with torch.no_grad():
                    lg_t = (ht_c @ W.T).float() / T
                    pt = torch.softmax(lg_t, -1)
                    ent = (pt * torch.log_softmax(lg_t, -1)).sum(-1)
                ce = -(pt * torch.log_softmax(lg_s, -1)).sum(-1)
                return (ce + ent).sum()

            S = h_s.shape[1]
            tot = None
            for i in range(0, S, chunk):
                c = checkpoint(_kl_chunk, h_s[:, i:i + chunk], h_t[:, i:i + chunk],
                               use_reentrant=False)
                tot = c if tot is None else tot + c
            return (T * T) * tot / (S * h_s.shape[0])

    E_total = getattr(model.config, "text_config", model.config).num_experts
    anneal_steps = int(A.r_anneal * steps)
    if anneal_steps and A.anneal_style == "damage":
        # Fit d(R) = A_fit * R^-gamma to the measured training-free damage curve (qwen3,
        # 2026-08-07; damage = BPB above free 0.615306), log-log least squares. The schedule
        # inverts the fit so damage increases EXACTLY linearly in window time -- continuous,
        # not piecewise -- with integer dwells falling out of the rounding.
        import math
        assert A.family == "qwen3" and A.R == 8, "damage schedule is qwen3/R=8-specific"
        _pts = [(64, 0.007892), (32, 0.025135), (16, 0.056882),
                (12, 0.078085), (10, 0.093056), (8, 0.118096)]
        _xs = [math.log(r) for r, _ in _pts]
        _ys = [math.log(d) for _, d in _pts]
        _mx, _my = sum(_xs) / len(_pts), sum(_ys) / len(_pts)
        _gamma = -sum((x - _mx) * (y - _my) for x, y in zip(_xs, _ys)) \
            / sum((x - _mx) ** 2 for x in _xs)
        _logA = _my + _gamma * _mx
        # Ramp from d(E) (start at FULL residency: the loose descent E -> E/4 costs ~1 step
        # per R under damage pacing -- a warm-up sweep) to d(target R): ramping the damage
        # target all the way to d(8) with R clamped >= 9 in-window gives R=9 its natural
        # dwell share, so dwell is monotone in tightness right up to the hold.
        _dStart = math.exp(_logA - _gamma * math.log(E_total))
        _dEnd = math.exp(_logA - _gamma * math.log(A.R))
        print(f"  [anneal] fitted d(R) = A*R^-gamma, gamma = {_gamma:.3f}; window walks "
              f"damage linearly {_dStart:.4f} -> {_dEnd:.4f} (R {E_total} -> 9, clamped), "
              f"then R={A.R}", flush=True)

    hist, seen, t0, nxt, ptr = [], 0, time.time(), A.eval_every, 0
    cache_v = cache_i = None
    cache_row0 = cache_rows = 0
    seg_rows = max(A.mb, A.teacher_cache_tokens // A.seq) if A.teacher_cache_tokens else 0
    for step in range(steps):
        if anneal_steps and step < anneal_steps:
            u = step / anneal_steps
            if A.anneal_style == "damage":
                # Continuous damage-linear schedule from the fitted power law: R >= 9 strictly
                # inside the window; the deployment setting R=8 owns the entire hold.
                import math
                _d_u = _dStart + u * (_dEnd - _dStart)
                new_R = max(9, min(E_total,
                                   int(round(math.exp((_logA - math.log(_d_u)) / _gamma)))))
            else:
                # Linear: E/4 -> R across the window (the user's arm (i) start point).
                # Clamped >= R+1 in-window so BOTH arms spend exactly the 40% hold at the
                # deployment R -- the linear ramp otherwise touches R a few steps early.
                start = E_total // 4
                new_R = max(A.R + 1, int(round(start - (start - A.R) * u)))
            if RES._CFG["R"] != new_R:
                print(f"  [anneal] step {step}/{steps} R -> {new_R}", flush=True)
            RES._CFG["R"] = new_R
        elif anneal_steps and step == anneal_steps:
            print(f"  [anneal] step {step}/{steps} window closed, R -> {A.R} (deployment)",
                  flush=True)
            RES._CFG["R"] = A.R
            anneal_steps = 0                                 # hold permanently; stop scheduling
        opt.zero_grad(set_to_none=True)
        aux_vals = []
        for _ in range(A.accum):
            bp = ptr                                     # row index of THIS batch (for cache)
            b = corpus[ptr:ptr + A.mb, :A.seq].to("cuda").long()
            ptr = 0 if ptr + 2 * A.mb > len(corpus) else ptr + A.mb
            RU.AUX_LOG.clear()
            if A.distill and seg_rows:
                # Rolling cached teacher: (re)generate when the batch leaves the segment.
                if cache_v is None or bp < cache_row0 or bp + A.mb > cache_row0 + cache_rows:
                    n = min(seg_rows, len(corpus) - bp)
                    print(f"  [cache] generating teacher segment rows {bp}..{bp + n} "
                          f"({n * A.seq / 1e6:.0f}M tokens, top-{A.teacher_topk}) "
                          f"{(time.time()-t0)/60:.1f}min", flush=True)
                    # Free the old segment first: two live segments peak at 2x cache RAM,
                    # which is what breached the 251GB cgroup cap on 08-07.
                    cache_v = cache_i = None
                    cache_v, cache_i = generate_cache(bp, n)
                    cache_row0, cache_rows = bp, n
                    print(f"  [cache] segment ready {(time.time()-t0)/60:.1f}min", flush=True)
                o = bp - cache_row0
                h = causal.model(b)[0]
                lm = cached_kl(h, cache_v[o:o + A.mb], cache_i[o:o + A.mb], A.distill_T)
            elif A.distill:
                # Inline chunked-checkpointed KL (exact; used when caching is off).
                h_t = teacher_hidden(b)
                h = causal.model(b)[0]
                lm = chunked_kl(h, h_t, A.distill_T)
                del h_t
            else:
                h = causal.model(b)[0]                   # logits never materialised (CCE)
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
           "aux_c": A.aux_c, "z_c": A.z_c, "lr": A.lr, "opt": "adamw8bit",
           "ce": "distill_kl" if A.distill else "cce",
           "r_anneal": A.r_anneal,
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
