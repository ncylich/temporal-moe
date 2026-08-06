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

# Family table. The expert and attention classes differ by family, so they are resolved at runtime
# rather than imported at module load -- importing Qwen3_5 classes unconditionally would make a
# Qwen3-30B run depend on a model file it never touches.
#   qwen3_5  40 layers, 256 experts, 30 Gated DeltaNet + 10 full attention -> attn LoRA reaches 10/40
#   qwen3    48 layers, 128 experts, full attention throughout            -> attn LoRA reaches 48/48
# That asymmetry is architectural: "CE + attention LoRA" is a materially weaker intervention on
# qwen3_5, and the two arms should not be read as the same recipe applied twice.
FAMILY = {
    "qwen3_5": {"mod": "qwen3_5_moe", "experts": "Qwen3_5MoeExperts", "attn": "Qwen3_5MoeAttention",
                "model": "/workspace/qwen35-adapt/model", "data": "/workspace/qwen35-adapt/data",
                "out": "/workspace/qwen35-adapt/results", "suffix": "qwen"},
    "qwen3":   {"mod": "qwen3_moe", "experts": "Qwen3MoeExperts", "attn": "Qwen3MoeAttention",
                "model": "/dev/shm/qwen3-30b", "data": "/workspace/qwen3moe-adapt/data",
                "out": "/workspace/qwen3moe-adapt/results", "suffix": "qwen3"},
}
_CLS = {}


def resolve(family):
    f = FAMILY[family]
    m = __import__(f"transformers.models.{f['mod']}.modeling_{f['mod']}",
                   fromlist=[f["experts"], f["attn"]])
    _CLS["experts"] = getattr(m, f["experts"])
    _CLS["attn"] = getattr(m, f["attn"])
    return f

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
        if not isinstance(m, _CLS["experts"]):
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
        _orig_experts_forward = _CLS["experts"].forward
    _CLS["experts"].forward = _experts_forward_lora
    return ps


def _hook(mod, inp, out):
    x = inp[0]
    return out + F.linear(F.linear(x, mod.lora_A), mod.lora_B) * mod.lora_s


def add_lora_attn(model, r=32, alpha=64, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    """LoRA on the full-attention layers. Gated DeltaNet layers are skipped -- they have no q/k/v/o."""
    ps, n = [], 0
    for m in model.modules():
        if not isinstance(m, _CLS["attn"]):
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
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(F.cross_entropy(sl.reshape(-1, sl.shape[-1]),
                                         tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    model.train()
    # Unsloth loads Qwen3.5 with its composite config: num_experts lives on text_config.
    # The stock text-only load keeps it top-level; resolve both.
    _cfg = getattr(model.config, "text_config", model.config)
    swap, ent = RES.telem_summary(_cfg.num_experts)
    return (tot / ntok) / divisor, swap, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--family", default="qwen3_5", choices=("qwen3_5","qwen3"))
    # Swept in stage 3. The aux pushes routing toward uniform expert usage while residency
    # restricts which experts are reachable; aux_c=0 is the arm that tests whether they fight.
    # Default None means "use what the model ships". Qwen3-30B and Qwen3.5 both declare
    # router_aux_loss_coef=0.001; OLMoE declares 0.01. Every run to date used 0.01 on all
    # three, i.e. 10x the intended balancing pressure on both Qwen models, inherited from
    # OLMoE along with the learning rate.
    ap.add_argument("--aux-c", type=float, default=None)
    ap.add_argument("--aux-scope", default="micro", choices=("micro", "global"))
    # "balance BSZ": how many SEQUENCES the dispatch statistics are pooled over. Qwen report
    # perplexity falling steeply from 2 to 128 and saturating after 128, and note mainstream
    # frameworks sit at 8-16 (arXiv 2501.11873). A micro-batch here is 1-2 sequences, i.e.
    # below the setting that paper criticises. Pooling within one optimiser step only reaches
    # 8, so the window extends across steps -- f is detached, so this costs an [E] buffer per
    # layer and nothing else.
    ap.add_argument("--aux-balance-seqs", type=int, default=128)
    # THE missing adaptation surface. train_ple.py calls RES.freeze_all_but_router and trains
    # the router directly, which is why OLMoE recovers effective expert count 20.5 -> 49.5.
    # This trainer froze every base parameter, so on Qwen the router could only be nudged
    # indirectly through attention LoRA shifting the hidden states. Rolling residency is a
    # constraint ON ROUTING; adapting to it with the router frozen is adapting around it.
    # Cost is small: gate is [E, hidden], 12.6M params on Qwen3-30B, 21M on Qwen3.5.
    ap.add_argument("--train-router", action="store_true")
    # train_ple.py:137 adds learnable RMSNorm gains ("arm C surface"). Small (~65K on OLMoE)
    # but it is part of the adaptation surface, and the arms have to match or the cross-model
    # comparison is measuring the recipe rather than the model.
    ap.add_argument("--train-norms", action="store_true")
    ap.add_argument("--z-c", type=float, default=0.001)
    ap.add_argument("--tokens", type=int, default=30_000_000)
    ap.add_argument("--free-set", default="", help="comma list; 'all' = unconstrained null")
    ap.add_argument("--lora", type=int, default=8)
    ap.add_argument("--lora-attn", type=int, default=0)
    ap.add_argument("--seq", type=int, default=2048)
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
    # NOT named F: `import torch.nn.functional as F` is module-level and shadowing it
    # replaces cross_entropy with a dict lookup at the first training step.
    FAM = resolve(A.family)
    DATA, OUT, SFX = FAM["data"], FAM["out"], FAM["suffix"]
    os.makedirs(OUT, exist_ok=True)

    meta = json.load(open(f"{DATA}/bpb_slice_meta_{SFX}.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model(path=FAM["model"], family=A.family)
    if A.aux_c is None:
        _tc = getattr(model.config, "text_config", model.config)
        A.aux_c = float(getattr(_tc, "router_aux_loss_coef",
                                getattr(model.config, "router_aux_loss_coef", 0.01)))
        print(f"  [aux] using the model's shipped router_aux_loss_coef = {A.aux_c}", flush=True)
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
    # --lora 0 means no expert LoRA, which is what the OLMoE "CE + attention" cell used
    # (--rank off --lora-attn 32). It also leaves the stock expert forward in place: attaching expert
    # LoRA swaps in _experts_forward_lora, a Python loop over experts with index_add_, which is the
    # pattern profiling put at 89.5% of layer time. Skipping it keeps the measured 6,704 tok/s.
    params = add_lora(model, r=A.lora) if A.lora else []
    if A.lora_attn:
        params += add_lora_attn(model, r=A.lora_attn)
    if A.train_router:
        _rtr = RQ.FAMILIES[A.family][1]
        rps = [m.weight for m in model.modules() if isinstance(m, _rtr)]
        for p in rps:
            p.requires_grad_(True)
        params = params + rps
        print(f"  [router] training {len(rps)} router gates, "
              f"{sum(p.numel() for p in rps)/1e6:.1f}M params", flush=True)
    if A.train_norms:
        import torch.nn as _nn
        nps = [m.weight for m in model.modules()
               if m.__class__.__name__.endswith("RMSNorm") and getattr(m, "weight", None) is not None]
        for p in nps:
            p.requires_grad_(True)
        params = params + nps
        print(f"  [norms] training {len(nps)} RMSNorm gains, "
              f"{sum(p.numel() for p in nps)/1e6:.2f}M params", flush=True)
    if not params:
        sys.exit("no trainable parameters: pass --lora, --lora-attn, --train-router and/or --train-norms")
    for p in params:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in params)
    print(f"  [qwen-train] tag={A.tag} free={A.free_set or 'none'} R={A.R} trainable={n_tr/1e6:.1f}M "
          f"tokens={A.tokens/1e6:.0f}M divisor={D:.7f}", flush=True)

    corpus = torch.load(f"{DATA}/finetune_ids_{SFX}.pt", weights_only=False)
    # The corpus is packed at 4096 tokens per row. The training loop slices [:, :A.seq], so at
    # --seq 2048 the second half of every row was silently discarded: 33.5M usable tokens instead of
    # 66.9M, which forced the 50M runs into 1.49 epochs of repetition with half the corpus unread.
    # Repeating data at this scale while holding unseen tokens is simply a bug. Reshaping recovers
    # every token and keeps sequence length unchanged.
    if A.seq < corpus.shape[1]:
        nper = corpus.shape[1] // A.seq
        corpus = corpus[:, :nper * A.seq].reshape(-1, A.seq)
        print(f"  [data] reshaped to {corpus.shape[0]} x {A.seq} = "
              f"{corpus.numel()/1e6:.1f}M tokens (was {corpus.shape[0]//nper} x {nper*A.seq}, "
              f"of which only {corpus.shape[0]//nper*A.seq/1e6:.1f}M were reachable)", flush=True)
    epochs = A.tokens / corpus.numel()
    print(f"  [data] {A.tokens/1e6:.0f}M tokens requested = {epochs:.2f} epochs", flush=True)
    if epochs > 1.0:
        print(f"  [data] WARNING: >1 epoch, data will repeat", flush=True)
    bpb_ids = torch.load(f"{DATA}/bpb_slice_ids_{SFX}.pt", weights_only=False)[: A.eval_seq]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=A.lr, weight_decay=0.0, betas=(0.9, 0.95))
        print("  [opt] AdamW8bit (fp32 moments would cost 4x the state on 461M LoRA params)", flush=True)
    except Exception as e:
        opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.0, betas=(0.9, 0.95))
        print(f"  [opt] AdamW fp32 fallback ({e})", flush=True)
    model.gradient_checkpointing_enable()
    model.train()

    tok_per_step = A.mb * A.accum * A.seq
    steps = A.tokens // tok_per_step
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=max(1, steps),
                                                pct_start=0.02, anneal_strategy="cos")
    from collections import deque
    seqs_per_step = A.mb * A.accum
    win = max(1, -(-A.aux_balance_seqs // seqs_per_step))   # steps needed to reach the target
    gf_prev, gf_acc, gf_win = None, [], deque(maxlen=win)
    if A.aux_scope == "global":
        print(f"  [aux] scope=global, balance BSZ target {A.aux_balance_seqs} seqs = "
              f"{seqs_per_step} seqs/step x {win} step window", flush=True)
    hist, seen, t0, nxt = [], 0, time.time(), A.eval_every
    ptr = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(A.accum):
            b = corpus[ptr:ptr + A.mb, :A.seq].to("cuda").long(); ptr += A.mb
            if ptr + A.mb > len(corpus):
                ptr = 0
            RQ.capture(True)
            out = model(b)
            rl = RQ.captured(); RQ.capture(False)
            lg = out.logits[:, :-1]
            lm = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), b[:, 1:].reshape(-1))
            of = [] if A.aux_scope == "global" else None
            aux, z = RES.aux_z_from_router_logits(
                rl, b.shape[0], b.shape[1], A.R,
                global_f=(gf_prev if A.aux_scope == "global" else None), out_f=of)
            if of:
                gf_acc.append(of)
            (lm + A.aux_c * aux + A.z_c * z).div(A.accum).backward()
            seen += b.numel()
            del out, lg, rl
        if A.aux_scope == "global" and gf_acc:
            # Pool this optimiser step's micro-batches into one dispatch fraction per layer,
            # used as next step's global f. One step stale, which is the standard trade for
            # not holding every micro-batch's graph alive across accumulation.
            gf_win.append([sum(mb[i] for mb in gf_acc) / len(gf_acc)
                           for i in range(len(gf_acc[0]))])
            gf_prev = [sum(st[i] for st in gf_win) / len(gf_win) for i in range(len(gf_win[0]))]
            gf_acc = []
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
    res["family"] = A.family
    res["aux_c"], res["z_c"], res["lr"] = A.aux_c, A.z_c, A.lr
    res["aux_scope"], res["aux_balance_seqs"] = A.aux_scope, A.aux_balance_seqs
    res["train_router"], res["train_norms"] = A.train_router, A.train_norms
    # Persist the adapter. Without this the run yields a BPB number and discards the model, so no
    # downstream task evaluation is possible afterwards and the arm cannot be re-scored -- the OLMoE
    # side of this comparison has downstream accuracies precisely because train_ple.py checkpoints.
    # Only LoRA tensors are saved (~100 MB); the base weights are unchanged and already on disk.
    try:
        sd = {}
        for i, p in enumerate(params):
            sd[f"lora_{i}"] = p.detach().to(torch.bfloat16).cpu()
        ck = os.path.join(OUT, f"qwen_{A.tag}_adapter.pt")
        torch.save({"tensors": sd, "family": A.family, "lora": A.lora, "lora_attn": A.lora_attn,
                    "free_set": A.free_set, "R": A.R, "tokens": seen}, ck)
        res["adapter"] = ck
        print(f"  [save] {ck} ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)", flush=True)
    except Exception as e:
        print(f"  [save] FAILED {type(e).__name__}: {e}", flush=True)
    with open(os.path.join(OUT, f"qwen_{A.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[DONE] {A.tag} final BPB={bpb:.6f} swap={swap:.4f} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
