#!/usr/bin/env python3
"""What does expert LoRA actually cost, per model, in the configuration we will run?

Asked because the only number on record for expert LoRA on the stock path is 93 tok/s (Qwen3.5,
micro-batch 1), which would make a 15M-token run take 45 hours. That figure has already been misused
once as a general baseline; this measures each model in the configuration the sweep will actually
use, so the schedule rests on measurement rather than on extrapolating a single point.

Measures tokens/sec for a real training step -- forward, backward, optimiser -- with the full
trainable surface: expert LoRA + attention LoRA + router + RMSNorm gains.

Reference points from runs already completed:
    OLMoE      attn-only LoRA, mb4 x seq4096   13,900 tok/s
    Qwen3-30B  attn-only LoRA, mb4 x acc2      6,274 tok/s
    Qwen3.5    attn-only LoRA, mb1 x acc8      3,850 tok/s
    Qwen3.5    expert+attn LoRA, mb1              93 tok/s   <- the number that motivates this probe

    probe_expert_lora_cost.py --model olmoe
"""
import argparse
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("olmoe", "qwen3", "qwen3_5"))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lora", type=int, default=32)
    ap.add_argument("--opt", default="fused", choices=("fused", "adamw8bit"))
    # cce: cut_cross_entropy computes exact CE fused from hidden states + lm_head, never
    # materialising the [B,S,V] logits -- ~2 GB on a 248k vocab at mb1. The alternative
    # route to fitting r32: unlike adamw8bit it changes no training dynamics.
    ap.add_argument("--ce", default="plain", choices=("plain", "cce"))
    A = ap.parse_args()
    sys.path.insert(0, "/workspace/temporal-moe/analysis/ple")
    import residency as RES

    if A.model == "olmoe":
        model, _ = RES.load_model()
        mb, seq = 4, 4096
        params = RES.router_params(model) + RES.norm_params(model)
        params += RES.add_lora(model, r=A.lora, alpha=2 * A.lora)
        params += RES.add_lora_attn(model, r=A.lora, alpha=2 * A.lora)
        E = model.config.num_experts
    else:
        import residency_qwen as RQ
        import train_qwen as TQ
        FAM = TQ.resolve(A.model)
        model, _ = RQ.load_model(path=FAM["model"], family=A.model)
        # Qwen3.5's checkpoint is ForConditionalGeneration: an ~8 GB vision tower rides along
        # in every load. Text-only forwards never touch it (guarded by pixel_values), so drop
        # it -- on this model the difference between r32 fitting and OOM.
        for mod in model.modules():
            if getattr(mod, "visual", None) is not None \
                    and "Vision" in type(mod.visual).__name__:
                mod.visual = None
                torch.cuda.empty_cache()
                break
        mb, seq = (2, 2048) if A.model == "qwen3" else (1, 2048)
        for p in model.parameters():
            p.requires_grad_(False)
        params = TQ.add_lora(model, r=A.lora)
        params += TQ.add_lora_attn(model, r=A.lora)
        rtr = RQ.FAMILIES[A.model][1]
        params += [m.weight for m in model.modules() if isinstance(m, rtr)]
        for p in params:
            p.requires_grad_(True)
        E = model.config.num_experts if hasattr(model.config, "num_experts") \
            else model.config.text_config.num_experts

    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
    RES.set_free_layers(None)
    model.gradient_checkpointing_enable()
    model.train()
    # fused: single-kernel step with no foreach temporaries. The default (foreach) OOM'd the
    # unsloth arm by ~30 MiB at 1.3B trainable; both probes use fused so the arms stay matched.
    # adamw8bit: see probe_unsloth_cost -- 1-byte block-quantised states, saves 3.7 GB at r32.
    if A.opt == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=1e-5)
    else:
        opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
    ntr = sum(p.numel() for p in params)
    V = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    print(f"  {A.model}: E={E} mb={mb} seq={seq} trainable={ntr/1e6:.1f}M "
          f"(expert LoRA r{A.lora} + attn LoRA r{A.lora} + router + norms)", flush=True)

    # CE on bf16 logits directly: CUDA log_softmax accumulates in fp32 internally, and the
    # explicit .float() copy costs mb*seq*V*4 bytes twice (fwd + saved-for-bwd) -- 1.9 GB at
    # mb1 on Qwen3.5's 232k vocab, which was the difference between fitting and OOM.
    ids = torch.randint(0, V, (mb, seq), device="cuda")
    if A.ce == "cce":
        from cut_cross_entropy import linear_cross_entropy

        def compute_loss():
            h = model.model(ids)[0]                     # last hidden, logits never built
            return linear_cross_entropy(h, model.lm_head.weight, ids, shift=True)
    else:
        def compute_loss():
            lg = model(ids).logits[:, :-1]
            return torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))

    for _ in range(2):
        compute_loss().backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(A.steps):
        compute_loss().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / A.steps
    tps = mb * seq / dt
    print(f"  {tps:,.0f} tok/s   {dt:.2f} s/micro-step   peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"  -> 15M tokens = {15e6/tps/60:.0f} min; a 5-point sweep = {5*15e6/tps/3600:.1f} h")
    print("=== EXPERT LORA COST PROBE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
