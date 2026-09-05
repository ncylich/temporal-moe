#!/usr/bin/env python3
"""Where does a real training step actually spend its time on the unsloth grouped_mm path?

The 37.5% gather/scatter figure in TRAINING_OPTIM_PLAN.md was profiled on the RETIRED
fused library; unsloth's zoo path avoids at least one materialised copy, so the number
must be re-measured before any kernel work is justified (tier-1 of the permutation-
optimisation plan). Profiles N optimiser steps of the distillation config with
torch.profiler and rolls CUDA time up into categories.

    profile_moe_step.py --family qwen3 --mb 4 --steps 3
"""
import argparse
import os
import sys
from collections import defaultdict

import unsloth  # noqa: F401
from unsloth import FastModel
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_unsloth as RU                                     # noqa: E402
import train_qwen as TQ                                            # noqa: E402

CATS = {
    "grouped_mm/gemm": ("grouped_mm", "gemm", "cutlass", "matmul", "mm_", "addmm", "bmm"),
    "permute/gather/scatter": ("gather", "scatter", "index", "sort", "argsort", "bincount",
                               "cumsum", "one_hot", "unique", "embedding_dense"),
    "residency_scan": ("_scan_kernel", "resident"),
    "softmax/norm/elementwise": ("softmax", "norm", "silu", "sigmoid", "mul", "add_", "copy_",
                                 "elementwise", "vectorized"),
    "attention": ("attention", "sdpa", "flash", "cudnn"),
    "optimizer": ("adam", "optimizer", "8bit"),
}


def cat_of(name):
    n = name.lower()
    for c, keys in CATS.items():
        if any(k in n for k in keys):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen3", choices=("qwen3", "qwen3_5"))
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lora", type=int, default=32)
    A = ap.parse_args()

    FAM = TQ.resolve(A.family)
    model, _ = FastModel.from_pretrained(FAM["model"], max_seq_length=2048,
                                         dtype=torch.bfloat16, load_in_4bit=False,
                                         full_finetuning=False)
    for mod in model.modules():
        if getattr(mod, "visual", None) is not None and "Vision" in type(mod.visual).__name__:
            mod.visual = None
            torch.cuda.empty_cache()
            break
    model = FastModel.get_peft_model(model, r=A.lora, lora_alpha=2 * A.lora,
                                     lora_dropout=0.0, use_gradient_checkpointing="unsloth")
    RU.install(model)
    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
    RES.set_free_layers(None)
    for name, p in model.named_parameters():
        if name.endswith(".gate.weight") or "norm" in name.split(".")[-2]:
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    import bitsandbytes as bnb
    opt = bnb.optim.PagedAdamW8bit(params, lr=1e-5)
    model.train()
    cfg = getattr(model.config, "text_config", model.config)
    causal = model.base_model.model
    ids = torch.randint(0, cfg.vocab_size, (A.mb, 2048), device="cuda")

    def step():
        h = causal.model(ids)[0]
        from cut_cross_entropy import linear_cross_entropy
        loss = linear_cross_entropy(h, causal.lm_head.weight, ids, shift=True)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):                                   # warmup: JIT, autotune, cache
        step()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(A.steps):
            step()
        torch.cuda.synchronize()

    tot = defaultdict(float)
    grand = 0.0
    for ev in prof.key_averages():
        t = ev.device_time_total
        if t <= 0:
            continue
        tot[cat_of(ev.key)] += t
        grand += t
    print(f"\n[profile] {A.family} mb={A.mb}, {A.steps} steps, CE objective "
          f"(distill adds a no-grad teacher fwd on top):")
    for c, t in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {c:28s} {t/1e3:9.1f} ms  {100*t/grand:5.1f}%")
    print("\n[profile] top 12 individual kernels:")
    for ev in sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:12]:
        print(f"  {ev.device_time_total/1e3:8.1f} ms  {100*ev.device_time_total/grand:5.1f}%  "
              f"{ev.key[:90]}")
    print("=== PROFILE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
