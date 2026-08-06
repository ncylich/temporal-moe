#!/usr/bin/env python3
"""Does the fused MoE path make Qwen adaptation practical? Measures a real training step.

Every throughput claim in this programme that came from a proxy has been wrong -- an inference
benchmark predicted a 2x that was 1.15x, a microbenchmark predicted 2x that was 1.15x again, and a
"2.69x" was a crippled baseline. Training is forward + backward + gradient checkpointing, with an
optimiser step and a vocab-sized loss, so it is measured here directly rather than inferred.

!! THE BASELINE IN THIS FILE WAS WRONG AND EVERY SPEEDUP DERIVED FROM IT IS WITHDRAWN. !!
The "176 s/step / 93 tok/s stock" reference below was measured in qwen35_RESULTS.md on a DIFFERENT
model in a DIFFERENT configuration: Qwen3.5 (40 layers, 256 experts) carrying 461M of EXPERT LoRA
through _experts_forward_lora's Python loop, at micro-batch 1. This file benchmarks Qwen3-30B with
ATTENTION-ONLY LoRA and the stock expert forward. Citing one as the other produced a 22.7x and a 65x
that do not exist.

What stock actually does in the configuration we train at: the 50M Qwen3-30B run did 49,987,584
tokens in 132.8 min = 6,274 tok/s, against this file's best fused figure of 6,046 tok/s at seq 1024.
Stock is at or above fused where it counts. See results/ablations/crossmodel_RESULTS.md S9.

Residency is ON for the constrained arm, so this measures the configuration we would actually train,
not a bare forward.

    bench_train_fused.py --model /root/models/qwen3-30b-fused --batches 1,4,8,16
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/qwen3-moe-fused")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_fused as RF                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/qwen3-30b-fused")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--lora", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--free-set", default="")
    A = ap.parse_args()

    from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
    from qwen3_moe_fused.lora import patch_lora_config
    from peft import LoraConfig, get_peft_model

    patch_lora_config()
    t0 = time.time()
    model = Qwen3MoeFusedForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16).to("cuda")
    print(f"  loaded in {time.time()-t0:.0f}s  weights {torch.cuda.memory_allocated()/1e9:.1f} GB",
          flush=True)
    L, E = model.config.num_hidden_layers, model.config.num_experts
    V = model.config.vocab_size

    cfg = LoraConfig(r=A.lora, lora_alpha=2 * A.lora, use_rslora=True, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA r={A.lora}: {n_tr/1e6:.1f}M trainable", flush=True)

    n = RF.install(model.base_model.model if hasattr(model, "base_model") else model)
    RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False)
    RES.set_free_layers([int(x) for x in A.free_set.split(",")] if A.free_set else None)
    print(f"  residency installed on {n} blocks, R=8 of {E}, free_set={A.free_set or 'none'}",
          flush=True)

    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    print(f"\n  {'micro-batch':>12}{'tok/step':>10}{'s/step':>9}{'tok/s':>11}{'peak GB':>10}", flush=True)
    for mb in [int(x) for x in A.batches.split(",")]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, V, (mb, A.seq), device="cuda")
            for _ in range(2):                                     # warm + autotune
                out = model(ids, labels=ids); out.loss.backward(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(A.steps):
                out = model(ids, labels=ids)
                out.loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / A.steps
            tok = mb * A.seq
            print(f"  {mb:>12}{tok:>10}{dt:>9.2f}{tok/dt:>11,.0f}"
                  f"{torch.cuda.max_memory_allocated()/1e9:>10.1f}", flush=True)
            del ids, out
        except torch.OutOfMemoryError:
            print(f"  {mb:>12}{'':>10}{'OOM':>9}", flush=True)
            torch.cuda.empty_cache()
            break
    print("\n  reference: stock in the ACTUAL training configuration (Qwen3-30B, attention-only LoRA,"
          "\n  seq 2048, mb 4 x accum 2) ran at 6,274 tok/s over the 50M run. The old '93 tok/s'"
          "\n  reference was Qwen3.5 with 461M expert LoRA at mb=1 -- a different model, adapter and"
          "\n  code path -- and must not be used as a baseline here.", flush=True)
    print("=== TRAIN BENCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
