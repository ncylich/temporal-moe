#!/usr/bin/env python3
"""Qwen3-30B against Qwen3.5-35B on the same training step, same stack, same process.

The requirement is that Qwen3.5 trains no more than 20% slower than Qwen3. On expert arithmetic it
should be FASTER: 8 experts x 3 matmuls x 2048 x 512 over 40 layers is 2.01 GFLOP/token against
Qwen3-30B's 2048 x 768 over 48 layers at 3.62 GFLOP/token, i.e. 44% fewer expert FLOPs. What Qwen3.5
pays instead is 30 Gated DeltaNet layers of its 40, and 256 experts to route among rather than 128.

`fla` supplies a Triton `chunk_gated_delta_rule`; without it transformers falls back to
`torch_chunk_gated_delta_rule`, a pure-PyTorch chunked implementation. Both are chunked -- the
recurrent path is for decoding with a cache, not for training -- so the fallback costs a constant
factor on 30 layers, not a sequential blow-up. This measures that factor rather than assuming it.

Arms are run in ONE process so the driver, allocator and clocks are shared, and sequentially with an
explicit free between, because two models of this size do not co-reside on 80 GB.

    bench_qwen_pair.py --batches 1,2,4 --seq 2048
"""
import argparse
import gc
import os
import sys
import time

import torch

QWEN3 = "/dev/shm/qwen3-30b"
QWEN35 = "/workspace/qwen35-adapt/model"


def fla_present():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule    # noqa: F401
        return True
    except Exception:
        return False


def load(path, quant, experts_impl):
    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16}
    if quant == "4bit":
        # Kept as the documented negative control: bitsandbytes converts nn.Linear, and Qwen holds
        # experts as 3-D tensors, so this quantises attention and skips ~90% of the weights. Measured
        # at 59.7 GB for Qwen3-30B against ~57 GB in bf16 -- i.e. no saving where it matters.
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    elif quant in ("fp8", "mxfp8", "mxfp4"):
        # These are MoE-aware: they were built for expert tensors rather than for Linear modules,
        # which is the whole reason bitsandbytes cannot help here. Available only on torch >= 2.5 --
        # the earlier attempt died on torch 2.4 lacking nn.Module.set_submodule and a float8 cat
        # kernel, both of which exist natively now.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path)
        cfg.quantization_config = {"quant_method": quant}
        kw["config"] = cfg
        kw["device_map"] = {"": 0}
    m = AutoModelForCausalLM.from_pretrained(path, **kw)
    if quant == "bf16":
        m = m.to("cuda")
    if experts_impl:
        # transformers' ExpertsInterface: one dispatch point that works for ANY registered MoE
        # architecture, which is why it is used here instead of the Qwen3-only fused library --
        # that library has no qwen3_5 model file and so can never cover both arms.
        try:
            m.config._experts_implementation = experts_impl
            print(f"    experts_implementation={experts_impl}", flush=True)
        except Exception as e:
            print(f"    experts_implementation={experts_impl} REJECTED: {type(e).__name__}", flush=True)
    return m


def measure(model, mb, seq, steps, lora_r):
    from peft import LoraConfig, get_peft_model
    V = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ids = torch.randint(0, V, (mb, seq), device="cuda")
    for _ in range(2):
        out = model(ids, labels=ids); out.loss.backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        out = model(ids, labels=ids)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    return mb * seq / dt, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lora", type=int, default=8)
    ap.add_argument("--quant", default="bf16", choices=("bf16","4bit","fp8","mxfp8","mxfp4"))
    ap.add_argument("--experts-impl", default="", help="e.g. grouped_mm, deepgemm, sonicmoe")
    A = ap.parse_args()

    print(f"  fla available: {fla_present()}   quant={A.quant}   seq={A.seq}", flush=True)
    best = {}
    for name, path in (("qwen3-30b", QWEN3), ("qwen3.5-35b", QWEN35)):
        print(f"\n  === {name} ===", flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            model = load(path, A.quant, A.experts_impl)
        except Exception as e:
            print(f"    LOAD FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        print(f"    loaded in {time.time()-t0:.0f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB",
              flush=True)
        for mb in [int(x) for x in A.batches.split(",")]:
            try:
                torch.cuda.reset_peak_memory_stats()
                tps, peak = measure(model, mb, A.seq, A.steps, A.lora)
                best[name] = max(best.get(name, 0), tps)
                print(f"    mb={mb:<3} {tps:>9,.0f} tok/s   peak {peak:.1f} GB", flush=True)
            except torch.OutOfMemoryError:
                print(f"    mb={mb:<3} OOM", flush=True)
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"    mb={mb:<3} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                break
        del model
        gc.collect(); torch.cuda.empty_cache()

    if len(best) == 2:
        a, b = best["qwen3-30b"], best["qwen3.5-35b"]
        slower = (a - b) / a * 100
        print(f"\n  === requirement: Qwen3.5 no more than 20% slower ===")
        print(f"  qwen3-30b   {a:>9,.0f} tok/s")
        print(f"  qwen3.5-35b {b:>9,.0f} tok/s   ({b/a:.2f}x, {slower:+.1f}% vs qwen3)")
        print(f"  VERDICT: {'PASS' if slower <= 20 else 'FAIL'}")
    print("=== QWEN PAIR BENCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
