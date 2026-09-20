#!/usr/bin/env python3
"""Which off-the-shelf quantizer actually shrinks a Qwen MoE, and still runs a training step?

Two failures are already on record and this exists to avoid a third. bitsandbytes 4-bit loaded
Qwen3-30B at 59.7 GB against ~57 GB in bf16, because it converts nn.Linear and Qwen holds experts as
3-D tensors -- roughly 90% of the weights were skipped. Then passing `{"quant_method": "fp8"}` through
the model config told transformers the CHECKPOINT was already FP8, so every `weight_scale_inv` was
reported MISSING and newly initialised: a garbage model that then died in the Triton GEMM anyway.

The correct call is a config OBJECT passed as `quantization_config=` to from_pretrained, which
quantises a bf16 checkpoint during load.

Two things are checked per quantizer, because either alone is misleading:
    memory   weights actually resident after load -- catches "quantised everything except the 90%"
    step     a real forward+backward+optimiser step -- catches kernels that do not compile

Nothing here is adopted on these numbers. Whatever passes both goes to the BPB acceptance test
against the same-kernel bf16 noise floor, the test that rejected grouped_mm at 4.93e-04.

    probe_quantizers.py --model /dev/shm/qwen3-30b
"""
import argparse
import gc
import time

import torch


def build(name):
    import transformers as t
    if name == "bf16":
        return None
    if name == "fp8_finegrained":
        return t.FineGrainedFP8Config()
    if name == "fp8_fbgemm":
        return t.FbgemmFp8Config()
    if name == "mxfp4":
        return t.Mxfp4Config()
    if name == "torchao_fp8":
        from torchao.quantization import Float8WeightOnlyConfig
        return t.TorchAoConfig(quant_type=Float8WeightOnlyConfig())
    if name == "torchao_int8":
        from torchao.quantization import Int8WeightOnlyConfig
        return t.TorchAoConfig(quant_type=Int8WeightOnlyConfig())
    raise ValueError(name)


def try_one(path, name, seq, mb):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        qc = build(name)
    except Exception as e:
        return f"  {name:18} UNAVAILABLE  {type(e).__name__}: {str(e)[:70]}"
    kw = {"dtype": torch.bfloat16}
    if qc is not None:
        kw["quantization_config"] = qc
        kw["device_map"] = {"": 0}
    try:
        m = AutoModelForCausalLM.from_pretrained(path, **kw)
        if qc is None:
            m = m.to("cuda")
    except Exception as e:
        torch.cuda.empty_cache()
        return f"  {name:18} LOAD FAILED   {type(e).__name__}: {str(e)[:70]}"
    wmem = torch.cuda.memory_allocated() / 1e9
    load_s = time.time() - t0

    msg = f"  {name:18} {wmem:6.1f} GB  load {load_s:5.0f}s  "
    try:
        V = getattr(m.config, "vocab_size", None) or m.config.text_config.vocab_size
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, bias="none",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        m.gradient_checkpointing_enable(); m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        ids = torch.randint(0, V, (mb, seq), device="cuda")
        out = m(ids, labels=ids); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t1 = time.time()
        out = m(ids, labels=ids); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        msg += f"step OK  {mb*seq/(time.time()-t1):,.0f} tok/s"
    except Exception as e:
        msg += f"STEP FAILED  {type(e).__name__}: {str(e)[:60]}"
    del m
    gc.collect(); torch.cuda.empty_cache()
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/qwen3-30b")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--quants", default="bf16,fp8_finegrained,fp8_fbgemm,mxfp4,torchao_fp8,torchao_int8")
    A = ap.parse_args()
    print(f"  model={A.model}  seq={A.seq}  mb={A.mb}", flush=True)
    print(f"  reference: bf16 weights ~57 GB (qwen3-30b); bnb-4bit measured 59.7 GB i.e. no saving\n",
          flush=True)
    for name in A.quants.split(","):
        print(try_one(A.model, name, A.seq, A.mb), flush=True)
    print("\n=== QUANT PROBE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
