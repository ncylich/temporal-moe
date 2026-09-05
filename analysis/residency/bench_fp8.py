#!/usr/bin/env python3
"""FP8 against bf16: is it faster, does it fit more, and does it compute the same model?

FP8 is attractive here for a specific reason that has nothing to do with arithmetic throughput.
57 GB of bf16 weights on an 80 GB card left ~10 GB for adapter, optimizer states and activations,
which forced LoRA training to micro-batch 1 -- and micro-batch 1 is the regime the inference
benchmark shows is 8x off peak. Halving the weights buys the batch size that fixes training. Larger
eval batches come for free with it.

The equivalence check is not optional. Every effect this program measures is small: the aux-loss
correction was 4.85e-04 BPB and free-set differences are ~2.5e-03. A quantisation that shifts BPB by
a comparable amount would be indistinguishable from the residency damage we are trying to measure,
and we would be charging quantisation error to the constraint. `grouped_mm` already failed exactly
this test at 4.93e-04, so the bar is not hypothetical.

Reported per precision: peak memory, tok/s across batch sizes, and BPB on the audited slice.
Adoption requires the BPB delta to be small against the effects being measured, not merely "close".

    bench_fp8.py --model /dev/shm/qwen3-30b --family qwen3
"""
import argparse
import json
import os
import sys
import time

import torch


# --- torch 2.4 compatibility -------------------------------------------------------------------
# transformers' FineGrainedFP8 quantizer calls nn.Module.set_submodule, which was added in torch 2.5;
# this environment is on 2.4.1, so FP8 fails at load with AttributeError before doing any work. The
# method is a tree walk plus a setattr -- get_submodule already exists here and does the same walk --
# so shimming it is a compatibility fix, not a reimplementation of anything numerical. Guarded so it
# is a no-op on any torch that already provides it.
import torch.nn as _nn                                              # noqa: E402
if not hasattr(_nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "_nn.Module") -> None:
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(f"{mod._get_name()} has no attribute `{item}`")
            mod = getattr(mod, item)
            if not isinstance(mod, _nn.Module):
                raise AttributeError(f"`{item}` is not an nn.Module")
        setattr(mod, name, module)
    _nn.Module.set_submodule = _set_submodule
    print("  [compat] shimmed nn.Module.set_submodule (torch 2.4 lacks it)", flush=True)


# torch has no cat kernel for float8 even at 2.5.1, and transformers' FP8 loader concatenates expert
# weight shards (MergeModulelist on experts.gate_up_proj / down_proj) after quantising. cat moves
# bytes and computes nothing, so concatenating the bit patterns as uint8 and reinterpreting is exact
# -- verified bit-identical against the float path. Guarded, and non-fp8 cat is untouched.
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
try:
    torch.cat([torch.zeros(1, dtype=torch.float8_e4m3fn, device="cuda")] * 2)
except Exception:
    def _byteop(orig):
        def f(tensors, dim=0, *, out=None):
            ts = list(tensors)
            if ts and ts[0].dtype in _FP8:
                dt = ts[0].dtype
                return orig([t.view(torch.uint8) for t in ts], dim=dim).view(dt)
            return orig(ts, dim=dim) if out is None else orig(ts, dim=dim, out=out)
        return f
    # BOTH are needed: transformers' MergeModulelist stacks expert shards with torch.stack
    # (core_model_loading.py), and stack dispatches to the same missing cat_cuda kernel.
    torch.cat, torch.stack = _byteop(torch.cat), _byteop(torch.stack)
    print("  [compat] shimmed torch.cat and torch.stack for float8 "
          "(no CUDA kernel; byte-exact via uint8 view)", flush=True)



# transformers' FP8 path calls torch.cuda.get_device_properties() with no argument to check the
# compute capability. torch 2.5.1 types the parameter as Optional but gives it no default, so it is
# still positionally required and the call raises TypeError. Defaulting to the current device is what
# the caller means. (I misread the Optional annotation as implying a default -- it does not.)
if not hasattr(torch.cuda, "_gdp_orig"):
    import inspect as _insp
    if _insp.signature(torch.cuda.get_device_properties).parameters["device"].default is _insp.Parameter.empty:
        torch.cuda._gdp_orig = torch.cuda.get_device_properties
        torch.cuda.get_device_properties = lambda device=None: torch.cuda._gdp_orig(
            torch.cuda.current_device() if device is None else device)
        print("  [compat] shimmed torch.cuda.get_device_properties(device=None)", flush=True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402


@torch.no_grad()
def throughput(model, V, batches, seq, reps=3):
    out = {}
    for b in batches:
        ids = torch.randint(0, V, (b, seq), device="cuda")
        try:
            torch.cuda.empty_cache()
            o = model(ids); del o; torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                o = model(ids); del o
            torch.cuda.synchronize()
            out[b] = ids.numel() / ((time.time() - t0) / reps)
        except torch.OutOfMemoryError:
            out[b] = None
            torch.cuda.empty_cache()
        del ids
    return out


@torch.no_grad()
def bpb(model, slice_ids, divisor, n_seq):
    tot = ntok = 0
    for i in range(n_seq):
        b = slice_ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor


def load(path, family, fp8, mode="weight_only"):
    """bf16 load, then our own expert quantisation -- no quantisation library involved.

    Going through transformers' FineGrainedFP8 was six blockers deep and the fifth fix broke
    transformers at import. Every off-the-shelf alternative (bitsandbytes, torchao, FbgemmFp8)
    converts nn.Linear and so skips the 3-D expert tensors, which are ~90% of these models. Doing it
    directly means the numerics are ours to verify.
    """
    import fp8_experts as FQ
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    RQ.install(family)
    RQ.tag_layers(model)
    if fp8:
        FQ.quantize_experts_fp8(model)
        FQ.install_fp8_forward(mode=mode)
    else:
        FQ.restore()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/qwen3-30b")
    ap.add_argument("--family", default="qwen3")
    ap.add_argument("--data", default="/workspace/qwen3moe-adapt/data")
    ap.add_argument("--slice-name", default="qwen3")
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batches", default="8,16,32,64")
    ap.add_argument("--precisions", default="bf16,fp8,fp8_scaled")
    A = ap.parse_args()

    meta = json.load(open(f"{A.data}/bpb_slice_meta_{A.slice_name}.json"))
    D = meta["divisor_D"]
    sl = torch.load(f"{A.data}/bpb_slice_ids_{A.slice_name}.pt", weights_only=False)[: A.n_seq]
    batches = [int(x) for x in A.batches.split(",")]

    res = {}
    for prec in A.precisions.split(","):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, tok = load(A.model, A.family, fp8=prec.startswith("fp8"),
                          mode=({"fp8_scaled":"scaled_mm","fp8_grouped":"grouped"}.get(prec,"weight_only")))
        wmem = torch.cuda.memory_allocated() / 1e9
        # residency OFF: this compares precisions, not the constraint
        RES._CFG.update(on=True, R=8, collect_telem=False)
        RES.set_free_layers(list(range(model.config.num_hidden_layers)))
        b = bpb(model, sl, D, A.n_seq)
        tp = throughput(model, model.config.vocab_size, batches, A.seq)
        res[prec] = {"weights_GB": wmem, "bpb": b, "tps": tp, "load_s": time.time() - t0}
        print(f"  {prec:5} weights {wmem:5.1f} GB  BPB {b:.6f}  load {time.time()-t0:.0f}s", flush=True)
        print(f"        " + "  ".join(f"bs{k}={v:,.0f}" if v else f"bs{k}=OOM" for k, v in tp.items()),
              flush=True)
        del model
        torch.cuda.empty_cache()

    for _v in [k for k in res if k != "bf16"]:
      if "bf16" in res:
        res["fp8"] = res[_v]; print(f"\n  --- {_v} ---", flush=True)
        d = res[_v]["bpb"] - res["bf16"]["bpb"]
        print(f"\n  === FP8 vs bf16 ===", flush=True)
        print(f"  weights   {res['bf16']['weights_GB']:.1f} GB -> {res['fp8']['weights_GB']:.1f} GB "
              f"({res['bf16']['weights_GB']-res['fp8']['weights_GB']:.1f} GB freed)", flush=True)
        print(f"  BPB       {res['bf16']['bpb']:.6f} -> {res['fp8']['bpb']:.6f}  delta {d:+.6f}", flush=True)
        for b in batches:
            a, c = res["bf16"]["tps"].get(b), res["fp8"]["tps"].get(b)
            if a and c:
                print(f"  bs={b:<4} {a:,.0f} -> {c:,.0f} tok/s  ({c/a:.2f}x)", flush=True)
            elif c and not a:
                print(f"  bs={b:<4} bf16 OOM -> fp8 {c:,.0f} tok/s  (fp8 fits where bf16 does not)",
                      flush=True)
        print(f"\n  Reference scale: the aux correction was 4.85e-04 BPB and free-set differences are"
              f"\n  ~2.5e-03. A BPB delta of {abs(d):.2e} is {'BELOW' if abs(d) < 2e-4 else 'COMPARABLE TO'}"
              f" those effects.", flush=True)
        print(f"  VERDICT: FP8 {'is usable for residency measurements' if abs(d) < 2e-4 else 'needs the downstream check before adoption'}",
              flush=True)
    print("=== FP8 BENCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
