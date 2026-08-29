#!/usr/bin/env python3
"""Apply a trained adapter to a live vLLM engine, no merged checkpoint on disk.

Why. The merge stage wrote a full 49 GB checkpoint (base + delta on every expert tensor) so
the eval engine could load it: ~2 min per adapter, 49 GB of disk each, and a 49 GB reload.
The engine can instead load the BASE (from /dev/shm, ~10 s) and receive `base + delta` for
the trained surfaces straight from the adapter file: the same arithmetic as the disk merge
(and as online_sampler.py's sync), bit-exact, ~20 s, nothing written.

Surfaces in a train_gemma_ce.py adapter (family gemma4, hf+peft stack):
  experts   elora_{gu,dp}_{A,B} per layer, bf16, grouped layout (E,in,out); merged =
            W_grouped + 2.0 * bmm(A,B), transposed back to the checkpoint's (E,out,in)
  attention peft lora_{A,B}.default.weight, fp32, on q/k/v/o; merged = W + (B@A) * (alpha/r)
            added with ONE rounding (peft's in-place +=), never via a bf16 delta
  full      router.proj.weight, every *norm.weight of the language model: the tensors themselves
Vision-tower entries (never trained; B == 0) are ignored.

    apply_adapter.py --base /dev/shm/gemma4-26b-it --adapter X.pt --check /root/models/X-merged
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch                                                         # noqa: E402

PREFIX = "base_model.model."


def find_engine_model(llm):
    for path in ("llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.engine_core.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.model_runner.model"):
        obj = llm
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if hasattr(obj, "load_weights"):
                return obj
        except AttributeError:
            continue
    raise RuntimeError("[apply_adapter] cannot locate the in-process vLLM model object")


class _Base:
    """Named tensors of the base checkpoint, read on demand straight to the GPU."""
    def __init__(self, base_path):
        self.d = base_path
        self.map = json.load(open(f"{base_path}/model.safetensors.index.json"))["weight_map"]
        self._h = {}

    def get(self, name):
        from safetensors import safe_open
        if name not in self.map and name.startswith("model."):
            name = "model.language_model." + name[len("model."):]       # raw qwen3.5 checkpoint names
        f = self.map[name]
        if f not in self._h:
            self._h[f] = safe_open(f"{self.d}/{f}", "pt", device="cuda")
        return self._h[f].get_tensor(name)


def adapter_pairs_qwen35(ck_tensors, base, meta=None):
    """qwen3.5 on the RAW base (model.language_model.layers.N..., experts fused (E,2I,H)/(E,H,I)
    exactly like gemma; router at mlp.gate.weight; attention LoRA on the full-attention layers).
    The trainer names are model.layers.N... (text-only load); vLLM's raw-base engine takes the
    model.language_model.* names."""
    escale = 2.0; lscale = 64 / 32
    t = {n[len(PREFIX):] if n.startswith(PREFIX) else n: v for n, v in ck_tensors.items()
         if "visual" not in n and "mtp" not in n}
    t = {re.sub(r"^model\.language_model\.", "model.", n): v for n, v in t.items()}
    layers = sorted({int(m.group(1)) for n in t for m in [re.search(r"\.layers\.(\d+)\.", n)] if m})
    with torch.no_grad():
        for L in layers:
            p = f"model.layers.{L}."; raw = f"model.language_model.layers.{L}."
            a = t.get(p + "mlp.experts.elora_gu_A")
            if a is not None:
                A = a.cuda(); B = t[p + "mlp.experts.elora_gu_B"].cuda()
                W = base.get(raw + "mlp.experts.gate_up_proj").transpose(1, 2).contiguous()    # (E,H,2I) grouped
                W += escale * torch.bmm(A, B)
                yield raw + "mlp.experts.gate_up_proj", W.transpose(1, 2).contiguous()
                del W, A, B
                A = t[p + "mlp.experts.elora_dp_A"].cuda(); B = t[p + "mlp.experts.elora_dp_B"].cuda()
                W = base.get(raw + "mlp.experts.down_proj").transpose(1, 2).contiguous()       # (E,I,H) grouped
                W += escale * torch.bmm(A, B)
                yield raw + "mlp.experts.down_proj", W.transpose(1, 2).contiguous()
                del W, A, B
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                a = t.get(p + f"self_attn.{proj}.lora_A.default.weight")
                if a is None:
                    continue
                A = a.cuda(); B = t[p + f"self_attn.{proj}.lora_B.default.weight"].cuda()
                W = base.get(raw + f"self_attn.{proj}.weight").clone()
                W += (B @ A) * lscale
                yield raw + f"self_attn.{proj}.weight", W
                del W, A, B
            for n, v in t.items():
                if n.startswith(p) and "lora_" not in n and "elora_" not in n:
                    yield raw + n[len(p):], v.cuda()
        if "model.norm.weight" in t:
            yield "model.language_model.norm.weight", t["model.norm.weight"].cuda()


def adapter_pairs(ck_tensors, base, meta=None):
    if (meta or {}).get("family") == "qwen35":
        yield from adapter_pairs_qwen35(ck_tensors, base, meta)
        return
    """Yield (hf_name, merged tensor) for every trained language-model surface."""
    r = int((meta or {}).get("expert_lora_r", 32))
    escale = 2.0                                     # elora alpha = 2r  ->  alpha/r
    lscale = 64 / 32                                 # peft LoraConfig(r=32, lora_alpha=64)
    t = {n[len(PREFIX):] if n.startswith(PREFIX) else n: v for n, v in ck_tensors.items()
         if "vision" not in n and "embed_vision" not in n}
    layers = sorted({int(m.group(1)) for n in t for m in [re.search(r"\.layers\.(\d+)\.", n)] if m})
    with torch.no_grad():
        for L in layers:
            p = f"model.language_model.layers.{L}."
            a = t.get(p + "experts.elora_gu_A")
            if a is not None:
                A = a.cuda(); B = t[p + "experts.elora_gu_B"].cuda()
                W = base.get(p + "experts.gate_up_proj").transpose(1, 2).contiguous()       # (E,H,2I) grouped
                W += escale * torch.bmm(A, B)
                yield p + "experts.gate_up_proj", W.transpose(1, 2).contiguous()
                del W, A, B
                A = t[p + "experts.elora_dp_A"].cuda(); B = t[p + "experts.elora_dp_B"].cuda()
                W = base.get(p + "experts.down_proj").transpose(1, 2).contiguous()          # (E,I,H) grouped
                W += escale * torch.bmm(A, B)
                yield p + "experts.down_proj", W.transpose(1, 2).contiguous()
                del W, A, B
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                a = t.get(p + f"self_attn.{proj}.lora_A.default.weight")
                if a is None:
                    continue
                A = a.cuda(); B = t[p + f"self_attn.{proj}.lora_B.default.weight"].cuda()
                W = base.get(p + f"self_attn.{proj}.weight").clone()
                W += (B @ A) * lscale                                                        # fp32 delta, one rounding
                yield p + f"self_attn.{proj}.weight", W
                del W, A, B
            for n, v in t.items():
                if n.startswith(p) and "lora_" not in n and "elora_" not in n:
                    yield n, v.cuda()
        if "model.language_model.norm.weight" in t:
            yield "model.language_model.norm.weight", t["model.language_model.norm.weight"].cuda()


def apply_adapter(llm, adapter_path, base_path):
    t0 = time.time()
    ck = torch.load(adapter_path, map_location="cpu", weights_only=False)
    assert ck.get("family", "gemma4") in ("gemma4", "qwen35") and ck.get("stack", "hf+peft") == "hf+peft", ck.get("family")
    vm = find_engine_model(llm)
    base = _Base(base_path)
    loaded = vm.load_weights(adapter_pairs(ck["tensors"], base, ck))
    torch.cuda.synchronize()
    print(f"[apply_adapter] {os.path.basename(adapter_path)} (seen={ck.get('seen', 0)/1e6:.2f}M) applied to the engine: "
          f"{len(loaded)} engine params in {time.time()-t0:.1f}s", flush=True)
    return loaded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/dev/shm/gemma4-26b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--check", required=True, help="merged checkpoint dir to compare engine tensors against")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    A = ap.parse_args()
    import vllm_glue
    vllm_glue.install()
    from vllm import LLM
    llm = LLM(model=A.base, **vllm_glue.llm_kwargs(), gpu_memory_utilization=A.gpu_mem, max_model_len=2048)
    apply_adapter(llm, A.adapter, A.base)
    vm = find_engine_model(llm); vp = dict(vm.named_parameters())
    from safetensors import safe_open
    import glob
    # Stream each shard sequentially by byte offset (no mmap page faults: per-expert get_tensor
    # ran at ~50 MB/s, and loading the whole 70 GB checkpoint took the container to 187 of
    # 233 GiB) and diff every tensor against its engine slice on the GPU.
    import json, struct
    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    eng = {}
    for name, p in vp.items():
        m = re.search(r"layers\.(\d+)\.(.*)$", name)
        if not m:
            continue
        L, tail = int(m.group(1)), m.group(2)
        if tail.endswith("qkv_proj.weight"): eng[(L, "qkv")] = p
        elif tail.endswith("self_attn.o_proj.weight") or tail.endswith("attn.o_proj.weight"): eng[(L, "o")] = p
        elif tail.endswith("router.proj.weight") or tail.endswith("mlp.gate.weight"): eng[(L, "router")] = p
        elif "w13" in tail: eng[(L, "w13")] = p
        elif "w2" in tail and "experts" in tail: eng[(L, "w2")] = p
        elif tail.endswith("norm.weight"): eng[(L, tail)] = p
    worst, n, skipped = 0.0, 0, 0
    for f in sorted(glob.glob(f"{A.check}/*.safetensors")):
        with open(f, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(hlen)); hdr.pop("__metadata__", None)
            shapes = {k: v["shape"] for k, v in hdr.items()}
            def rows(L, x):                             # q/k/v row counts from the header
                for k in (f"model.layers.{L}.self_attn.{x}_proj.weight", f"model.language_model.layers.{L}.self_attn.{x}_proj.weight"):
                    if k in shapes: return shapes[k][0]
                return 0
            for k in sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0]):
                o0, o1 = hdr[k]["data_offsets"]
                m = re.match(r"model\.(?:language_model\.)?layers\.(\d+)\.(.*)$", k)
                if not m or "vision" in k:
                    continue
                L, rest = int(m.group(1)), m.group(2)
                target = None
                mm = re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", rest)
                if mm:
                    e, kind = int(mm.group(1)), mm.group(2); I = shapes[k][0] if kind != "down" else shapes[k][1]
                    if kind == "gate" and (L, "w13") in eng: target = eng[(L, "w13")][e, :I]
                    elif kind == "up" and (L, "w13") in eng: target = eng[(L, "w13")][e, I:]
                    elif kind == "down" and (L, "w2") in eng: target = eng[(L, "w2")][e]
                elif rest.endswith("experts.gate_up_proj"): target = eng.get((L, "w13"))
                elif rest.endswith("experts.down_proj"): target = eng.get((L, "w2"))
                elif rest.endswith("self_attn.q_proj.weight") and (L, "qkv") in eng: target = eng[(L, "qkv")][:rows(L, "q")]
                elif rest.endswith("self_attn.k_proj.weight") and (L, "qkv") in eng: target = eng[(L, "qkv")][rows(L, "q"):rows(L, "q") + rows(L, "k")]
                elif rest.endswith("self_attn.v_proj.weight") and (L, "qkv") in eng: target = eng[(L, "qkv")][rows(L, "q") + rows(L, "k"):]
                elif rest.endswith("self_attn.o_proj.weight"): target = eng.get((L, "o"))
                elif rest.endswith("router.proj.weight") or rest.endswith("mlp.gate.weight"): target = eng.get((L, "router"))
                elif rest.endswith("norm.weight"): target = eng.get((L, rest))
                if target is None:
                    skipped += 1; continue
                fh.seek(8 + hlen + o0); buf = fh.read(o1 - o0)
                want = torch.frombuffer(bytearray(buf), dtype=DT[hdr[k]["dtype"]]).reshape(shapes[k])
                if tuple(want.shape) != tuple(target.shape):
                    print(f"[apply_adapter-check] SHAPE {k}: ck {tuple(want.shape)} vs engine {tuple(target.shape)}", flush=True); skipped += 1; continue
                d = (target.detach().float() - want.cuda().float()).abs().max().item(); worst = max(worst, d); n += 1
                if d != 0:
                    print(f"[apply_adapter-check] MISMATCH {k}: max diff {d:.3e}", flush=True)
    print(f"[apply_adapter-check] skipped {skipped} checkpoint tensors with no adapted engine target", flush=True)
    print(f"[apply_adapter-check] {n} checkpoint tensors compared against {A.check}: worst max|diff| = {worst:.3e} -> "
          f"{'EXACT' if worst == 0 else 'NOT EXACT'}", flush=True)
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
