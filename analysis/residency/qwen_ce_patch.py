#!/usr/bin/env python3
"""Streaming patch-onto-base-copy merge for a qwen35 CE-adaptation expert-LoRA
adapter (e.g. qwen_ce_d12r2_adapter.pt). Reads the ORIGINAL multimodal
composite checkpoint shard-by-shard, applies the adapter's expert-LoRA deltas
(and any full-precision replaced tensors) to the text-side layers only, and
copies every other tensor -- including the vision tower -- through unchanged.

Root cause this works around: train_gemma_ce.py's --merge-out path only ever
has the text-only submodule in memory (--no-unsloth/HF+peft loads just that),
so its save_pretrained() never writes a vision tower, and the resulting
checkpoint is unservable by this vLLM version regardless of which config is
paired with it (see TODO.md section 6). This script never loads the model
into memory at all -- it only ever holds one shard's tensors, patched or
copied verbatim -- so the vision tower survives because it was never dropped
in the first place.

Env: ADAPTER_PATH, DST_PATH. SRC is the qwen35 instruct base on /workspace."""
import json
import os
import shutil

import torch
from safetensors.torch import safe_open, save_file

SRC = "/workspace/instruct-models/qwen35-35b-a3b-instruct"
DST = os.environ["DST_PATH"]
os.makedirs(DST, exist_ok=True)

ck = torch.load(os.environ["ADAPTER_PATH"], map_location="cpu", weights_only=False)
assert ck["family"] == "qwen35", f"adapter family {ck['family']} != qwen35"
ELORA_S = 2.0  # alpha/r with alpha = 2r, must match train_gemma_ce.py's elora_scale
PEFT_S = 2.0   # alpha/r = 64/32 for the attention q/k/v/o_proj LoRA, confirmed
               # against the adapter's own q_proj.lora_A shape (32, 2048) -> r=32

T = {}
for n, t in ck["tensors"].items():
    assert n.startswith("base_model.model.model."), f"unexpected adapter key {n}"
    T[n[len("base_model.model.model."):]] = t.float()
print(f"[qcp] adapter tensors: {len(T)}; sample keys: {sorted(T)[:3]}", flush=True)
used = set()


def deltas_for(key):
    if not key.startswith("model.language_model."):
        return None
    k = key[len("model.language_model."):]
    if k.endswith("mlp.experts.gate_up_proj"):
        L = k.split(".")[1]
        A = T.get(f"layers.{L}.mlp.experts.elora_gu_A")
        B = T.get(f"layers.{L}.mlp.experts.elora_gu_B")
        if A is None:
            return None
        used.update({f"layers.{L}.mlp.experts.elora_gu_A", f"layers.{L}.mlp.experts.elora_gu_B"})
        # bmm(A, B) is (E, H, 2I) in the live nn.Parameter layout
        # train_gemma_ce.py's in-memory merge writes to; the on-disk grouped-GEMM
        # safetensors layout is the transpose, (E, 2I, H) -- matches
        # qwen_half_split_patch.py's identical transpose for the same tensor.
        return ("add", (ELORA_S * torch.bmm(A, B)).transpose(1, 2))
    if k.endswith("mlp.experts.down_proj"):
        L = k.split(".")[1]
        A = T.get(f"layers.{L}.mlp.experts.elora_dp_A")
        B = T.get(f"layers.{L}.mlp.experts.elora_dp_B")
        if A is None:
            return None
        used.update({f"layers.{L}.mlp.experts.elora_dp_A", f"layers.{L}.mlp.experts.elora_dp_B"})
        return ("add", (ELORA_S * torch.bmm(A, B)).transpose(1, 2))
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if k.endswith(f"self_attn.{proj}.weight"):
            stem = k[:-len(".weight")]
            A = T.get(f"{stem}.lora_A.default.weight")
            B = T.get(f"{stem}.lora_B.default.weight")
            if A is None:
                return None
            used.update({f"{stem}.lora_A.default.weight", f"{stem}.lora_B.default.weight"})
            return ("add", PEFT_S * (B @ A))
    if k in T:
        used.add(k)
        return ("replace", T[k])
    return None


idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
n_add = n_rep = 0
total = 0
for si, sh in enumerate(shards):
    out = {}
    with safe_open(f"{SRC}/{sh}", framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            d = deltas_for(key)
            if d is None:
                out[key] = t.contiguous()
            else:
                assert d[1].shape == t.shape, (key, d[1].shape, t.shape)
                if d[0] == "add":
                    assert float(d[1].abs().max()) > 0, f"zero delta {key}"
                    out[key] = (t.float() + d[1]).to(t.dtype).contiguous()
                    n_add += 1
                else:
                    out[key] = d[1].to(t.dtype).contiguous()
                    n_rep += 1
            total += out[key].numel() * out[key].element_size()
    save_file(out, f"{DST}/{sh}", metadata={"format": "pt"})
    print(f"[qcp] shard {si+1}/{len(shards)} done", flush=True)

idx["metadata"]["total_size"] = total
json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"))
for f in os.listdir(SRC):
    p = f"{SRC}/{f}"
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json" \
            and not f.startswith(".") and os.path.isfile(p):
        shutil.copy(p, f"{DST}/{f}")
unused = set(T) - used
print(f"[qcp] patched: {n_add} adds, {n_rep} replaces; unused adapter keys: {len(unused)}",
      flush=True)
if unused:
    print("[qcp] UNUSED sample:", sorted(unused)[:6], flush=True)
assert n_add > 0 and n_rep > 0 and not unused, "adapter not fully consumed"
print("[qcp] DONE", flush=True)
