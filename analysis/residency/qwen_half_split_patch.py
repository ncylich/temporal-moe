#!/usr/bin/env python3
"""Streaming split+patch for the qwen half-grain adapter: reads the ORIGINAL
composite checkpoint shard-by-shard, applies the function-preserving half-grain
split (same transform as split_experts.py), then applies the adapter deltas
(trained in split-space), writing a merged split checkpoint. Source and merged
never coexist in /dev/shm.

Env: ADAPTER_PATH, DST_PATH."""
import json
import os
import re
import shutil

import torch
from safetensors.torch import safe_open, save_file

SRC = "/workspace/instruct-models/qwen35-35b-a3b-instruct"
DST = os.environ["DST_PATH"]
os.makedirs(DST, exist_ok=True)

ck = torch.load(os.environ["ADAPTER_PATH"], map_location="cpu", weights_only=False)
# normalize adapter keys to start at "layers.N...."
T = {}
for n, t in ck["tensors"].items():
    m = re.search(r"layers\.\d+\..*$", n)
    if m:
        T[m.group(0)] = t.float()
    elif n.startswith("base_model.model.model."):
        # non-layer trainables (e.g. final model.norm.weight): text-view name
        # minus the peft+text prefix == composite name minus "model.language_model."
        T[n[len("base_model.model.model."):]] = t.float()
    else:
        raise AssertionError(f"unmapped adapter key {n}")
print(f"[qsp] adapter tensors: {len(T)}; sample keys: {sorted(T)[:3]}", flush=True)
used = set()
ELORA_S, PEFT_S = 2.0, 2.0

cfg = json.load(open(f"{SRC}/config.json"))
tc = cfg["text_config"]
E, D = tc["num_experts"], tc["moe_intermediate_size"]
tc["num_experts"], tc["num_experts_per_tok"], tc["moe_intermediate_size"] = \
    2 * E, 2 * tc["num_experts_per_tok"], D // 2
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)


PERMS = None
if os.environ.get("PARTITION"):
    import numpy as np
    PERMS = torch.from_numpy(np.load(os.environ["PARTITION"])["perms"]).long()
ROTS = None
if os.environ.get("ROTATION"):
    import numpy as np
    ROTS = torch.from_numpy(np.load(os.environ["ROTATION"])["rotations"]).float()


def _layer_of(key):
    return int(re.search(r"layers\.(\d+)\.", key).group(1))


def split_tensor(key, v):
    """Half-grain transform for expert/router tensors; identity otherwise.
    With PARTITION set, applies the same per-expert channel permutation the
    training split used (must match, or adapter deltas land on wrong channels)."""
    if key.endswith("mlp.experts.gate_up_proj"):
        gate, up = v[:, :D], v[:, D:]
        if ROTS is not None:
            Rl = ROTS[_layer_of(key)]
            gate = torch.bmm(Rl, gate.float()).to(gate.dtype)
            up = torch.bmm(Rl, up.float()).to(up.dtype)
        if PERMS is not None:
            p = PERMS[_layer_of(key)][:, :, None].expand(-1, -1, v.shape[2])
            gate, up = gate.gather(1, p), up.gather(1, p)
        newA = torch.cat([gate[:, : D // 2], up[:, : D // 2]], dim=1)
        newB = torch.cat([gate[:, D // 2:], up[:, D // 2:]], dim=1)
        return torch.stack([newA, newB], 1).reshape(2 * E, D, v.shape[2])
    if key.endswith("mlp.experts.down_proj"):
        if ROTS is not None:
            Rl = ROTS[_layer_of(key)]
            v = torch.bmm(v.float(), Rl.transpose(1, 2)).to(v.dtype)
        if PERMS is not None:
            p = PERMS[_layer_of(key)][:, None, :].expand(-1, v.shape[1], -1)
            v = v.gather(2, p)
        return (2.0 * torch.stack([v[:, :, : D // 2], v[:, :, D // 2:]], 1)
                ).reshape(2 * E, v.shape[1], D // 2)
    if key.endswith("mlp.gate.weight"):
        return v.repeat_interleave(2, dim=0)
    return v


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
        return ("add", ELORA_S * torch.bmm(A, B).transpose(1, 2))
    if k.endswith("mlp.experts.down_proj"):
        L = k.split(".")[1]
        A = T.get(f"layers.{L}.mlp.experts.elora_dp_A")
        B = T.get(f"layers.{L}.mlp.experts.elora_dp_B")
        if A is None:
            return None
        used.update({f"layers.{L}.mlp.experts.elora_dp_A", f"layers.{L}.mlp.experts.elora_dp_B"})
        return ("add", ELORA_S * torch.bmm(A, B).transpose(1, 2))
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
            t = split_tensor(key, f.get_tensor(key))
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
    print(f"[qsp] shard {si+1}/{len(shards)} done", flush=True)

idx["metadata"]["total_size"] = total
json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"))
for f in os.listdir(SRC):
    p = f"{SRC}/{f}"
    if (not f.endswith(".safetensors") and f not in ("config.json", "model.safetensors.index.json")
            and not f.startswith(".") and os.path.isfile(p)):
        shutil.copy(p, f"{DST}/{f}")
unused = set(T) - used
print(f"[qsp] patched: {n_add} adds, {n_rep} replaces; unused adapter keys: {len(unused)}",
      flush=True)
if unused:
    print("[qsp] UNUSED sample:", sorted(unused)[:6], flush=True)
assert n_add > 0 and n_rep > 0 and not unused, "adapter not fully consumed"
print("[qsp] DONE", flush=True)
