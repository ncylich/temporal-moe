#!/usr/bin/env python3
"""Function-preserving expert split ("half-grain" relabeling), qwen3.5/gemma4.

Each expert is cut in half along its intermediate dimension and the router row
is duplicated, top-k doubles. Duplicated logits mean the top-2k renormalised
softmax gives each half HALF the original weight, so each half's down_proj is
multiplied by 2: the composed function is exactly the original model. Halves
are interleaved (expert 2i / 2i+1 = halves of original expert i). Residency
then operates at half-expert grain (2E experts, R doubles at matched fraction,
each swap moves half the bytes).

Streams shard-by-shard; only mlp.experts.{gate_up_proj,down_proj} and
mlp.gate.weight are transformed, everything else is copied through.

Family differences: the top-k config field (num_experts_per_tok vs
top_k_experts) and the router tensors (qwen: mlp.gate.weight row-duplicated;
gemma: router.proj.weight row-duplicated AND router.per_expert_scale
duplicated, not doubled — the exactness factor 2 lives in down_proj).

    split_experts.py --family qwen35 --src <model dir> --dst <out dir>
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=("qwen35", "gemma4"))
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--partition", default=None,
                help="npz from split_stats.py: per-layer per-expert channel "
                     "permutations; halves become the top/bottom halves of the "
                     "principal co-activation ordering instead of contiguous "
                     "channel ranges (still a permutation, still exact)")
A = ap.parse_args()
os.makedirs(A.dst, exist_ok=True)
TOPK_FIELD = {"qwen35": "num_experts_per_tok", "gemma4": "top_k_experts"}[A.family]

cfg = json.load(open(os.path.join(A.src, "config.json")))
tc = cfg["text_config"]
E, K, D = tc["num_experts"], tc[TOPK_FIELD], tc["moe_intermediate_size"]
assert D % 2 == 0
tc["num_experts"], tc[TOPK_FIELD], tc["moe_intermediate_size"] = 2 * E, 2 * K, D // 2
json.dump(cfg, open(os.path.join(A.dst, "config.json"), "w"), indent=2)
print(f"[split] E {E}->{2*E}, k {K}->{2*K}, d_i {D}->{D//2}", flush=True)

PERMS = None
if A.partition:
    import numpy as np
    import re as _re
    PERMS = torch.from_numpy(np.load(A.partition)["perms"]).long()
    assert PERMS.shape[1] == E and PERMS.shape[2] == D, PERMS.shape

    def layer_of(key):
        return int(_re.search(r"layers\.(\d+)\.", key).group(1))

ix = json.load(open(os.path.join(A.src, "model.safetensors.index.json")))
total = 0
for shard in sorted(set(ix["weight_map"].values())):
    t = load_file(os.path.join(A.src, shard))
    out = {}
    for k, v in t.items():
        if k.endswith("experts.gate_up_proj"):
            e, twod, h = v.shape
            assert e == E and twod == 2 * D
            gate, up = v[:, :D], v[:, D:]
            if PERMS is not None:
                p = PERMS[layer_of(k)][:, :, None].expand(-1, -1, h)
                gate = gate.gather(1, p)
                up = up.gather(1, p)
            newA = torch.cat([gate[:, : D // 2], up[:, : D // 2]], dim=1)
            newB = torch.cat([gate[:, D // 2:], up[:, D // 2:]], dim=1)
            v = torch.stack([newA, newB], dim=1).reshape(2 * E, D, h)
        elif k.endswith("experts.down_proj"):
            e, h, d = v.shape
            assert e == E and d == D
            if PERMS is not None:
                p = PERMS[layer_of(k)][:, None, :].expand(-1, h, -1)
                v = v.gather(2, p)
            v = 2.0 * torch.stack([v[:, :, : D // 2], v[:, :, D // 2:]], dim=1)
            v = v.reshape(2 * E, h, D // 2)
        elif k.endswith("mlp.gate.weight") or k.endswith("router.proj.weight") \
                or k.endswith("router.per_expert_scale"):
            assert v.shape[0] == E
            v = v.repeat_interleave(2, dim=0)
        out[k] = v.contiguous()
        total += v.numel() * v.element_size()
    save_file(out, os.path.join(A.dst, shard), metadata={"format": "pt"})
    print(f"[split] wrote {shard}", flush=True)

ix["metadata"]["total_size"] = total
json.dump(ix, open(os.path.join(A.dst, "model.safetensors.index.json"), "w"))
for f in os.listdir(A.src):
    if (f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json")) \
            or f.endswith(".jinja") or f.endswith(".txt"):
        shutil.copy(os.path.join(A.src, f), os.path.join(A.dst, f))
print("[split] DONE", flush=True)
