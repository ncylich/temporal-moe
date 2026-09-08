#!/usr/bin/env python3
"""Antisymmetric router noise on a half-grain split checkpoint (in place):
row 2i += eps, row 2i+1 -= eps, eps ~ N(0, (rel*row-tensor std)^2). Breaks the
duplicated-pair logit ties so training desymmetrizes from a decisive start.
Usage: inject_router_noise.py <model_dir> <key_suffix> <rel_std> <seed>"""
import json
import sys

import torch
from safetensors.torch import load_file, save_file

SRC, SUFFIX, REL, SEED = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
torch.manual_seed(SEED)
ix = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
shards = sorted({sh for k, sh in ix.items() if k.endswith(SUFFIX)})
touched = 0
for sh in shards:
    t = load_file(f"{SRC}/{sh}")
    dirty = False
    for k, v in list(t.items()):
        if k.endswith(SUFFIX):
            E2 = v.shape[0]
            eps = torch.randn(E2 // 2, v.shape[1], dtype=torch.float32) * (REL * v.float().std())
            v32 = v.float().view(E2 // 2, 2, -1)
            v32[:, 0] += eps
            v32[:, 1] -= eps
            t[k] = v32.view(E2, -1).to(v.dtype).contiguous()
            dirty = True
            touched += 1
    if dirty:
        save_file(t, f"{SRC}/{sh}", metadata={"format": "pt"})
print(f"[noise] {touched} layers, suffix {SUFFIX}, rel {REL}, seed {SEED}", flush=True)
