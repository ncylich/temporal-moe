#!/usr/bin/env python3
"""Functional displacement for gpt-oss-120b via vLLM (MXFP4-native; HF dequant
infeasible under the 251GB cgroup). Patches MLPBlock.forward in the gpt-oss
model file: per prefill chunk, runs the FusedMoE experts twice on the same
input (free logits vs residency-masked logits, same scan as the W1 run) and
accumulates the same metrics as functional_displacement.py into the same CSV.
Decode steps (T==1) are skipped; only prefill chunks are measured."""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/temporal-moe/analysis/residency")
sys.path.insert(0, "/workspace/temporal-moe/analysis")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import granularity_ladder as GL                                      # noqa: E402
from paths import ABLATIONS                                          # noqa: E402

MODEL = "/dev/shm/gpt-oss-120b"
PROMPTS = "/workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl"
R, N = 4, 100

import vllm.model_executor.models.gpt_oss as G                       # noqa: E402
_orig_fwd = G.MLPBlock.forward

ACC = {}   # layer_idx -> dict of float sums
USE = {}   # layer_idx -> (usage_free, usage_masked) float64 cuda
STATE = {"on": False, "ntok": 0}


def _acc(li, key, val):
    ACC.setdefault(li, {})[key] = ACC.get(li, {}).get(key, 0.0) + val


def wrapped(self, x):
    if not STATE["on"] or x.shape[0] <= 1:
        return _orig_fwd(self, x)
    g = self.router(x)
    lg = g.float()
    y_f = self.experts(hidden_states=x.clone(), router_logits=g)[:, : self.hidden_size]
    mask = GL.compute_resident_mask_accel(
        lg.unsqueeze(1), R, evict="min_logit").squeeze(1).bool()
    g_m = g.masked_fill(~mask, float("-inf"))
    y_m = self.experts(hidden_states=x.clone(), router_logits=g_m)[:, : self.hidden_size]
    li = self.layer_idx
    yf, ym = y_f.float(), y_m.float()
    _acc(li, "dnorm", (ym - yf).norm(dim=-1).sum().item())
    _acc(li, "ynorm", yf.norm(dim=-1).sum().item())
    _acc(li, "xnorm", x[:, : self.hidden_size].float().norm(dim=-1).sum().item())
    _acc(li, "cos", torch.nn.functional.cosine_similarity(ym, yf, dim=-1).sum().item())
    pf = torch.softmax(lg, -1)
    q = pf * mask
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
    if li not in USE:
        USE[li] = (torch.zeros(pf.shape[-1], dtype=torch.float64, device=pf.device),
                   torch.zeros(pf.shape[-1], dtype=torch.float64, device=pf.device))
    USE[li][0].add_(pf.sum(0).double())
    USE[li][1].add_(q.sum(0).double())
    if li == 0:
        STATE["ntok"] += x.shape[0]
    return y_f


G.MLPBlock.forward = wrapped

from vllm import LLM, SamplingParams                                 # noqa: E402
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.92,
          max_model_len=1024, enable_prefix_caching=False)
sp = SamplingParams(max_tokens=1, temperature=0)

prompts = [json.loads(l)["text"] for l in open(PROMPTS)][:N]
STATE["on"] = True
for pi, text in enumerate(prompts):
    llm.chat([[{"role": "user", "content": text}]], sp, use_tqdm=False)
    if pi % 25 == 0:
        print(f"[fd-vllm] {pi}/{N}", flush=True)
STATE["on"] = False

L = len(ACC)
ntok = STATE["ntok"]
uf = np.stack([(USE[li][0] / ntok).cpu().numpy() for li in sorted(USE)])
um = np.stack([(USE[li][1] / ntok).cpu().numpy() for li in sorted(USE)])
usage_tv = 0.5 * np.abs(uf - um).sum(-1)
np.savez(os.path.join(ABLATIONS, "functional_displacement_usage_gpt-oss-120b.npz"),
         usage_free=uf, usage_masked=um, R=R, n_tokens=ntok)

out = os.path.join(ABLATIONS, "functional_displacement.csv")
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    for i, li in enumerate(sorted(ACC)):
        a = ACC[li]
        w.writerow(["gpt-oss-120b", "gptoss", R, li,
                    f"{a['dnorm']/a['ynorm']:.5f}",
                    f"{a['cos']/ntok:.5f}",
                    f"{usage_tv[i]:.5f}",
                    f"{a['dnorm']/a['xnorm']:.5f}",
                    ntok, ""])
tot_d = sum(a["dnorm"] for a in ACC.values())
tot_y = sum(a["ynorm"] for a in ACC.values())
tot_c = sum(a["cos"] for a in ACC.values())
print(f"[fd-vllm] DONE oss120: mean rel_out {tot_d/tot_y:.4f}, mean cos "
      f"{tot_c/(ntok*L):.4f}, mean usage_tv {usage_tv.mean():.4f} over {ntok} tokens",
      flush=True)
