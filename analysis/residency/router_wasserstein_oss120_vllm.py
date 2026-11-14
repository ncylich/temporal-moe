#!/usr/bin/env python3
"""oss-120b router W1 via vLLM prefill capture (MXFP4-native; HF dequant path is
infeasible under the 251GB cgroup). Hooks FusedMoE.forward in-process, prefders
WildChat prompts one at a time (max_tokens=1), computes imposed-variant W1 with
the same offline scan as router_wasserstein.py, appends to the same CSV."""
import csv
import json
import os
import sys

import torch

sys.path.insert(0, "/workspace/temporal-moe/analysis/residency")
sys.path.insert(0, "/workspace/temporal-moe/analysis")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import granularity_ladder as GL                                      # noqa: E402
from paths import ABLATIONS                                          # noqa: E402

MODEL = "/dev/shm/gpt-oss-120b"
PROMPTS = "/workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl"
R, N, MAXTOK = 4, 100, 512

CAP = []
import vllm_glue                                                     # noqa: E402
vllm_glue.install()
import vllm_residency as VR                                          # noqa: E402
_orig_apply = VR.apply
def _cap_apply(key, router_logits, *a, **k):
    CAP.append(router_logits.detach().float())
    return _orig_apply(key, router_logits, *a, **k)
VR.apply = _cap_apply

from vllm import LLM, SamplingParams                                 # noqa: E402
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.92,
          max_model_len=1024, enable_prefix_caching=False)
sp = SamplingParams(max_tokens=1, temperature=0)

prompts = [json.loads(l)["text"] for l in open(PROMPTS)][:N]
sums = None
ntok = 0
L = None
for pi, text in enumerate(prompts):
    CAP.clear()
    llm.chat([[{"role": "user", "content": text}]], sp, use_tqdm=False)
    if L is None:
        L = len(CAP)
        sums = torch.zeros(L)
        print(f"[rw-vllm] {L} MoE layers captured per pass", flush=True)
    assert len(CAP) % L == 0, (len(CAP), L)
    # last L entries = the single decode step; the first block(s) are prefill.
    # prefill block: first L captures cover the full prompt.
    for li in range(L):
        lg = CAP[li][: MAXTOK].float()                # [T,E] prefill logits
        pf = torch.softmax(lg, -1)
        mask = GL.compute_resident_mask_accel(
            lg.unsqueeze(1), R, evict="min_logit").squeeze(1)
        q = pf * mask
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
        sums[li] += 0.5 * (pf - q).abs().sum(-1).sum().cpu()
    ntok += min(CAP[0].shape[0], MAXTOK)
    if pi % 25 == 0:
        print(f"[rw-vllm] {pi}/{N}", flush=True)

out = os.path.join(ABLATIONS, "router_wasserstein.csv")
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    for li in range(L):
        w.writerow(["gpt-oss-120b", "gptoss", R, li,
                    f"{sums[li].item()/ntok:.5f}", "", ntok])
print(f"[rw-vllm] DONE oss120: mean imposed {sums.sum().item()/ntok/L:.4f} "
      f"over {ntok} tokens", flush=True)
