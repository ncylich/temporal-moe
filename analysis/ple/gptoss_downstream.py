#!/usr/bin/env python3
"""gpt-oss downstream deltas: free routing vs untrained R=k residency, accuracy only.

gpt-oss has no base checkpoint, so it never touches the BPB law (instruct models are
out-of-domain on the audited slice). Its role is the domain-neutral table: mean accuracy
over the ten 0-shot tasks, free vs constrained on the SAME model — the delta is the
constraint's downstream cost. Router convention: top-k on logits then softmax over the
selected values, so gate mass renormalizes over the selection by construction.

    gptoss_downstream.py --model /dev/shm/gpt-oss-20b --tag gptoss20b --batch-size 16
"""
import argparse
import csv
import os
import sys
import types

for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal.temporal_router import compute_resident_mask_accel     # noqa: E402

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "sciq", "winogrande",
         "openbookqa", "boolq", "lambada_openai", "copa"]
CFG = {"on": False, "R": 4}


def patch(model):
    """The MXFP4 quantizer installs per-instance mlp.forward overrides that compute routing
    straight from router.weight through the triton routing kernel, bypassing the router
    module entirely (verified: class-level router patches change nothing). So the port
    rebinds each instance's forward to the same mxfp4 flow with the resident mask applied
    to the router logits before the kernel's top-k."""
    import types as T
    import torch.nn as nn
    from transformers.integrations.mxfp4 import triton_kernels_hub, on_device

    def fwd(self, hidden_states):
        routing = triton_kernels_hub.routing.routing
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
        router_logits = nn.functional.linear(hidden_states, self.router.weight,
                                             self.router.bias)
        if CFG["on"]:
            # bs=1 REQUIRED for constrained arms: lm_eval LEFT-pads batches, and pad
            # tokens ahead of content corrupt the scan cold-fill (measured: arc_easy 0.67
            # at bs=1 vs 0.63 at bs=16). The [S,B,E] reshape is kept for correctness.
            lg = router_logits.view(batch_size, -1, router_logits.shape[-1])
            lg = lg.transpose(0, 1).contiguous().float()
            with torch.no_grad():
                mask = compute_resident_mask_accel(lg, CFG["R"], evict="min_logit", swaps=1)
            router_logits = router_logits.masked_fill(
                ~mask.transpose(0, 1).reshape(router_logits.shape), float("-inf"))
        with on_device(router_logits.device):
            routing_data, gather_idx, scatter_idx = routing(router_logits,
                                                            self.router.top_k)
        routed_out = self.experts(hidden_states, routing_data, gather_idx,
                                  scatter_idx=scatter_idx)
        routed_out = routed_out.reshape(batch_size, -1, self.router.hidden_dim)
        return routed_out, router_logits

    n = 0
    for layer in model.model.layers:
        layer.mlp.forward = T.MethodType(fwd, layer.mlp)
        n += 1
    print(f"[gptoss] rebound {n} mlp forwards (mxfp4 masked routing)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--r", type=int, default=None, help="residency R (default: k)")
    ap.add_argument("--skip-free", action="store_true")
    A = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype="auto", device_map="cuda")
    tok = AutoTokenizer.from_pretrained(A.model)
    model.eval()
    patch(model)
    E = model.config.num_local_experts
    k = model.config.num_experts_per_tok
    print(f"[gptoss] {A.tag}: E={E} k={k} layers={model.config.num_hidden_layers}", flush=True)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    out_rows = []
    R = A.r if A.r is not None else k
    arms = [("free", False, A.batch_size), (f"R{R}", True, 1)]
    if A.skip_free:
        arms = arms[1:]
    for arm, on, bs in arms:
        CFG.update(on=on, R=R)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
        res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                              limit=(A.limit or None), bootstrap_iters=1000)["results"]
        accs = {}
        for t in TASKS:
            r = res.get(t, {})
            v = r.get("acc,none", r.get("acc_norm,none"))
            se = r.get("acc_stderr,none", r.get("acc_norm_stderr,none"))
            accs[t] = (v, se)
        mean = sum(v for v, _ in accs.values()) / len(accs)
        print(f"[gptoss] {A.tag} arm={arm} mean_acc={mean:.4f}", flush=True)
        out_rows.append((arm, accs, mean))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "results", "ablations", "gptoss_downstream_deltas.csv")
    exists = os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            f.write('"# gpt-oss downstream deltas: ten 0-shot tasks (limit 500), free routing vs '
                    'untrained R=k min_logit residency on the same model. Instruct-only models: '
                    'accuracy deltas are the domain-neutral measure; no BPB cells by design. '
                    'Producer: analysis/ple/gptoss_downstream.py"\n')
            w.writerow(["model", "arm", "E", "k"] + TASKS + ["mean_acc"])
        for arm, accs, mean in out_rows:
            w.writerow([A.tag, arm, E, k] + [f"{accs[t][0]:.4f}" for t in TASKS]
                       + [f"{mean:.4f}"])
    print(f"[gptoss] wrote {out}", flush=True)
    print(f"GPTOSS {A.tag} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
