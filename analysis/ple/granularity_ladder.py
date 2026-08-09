#!/usr/bin/env python3
"""Granularity-law ladder cells for the non-resident-program models (training-free).

Per model: free baseline + the iso-fraction ladder from 12.5% resident down to the model's
fundamental floor k/E (R = f*E, R >= k). Damage = BPB(R) - BPB(free) on the byte-identical
audited slice retokenized per tokenizer (build_qwen_slice.py). min_logit rolling residency,
<=1 swap/token, selection masked / gate weights taken per each model's own convention.

    granularity_ladder.py --model lfm25
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import ABLATIONS                                          # noqa: E402
from temporal.temporal_router import compute_resident_mask_accel     # noqa: E402

DATA = "/workspace/olmoe-adapt/data"
MODELS = {
    "lfm25": {"path": "/dev/shm/lfm25-8b-base", "slice": "lfm25",
              "E": 32, "k": 4, "arch": "lfm"},
    "gemma4": {"path": "/gemma4-26b", "slice": "gemma4",
               "E": 128, "k": 8, "arch": "gemma4"},
}
CFG = {"on": False, "R": 8}


def fractions(E, k):
    """12.5% down to the floor k/E, halving-ish on the R lattice used by the qwen grids."""
    ladder = [int(round(f * E)) for f in (0.125, 0.09375, 0.0625, 0.046875, 0.03125,
                                          0.0234375, 0.01953125)]
    return sorted({max(R, k) for R in ladder if R >= k} | {k}, reverse=True)


def patch_lfm():
    from transformers.models.lfm2_moe import modeling_lfm2_moe as m
    orig_fwd = m.Lfm2MoeSparseMoeBlock.forward
    orig_route = m.Lfm2MoeSparseMoeBlock.route_tokens_to_experts

    def fwd(self, hidden_states):
        self._resid_shape = hidden_states.shape[:2]
        return orig_fwd(self, hidden_states)

    def route(self, router_logits):
        if not CFG["on"]:
            return orig_route(self, router_logits)
        # Selection signal is the model's own: sigmoid + expert bias. The scan constrains WHICH
        # experts are selectable; gate weights still come from the raw sigmoid at the selected
        # experts (the model's convention, norm_topk_prob=True renormalizes afterwards).
        scores = router_logits.sigmoid()
        sel_signal = scores + self.expert_bias if self.use_expert_bias else scores
        b, s = self._resid_shape
        lg = sel_signal.view(b, s, -1).transpose(0, 1).contiguous()
        with torch.no_grad():
            mask = compute_resident_mask_accel(lg.float(), CFG["R"], evict="min_logit", swaps=1)
        masked = sel_signal.masked_fill(~mask.transpose(0, 1).reshape(sel_signal.shape),
                                        float("-inf"))
        _, selected = torch.topk(masked, k=self.top_k, dim=-1)
        weights = torch.gather(scores, dim=1, index=selected).type_as(router_logits)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights = weights * self.routed_scaling_factor
        return selected, weights

    m.Lfm2MoeSparseMoeBlock.forward = fwd
    m.Lfm2MoeSparseMoeBlock.route_tokens_to_experts = route


def patch_gemma4():
    from transformers.models.gemma4 import modeling_gemma4 as m
    import torch.nn as nn

    def fwd(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)                     # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)
        if CFG["on"]:
            # eval runs batch 1, so [B*S, E] is [S, E]; scan wants [S, B, E]
            lg = expert_scores.unsqueeze(1).float()
            with torch.no_grad():
                mask = compute_resident_mask_accel(lg, CFG["R"], evict="min_logit", swaps=1)
            probs_for_topk = router_probabilities.masked_fill(
                ~mask.squeeze(1), 0.0)
        else:
            probs_for_topk = router_probabilities
        top_k_weights, top_k_index = torch.topk(probs_for_topk, k=self.config.top_k_experts,
                                                dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index

    m.Gemma4TextRouter.forward = fwd


def bpb(model, ids, divisor):
    tot = ntok = 0
    with torch.no_grad():
        for i in range(ids.shape[0]):
            b = ids[i:i + 1].to("cuda").long()
            lg = model(b).logits[:, :-1]
            tg = b[:, 1:]
            for i0 in range(0, lg.shape[1], 512):
                sl = lg[:, i0:i0 + 512].float()
                tot += float(torch.nn.functional.cross_entropy(
                    sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1),
                    reduction="sum"))
                del sl
            ntok += tg.numel()
            del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n-seq", type=int, default=16)
    A = ap.parse_args()
    M = MODELS[A.model]

    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(M["path"], **kw).to("cuda")
    model.eval()
    {"lfm": patch_lfm, "gemma4": patch_gemma4}[M["arch"]]()

    D = json.load(open(f"{DATA}/bpb_slice_meta_{M['slice']}.json"))["divisor_D"]
    ids = torch.load(f"{DATA}/bpb_slice_ids_{M['slice']}.pt", weights_only=False)[: A.n_seq]

    out = os.path.join(ABLATIONS, "granularity_ladder.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Granularity-law ladder, training-free, min_logit <=1 swap/token, base '
                 'checkpoints, byte-identical audited slice per tokenizer. damage = bpb - free '
                 'bpb of the same model. Producer: analysis/ple/granularity_ladder.py"\n')
        w.writerow(["model", "E", "k", "cell", "R", "frac_pct", "bpb", "damage", "n_seq", "secs"])

    def cell(name, R=None):
        CFG.update(on=R is not None, R=R or 0)
        t0 = time.time()
        v = bpb(model, ids, D)
        frac = "" if R is None else f"{100*R/M['E']:.2f}"
        w.writerow([A.model, M["E"], M["k"], name, R or "", frac, f"{v:.6f}",
                    "" if R is None else f"{v-free:+.6f}", A.n_seq, f"{time.time()-t0:.1f}"])
        fh.flush()
        print(f"  [{A.model}] {name:12} BPB={v:.6f} ({time.time()-t0:.0f}s)", flush=True)
        return v

    free = None
    free = cell("free")
    assert free < 1.5, f"free BPB implausible ({free}) - slice or model wiring is wrong"
    for R in fractions(M["E"], M["k"]):
        v = cell(f"R{R}", R=R)
        assert v > free - 0.01, f"constrained beat free ({v} vs {free}) - masking not engaged"
    fh.close()
    print(f"LADDER {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
