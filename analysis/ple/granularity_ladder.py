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
    "mixtral": {"path": "/dev/shm/mixtral-8x7b", "slice": "mixtral",
                "E": 8, "k": 2, "arch": "mixtral", "quant8": True},
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


def patch_mixtral():
    from transformers.models.mixtral import modeling_mixtral as m
    orig = m.MixtralSparseMoeBlock.forward
    import torch.nn.functional as F

    def fwd(self, hidden_states):
        if not CFG["on"]:
            return orig(self, hidden_states)
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        router_logits = self.gate(x)
        lg = router_logits.view(b, s, -1).transpose(0, 1).contiguous()
        with torch.no_grad():
            mask = compute_resident_mask_accel(lg.float(), CFG["R"], evict="min_logit", swaps=1)
        masked = router_logits.masked_fill(~mask.transpose(0, 1).reshape(router_logits.shape),
                                           float("-inf"))
        # Mixtral: softmax over experts THEN top-k, weights renormalized over the selected pair
        # (norm_topk equivalent) -- masked softmax + renorm is exactly its convention.
        probs = F.softmax(masked, dim=-1, dtype=torch.float)
        weights, selected = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(hidden_states.dtype)
        final = torch.zeros_like(x)
        one = torch.nn.functional.one_hot(selected, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            idx, top_x = torch.where(one[e])
            if top_x.numel() == 0:
                continue
            cur = self.experts[e](x[top_x]) * weights[top_x, idx, None]
            final.index_add_(0, top_x, cur.to(x.dtype))
        return final.view(b, s, h), router_logits

    m.MixtralSparseMoeBlock.forward = fwd


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
    if M.get("quant8"):
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    model = AutoModelForCausalLM.from_pretrained(M["path"], **kw)
    if not M.get("quant8"):
        model = model.to("cuda")
    model.eval()
    {"lfm": patch_lfm, "mixtral": patch_mixtral}[M["arch"]]()

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
