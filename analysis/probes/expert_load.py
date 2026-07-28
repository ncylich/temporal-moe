#!/usr/bin/env python3
"""Measure per-expert load (acceptance criterion 4) for a trained temporal-MoE checkpoint.

Monkeypatches TopKRouter.forward to accumulate per-expert token counts, then runs Megatron's
normal eval (driven by --skip-train + --load) so the routers fire on validation batches.
Writes per-layer max/mean load ratio to $EXPERT_LOAD_OUT.

Invoked by run.sh with EVAL_ONLY=1 (same model args as the training run). Run from Megatron-LM/.
"""
import os, sys, json
sys.path.insert(0, os.getcwd())   # run.sh cd's to Megatron-LM; put it on the path for `import megatron`
import torch
from megatron.core.transformer.moe.router import TopKRouter

_counts = {}   # layer_number -> tensor[num_experts]
_orig = TopKRouter.forward

def _patched(self, inp):
    probs, routing_map = _orig(self, inp)
    rm = routing_map.detach()
    # routing_map is [num_tokens, num_experts] boolean; sum over tokens -> per-expert counts
    ne = getattr(self.config, "num_moe_experts", None) or rm.shape[-1]
    c = rm.to(torch.float32).sum(dim=0) if rm.shape[-1] == ne else rm.to(torch.float32).sum(dim=1)
    ln = getattr(self, "layer_number", -1)
    if ln not in _counts:
        _counts[ln] = c.clone()
    else:
        _counts[ln] += c
    return probs, routing_map

TopKRouter.forward = _patched

def _dump():
    out = os.environ.get("EXPERT_LOAD_OUT", "/tmp/expert_load.json")
    res = {}
    for ln, c in _counts.items():
        cl = c.cpu().tolist()
        mean = sum(cl) / len(cl) if cl else 0.0
        res[str(ln)] = {
            "num_experts": len(cl),
            "max_over_mean": (max(cl) / mean if mean > 0 else 0.0),
            "min_over_mean": (min(cl) / mean if mean > 0 else 0.0),
            "max_tokens": max(cl) if cl else 0,
            "total_tokens": sum(cl),
        }
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    worst = max((v["max_over_mean"] for v in res.values()), default=0.0)
    print(f"EXPERT_LOAD worst max/mean across layers = {worst:.2f}x  (criterion 4: <= 8x)")
    print("EXPERT_LOAD per-layer max/mean: " +
          json.dumps({k: round(v["max_over_mean"], 2) for k, v in sorted(res.items())}))

if __name__ == "__main__":
    from megatron.training import pretrain
    from megatron.core.enums import ModelType
    import pretrain_gpt
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    try:
        pretrain(
            pretrain_gpt.train_valid_test_datasets_provider,
            pretrain_gpt.model_provider,
            ModelType.encoder_or_decoder,
            pretrain_gpt.forward_step,
            args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        )
    finally:
        _dump()
