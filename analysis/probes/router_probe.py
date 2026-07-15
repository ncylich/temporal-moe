#!/usr/bin/env python3
"""Router probe — log per-MoE-layer, per-token routing on one fixed forward pass of a trained model.

Captures the FIRST micro-batch's router state for every MoE layer:
  - `logits`  [seq, batch, E] raw gating logits (pre-mask) -> rank experts at any K post-hoc
  - `mask`    [seq, batch, E] resident set actually used (temporal models only; None for plain MoE)
Same fixed batch across models (seed 1234, same data/split) so token positions align.

Feeds the cheap mechanistic graphs, all from these logs:
  A per-token expert raster (MoE top-k / temporal resident / temporal unconstrained-preference)
  B rolling-policy hit-rate vs K      C expert lifetime vs K

Invoked by run.sh (PROBE=1) from inside Megatron-LM/. Mirrors expert_load.py.
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())                                  # -> import megatron
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -> import temporal_router
from megatron.core.transformer.moe.router import TopKRouter

_TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"
_EVICT = os.environ.get("TEMPORAL_EVICT", "min_logit")
_first = {}                                                      # layer_number -> {logits, mask, k}


def _probe_forward(self, input):
    """Drop-in TopKRouter.forward that records the first forward per layer, then routes normally."""
    input = self.apply_input_jitter(input)
    logits = self.gating(input)                                  # [seq, batch, E]
    k = int(self.config.moe_router_topk)
    ln = int(getattr(self, "layer_number", -1))
    mask = None
    if _TEMPORAL:
        from temporal_router import compute_resident_mask_accel   # same mask as training
        with torch.no_grad():
            mask = compute_resident_mask_accel(logits, k, evict=_EVICT)
        used = logits.masked_fill(~mask, float("-inf"))
    else:
        used = logits
    if ln not in _first:
        _first[ln] = {"logits": logits.detach().to(torch.float16).cpu(),
                      "mask": (mask.detach().cpu() if mask is not None else None), "k": k}
    return self.routing(used)


TopKRouter.forward = _probe_forward


def _dump():
    out = os.environ.get("ROUTER_LOG_OUT", "/tmp/router_log.pt")
    torch.save({"temporal": _TEMPORAL, "evict": _EVICT,
                "layers": {ln: _first[ln] for ln in sorted(_first)}}, out)
    ex = _first[sorted(_first)[0]]
    print(f"[probe] saved {out}: {len(_first)} MoE layers, logits {tuple(ex['logits'].shape)}, "
          f"k={ex['k']}, temporal={_TEMPORAL}")


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
        if _first:
            _dump()
