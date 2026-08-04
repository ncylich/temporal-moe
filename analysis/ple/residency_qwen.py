#!/usr/bin/env python3
"""Rolling-residency for Qwen3.5-MoE, sharing every audited part of the OLMoE path.

Why a separate module rather than a branch inside `residency.py`: that file is imported by runs
currently in flight, and the two model families differ in enough small ways (see below) that an
in-place generalisation would have meant editing the router hook while a training job depends on it.
What is NOT duplicated is anything that decides a number: the resident-set scan
(`compute_resident_mask` / `_accel`), the unified auxiliary loss, the effective-expert counter and
the shared `_CFG` all come from `residency` by import. This module contributes only the parts that
are genuinely Qwen-specific -- which classes to patch, and how that family's router normalises.

Differences from OLMoE that this file exists to absorb:

    shared expert    Qwen3.5 runs an always-on shared expert alongside the 256 routed ones, applied
                     in the block *outside* the router. Masking the router therefore leaves it
                     permanently resident, which is the architecturally correct reading: a shared
                     expert is not a swap candidate. It does mean "R resident" understates true
                     resident memory by one expert, and `resident_fraction()` reports both.
    normalisation    OlmoeTopKRouter honours a `norm_topk_prob` flag; Qwen3_5MoeTopKRouter always
                     renormalises the top-k probabilities. Referencing the flag here would raise.
    hybrid attention 3 Gated-DeltaNet layers per 1 full-attention layer. Irrelevant to residency,
                     which touches only the MoE path, but it is why attention LoRA does not port.

The router's own arithmetic is reproduced verbatim from the stock forward, with exactly one change:
logits are masked to the resident set before the softmax. Verified by `checks_qwen.py parity`.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
)

_CFG = RES._CFG                       # one config object, so free_set/R/evict cannot drift apart
_orig_block_forward = Qwen3_5MoeSparseMoeBlock.forward
_orig_router_forward = Qwen3_5MoeTopKRouter.forward

# Router logits captured per forward. Qwen's block does return them through the model's
# output_router_logits path, but capturing here as well means the aux and effective-expert
# measurements see the SAME tensor the mask was applied to, with no reliance on that plumbing.
CAPTURE = {"on": False, "logits": []}


def _block_forward(self, hidden_states):
    """Record the pack shape; the router sees flattened [B*S, E] and cannot recover S alone."""
    b, s, _ = hidden_states.shape
    self.gate._resid_shape = (b, s)
    return _orig_block_forward(self, hidden_states)


def _router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)              # [N, E]
    if CAPTURE["on"]:
        CAPTURE["logits"].append(router_logits)

    _li = getattr(self, "_layer_idx", None)
    freed = False
    if not _CFG["on"]:
        freed = True
    elif _li is not None:
        _fs = _CFG.get("free_set")
        if _fs is not None:
            freed = _li in _fs
        else:
            freed = _li < _CFG.get("free_layers", 0)

    used = router_logits
    if not freed:
        N, E = router_logits.shape
        b, s = getattr(self, "_resid_shape", (1, N))
        lg = router_logits.view(b, s, E).transpose(0, 1).contiguous()  # [S, B, E]
        with torch.no_grad():
            scan = (RES.compute_resident_mask_accel
                    if (lg.is_cuda and _CFG.get("accel", True)) else RES.compute_resident_mask)
            mask = scan(lg.float(), _CFG["R"], evict=_CFG["evict"])    # [S, B, E] bool, R per token
        if _CFG.get("collect_telem"):
            RES._accum_telem(mask)
        used = router_logits.masked_fill(~mask.transpose(0, 1).reshape(N, E), float("-inf"))

    # Stock Qwen3_5MoeTopKRouter arithmetic below, unchanged. Qwen always renormalises the top-k
    # probabilities -- there is no norm_topk_prob flag on this family, unlike OLMoE.
    router_probs = torch.softmax(used, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


def install():
    Qwen3_5MoeSparseMoeBlock.forward = _block_forward
    Qwen3_5MoeTopKRouter.forward = _router_forward


def tag_layers(model):
    """Index the routers in depth order so free_set/{layer} selection means what it says."""
    n = 0
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeTopKRouter):
            m._layer_idx = n
            n += 1
    return n


def capture(on=True):
    CAPTURE["on"] = on
    CAPTURE["logits"] = []


def captured():
    return tuple(CAPTURE["logits"])


def resident_fraction(cfg, R):
    """Routed-only and true fractions. Reporting one number here would be misleading: the shared
    expert is always resident, so a model with a shared expert holds R+1 of E+1 experts, not R of E.
    """
    E = cfg.num_experts
    return {"routed": R / E, "with_shared": (R + 1) / (E + 1), "E": E, "R": R}


def load_model(path="/workspace/qwen35-adapt/model", device="cuda", dtype=torch.bfloat16,
               device_map=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    kw = {"dtype": dtype, "trust_remote_code": False}
    if device_map is not None:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    if device_map is None:
        model = model.to(device)

    # Drop the multi-token-prediction head (1.69 GB) and the vision tower (0.89 GB). Neither
    # participates in a text BPB measurement or in expert adaptation, and on an 80 GB card against a
    # 69.3 GB text model, 2.6 GB is the difference between fitting and not. Done after load rather
    # than by filtering the state dict so the checkpoint is read exactly as published.
    freed = []
    for owner, attr in ((model, "mtp"), (getattr(model, "model", model), "visual"),
                        (getattr(model, "model", model), "vision_tower")):
        if hasattr(owner, attr) and getattr(owner, attr) is not None:
            n = sum(p.numel() for p in getattr(owner, attr).parameters())
            setattr(owner, attr, None)
            freed.append(f"{attr} ({n/1e9:.2f}B params)")
    if freed:
        torch.cuda.empty_cache()
        print(f"  [qwen] dropped {', '.join(freed)}", flush=True)

    model.eval()
    install()
    n = tag_layers(model)
    print(f"  [qwen] {n} MoE routers tagged, E={model.config.num_experts} "
          f"k={model.config.num_experts_per_tok} layers={model.config.num_hidden_layers}", flush=True)
    return model, tok
