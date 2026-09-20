#!/usr/bin/env python3
"""Rolling-residency for the Qwen MoE families, sharing every audited part of the OLMoE path.

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
    normalisation    Qwen3_5MoeTopKRouter always renormalises the top-k probabilities; OLMoE and
                     Qwen3-MoE honour a `norm_topk_prob` flag. Defaulting the missing attribute to
                     True reproduces each family's stock arithmetic exactly.
    families         `install(family)` selects which block/router pair to patch: "qwen3_5" (256
                     experts, shared expert, DeltaNet hybrid) or "qwen3" (128 experts, NO shared
                     expert, standard attention). The second exists to separate expert redundancy
                     from the shared expert as explanations for Qwen's cheap residency.
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
    Qwen3_5MoeSparseMoeBlock, Qwen3_5MoeTopKRouter,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (     # noqa: E402
    Qwen3MoeSparseMoeBlock, Qwen3MoeTopKRouter,
)

# Two families, one hook. They differ in exactly two ways that matter here: Qwen3.5 runs an
# always-on shared expert inside the block (outside the router, so masking leaves it resident),
# and Qwen3-MoE honours a norm_topk_prob flag where Qwen3.5 always renormalises. Everything else --
# router signature, attribute names, the 3-tuple return -- is identical, so a second copy of this
# file would have been a copy of the parts that decide numbers.
FAMILIES = {
    "qwen3_5": (Qwen3_5MoeSparseMoeBlock, Qwen3_5MoeTopKRouter),
    "qwen3":   (Qwen3MoeSparseMoeBlock, Qwen3MoeTopKRouter),
}
ACTIVE = {"name": "qwen3_5"}

_CFG = RES._CFG                       # one config object, so free_set/R/evict cannot drift apart
_ORIG = {k: (b.forward, r.forward) for k, (b, r) in FAMILIES.items()}

# Router logits captured per forward. Qwen's block does return them through the model's
# output_router_logits path, but capturing here as well means the aux and effective-expert
# measurements see the SAME tensor the mask was applied to, with no reliance on that plumbing.
CAPTURE = {"on": False, "logits": []}


def _block_forward(self, hidden_states):
    """Record the pack shape; the router sees flattened [B*S, E] and cannot recover S alone."""
    b, s, _ = hidden_states.shape
    self.gate._resid_shape = (b, s)
    return _ORIG[ACTIVE["name"]][0](self, hidden_states)


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
        R = _CFG["R"]
        _rmap = _CFG.get("R_map")
        if _rmap is not None and _li is not None:
            R = _rmap.get(_li, R)                # per-layer residency budget (frontier allocation)
        b, s = getattr(self, "_resid_shape", (1, N))
        lg = router_logits.view(b, s, E).transpose(0, 1).contiguous()  # [S, B, E]
        if _CFG.get("decode_mode"):    # generation: stateful rule across forwards, prefill free
            import decode_state as _DS
            mask = _DS.route(_li, lg)
            if mask is None:                                           # prefill observe = free
                mask = torch.ones_like(lg, dtype=torch.bool)
        else:
            with torch.no_grad():
                scan = (RES.compute_resident_mask_accel
                        if (lg.is_cuda and _CFG.get("accel", True)) else RES.compute_resident_mask)
                mask = scan(lg.float(), R, evict=_CFG["evict"],
                            swaps=_CFG.get("swaps", 1))                # [S, B, E] bool, R per token
        _ef = _CFG.get("enforce_from", 0)
        if _ef:        # instruct protocol: prefill positions free, rule enforced from response
            if _CFG.get("cold_start"):
                with torch.no_grad():
                    cm = RES.compute_resident_mask_accel(lg[_ef:].float(), R,
                                                         evict=_CFG["evict"],
                                                         swaps=_CFG.get("swaps", 1))
                mask = torch.ones_like(mask)
                mask[_ef:] = cm
            else:
                mask[:_ef] = True
        if _CFG.get("collect_telem"):
            RES._accum_telem(mask)
        used = router_logits.masked_fill(~mask.transpose(0, 1).reshape(N, E), float("-inf"))

    # Stock router arithmetic below, unchanged for whichever family is installed.
    router_probs = torch.softmax(used, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    # Qwen3.5 always renormalises; Qwen3-MoE only when norm_topk_prob is set. Defaulting the
    # missing attribute to True reproduces each family's stock arithmetic exactly.
    if getattr(self, "norm_topk_prob", True):
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


def _experts_forward_fast(self, hidden_states, top_k_index, top_k_weights):
    """MEASURED SLOWER THAN STOCK -- kept only as a documented negative result, default OFF.

    The reasoning was that `for expert_idx in expert_hit:` over a CUDA tensor costs a device->host
    copy per expert per layer, ~6k stalls per forward, and that hoisting the list to host once would
    remove them. Benchmarked against stock on Qwen3-30B it is 0.35x at batch 1-4 and 0.51x at batch
    16 -- i.e. three times SLOWER. The `.tolist()` is a full pipeline barrier: nothing can be queued
    until one_hot/sum/nonzero have all completed, so the CPU cannot run ahead. Stock's per-iteration
    syncs happen after kernels are already in flight, which costs far less than serialising the
    launch stream.

    Left in the tree because the negative result is worth more than the deletion: the obvious fix to
    an obvious-looking bottleneck is a 3x regression, and anyone reading the loop will have the same
    idea.

    The shipped forward does `for expert_idx in expert_hit:` where `expert_hit` is a CUDA tensor, so
    every iteration copies a scalar device->host, and the `expert_idx == self.num_experts` guard
    copies another. With 128-256 experts over 40-48 layers that is roughly 6-12k synchronisations per
    forward at ~50-100us each -- the GPU stalls on Python for most of the pass, which is why
    utilisation sat at 64% with a batch that should have saturated it.

    Moving the hit list to host ONCE and iterating Python ints removes every one of those syncs. The
    per-expert computation below is byte-for-byte the shipped code; only the loop bookkeeping changes,
    so outputs are identical rather than approximately equal -- checked by the suite's preflight,
    which compares against the stock forward bitwise.
    """
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
    for e in hit:                                   # Python ints: no device->host per iteration
        if e == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[e]).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        h = F.linear(h, self.down_proj[e])
        h = h * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
    return final_hidden_states


def assert_valid_R(R, k):
    """R < k is not a weaker constraint, it is a different (broken) model.

    The router selects top-k from the resident set. With R < k only R experts carry non-zero
    probability, but top-k still returns k indices, so the extra slots are dispatched with weight
    zero: compute is paid for k experts while R contribute. That is top-R routing with waste, not
    residency at a smaller budget, and reporting it as "R/E resident" implies a serving trade-off
    that is not what was measured. R = k is the tightest meaningful setting -- the one the OLMoE
    runs used.
    """
    if R < k:
        raise ValueError(
            f"R={R} is below top-k={k}. Only {R} experts could carry weight while the router still "
            f"dispatches {k}; this measures degraded top-{R} routing, not residency at R={R}. "
            f"R={k} is the tightest valid constraint.")


def install(family="qwen3_5", fast_experts=False):
    ACTIVE["name"] = family
    blk, rtr = FAMILIES[family]
    blk.forward = _block_forward
    rtr.forward = _router_forward
    if fast_experts:
        for mod, cls in (("qwen3_5_moe", "Qwen3_5MoeExperts"), ("qwen3_moe", "Qwen3MoeExperts")):
            m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[cls])
            getattr(m, cls).forward = _experts_forward_fast


def tag_layers(model):
    """Index the routers in depth order so free_set/{layer} selection means what it says."""
    n = 0
    rtr = FAMILIES[ACTIVE["name"]][1]
    for m in model.modules():
        if isinstance(m, rtr):
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
               device_map=None, family="qwen3_5"):
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
    install(family)
    n = tag_layers(model)
    print(f"  [qwen] {n} MoE routers tagged, E={model.config.num_experts} "
          f"k={model.config.num_experts_per_tok} layers={model.config.num_hidden_layers}", flush=True)
    return model, tok
