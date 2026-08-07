#!/usr/bin/env python3
"""Rolling residency for Unsloth's grouped_mm path (transformers 5.x + unsloth_zoo patches).

Unsloth leaves routing in plain PyTorch and fuses only the expert computation
(`unsloth_zoo.temporary_patches.qwen3_moe` installs a `torch._grouped_mm` experts forward),
which is the same split `residency_fused.py` exploited: the intervention goes where the
routing is, and the fused GEMMs never learn residency exists.

The complication 5.x adds: the gate is a `TopKRouter` that does softmax and top-k *inside*
and returns `(router_logits, routing_weights, selected_experts)`. Its top-k choice is made
on unmasked logits, so this forward takes only `router_logits` from the gate (keeping any
hooks on the gate module live), applies the residency mask with `[B, S]` still in scope,
and recomputes softmax/top-k/renorm exactly as the router's own body does. The wasted
unmasked top-k inside the gate is a [M, E] op — noise next to a 30B forward.

One file covers both Qwen families because unsloth_zoo builds both block forwards from the
same factory and the 5.x modeling code is structurally identical:
    qwen3     gate has norm_topk_prob (True on Qwen3-30B-A3B); no shared expert
    qwen3_5   gate always renormalises (no flag); block has shared_expert(+gate)
`getattr(gate, "norm_topk_prob", True)` reproduces both conventions.

Gate-mass note: masking before the softmax on a `norm_topk_prob=False` model rescales the
top-k gate mass (~0.40 -> 1.0) — the artifact that inflated this programme's OLMoE results
~10x. Both supported families renormalise, and `install()` refuses a False model rather
than silently reintroducing the bug.

Everything that decides a number is imported from `residency`: the resident-set scan, _CFG,
telemetry. This file only re-expresses where the mask is applied.

    import unsloth                    # must come first, applies zoo patches
    model, tok = FastModel.from_pretrained(...)
    import residency_unsloth as RU
    RU.install(model)                 # after load; class-level, wins over the zoo forward
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402

_CFG = RES._CFG
_ORIG = {}

# Per-micro-step (aux, z) values logged by the in-forward injection, detached floats.
AUX_LOG = []


class _AuxInject(torch.autograd.Function):
    """Megatron-style aux-loss injection: identity on the block output in forward; in
    backward, feeds d(total)/d(aux) = scale into the aux subgraph. This is how FLAME's own
    pretraining wires the balancing loss, and it is the only construction that survives
    every gradient-checkpointing mode: post-forward aux from hook-captured logits dies
    under any checkpoint whose outer forward runs under no-grad (unsloth's offloaded mode,
    HF reentrant) -- measured, not hypothetical: the first train_unsloth smoke test caught
    exactly that detachment."""

    @staticmethod
    def forward(ctx, out, aux, scale):
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, g):
        global _INJ_FIRED
        _INJ_FIRED = True
        return g, torch.tensor(ctx.scale, device=g.device, dtype=torch.float32), None


# Set by _AuxInject.backward; the trainer's step-0 guard checks it, because a zero-looking
# gate gradient cannot distinguish dead injection (the LM loss also reaches the gate,
# through the softmax routing weights).
_INJ_FIRED = False


@torch.compiler.disable
def _forward(self, hidden_states):
    """Sparse-MoE block forward with residency masking; body follows unsloth_zoo's own."""
    if hidden_states.dim() == 3:
        B, S, H = hidden_states.shape
    else:                                       # zoo supports pre-flattened input; keep that
        (M0, H), B, S = hidden_states.shape, 1, hidden_states.shape[0]
    hs = hidden_states.view(-1, H)
    M = hs.shape[0]

    shared_out = None
    if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
        shared_out = self.shared_expert(hs)

    router_logits = self.gate(hs)[0]                               # [M, E]; gate hooks fire

    li = getattr(self, "_layer_idx", None)
    freed = not _CFG["on"]
    if not freed and li is not None:
        fs = _CFG.get("free_set")
        freed = (li in fs) if fs is not None else (li < _CFG.get("free_layers", 0))

    used = router_logits
    if not freed:
        E = router_logits.shape[1]
        lg = router_logits.view(B, S, E).transpose(0, 1).contiguous()   # [S, B, E]
        with torch.no_grad():
            scan = (RES.compute_resident_mask_accel
                    if (lg.is_cuda and _CFG.get("accel", True)) else RES.compute_resident_mask)
            mask = scan(lg.float(), _CFG["R"], evict=_CFG["evict"],
                        swaps=_CFG.get("swaps", 1))
        if _CFG.get("collect_telem"):
            RES._accum_telem(mask)
        used = router_logits.masked_fill(~mask.transpose(0, 1).reshape(M, E), float("-inf"))

    probs = F.softmax(used, dim=-1, dtype=torch.float)
    weights, selected = torch.topk(probs, self.gate.top_k, dim=-1)
    if getattr(self.gate, "norm_topk_prob", True):                 # q3 flag; q3.5 always
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(router_logits.dtype)

    out = self.experts(hs, selected, weights)                      # zoo grouped_mm forward

    # Training-time aux/z injection. The per-layer math is residency.aux_z_from_router_logits
    # verbatim -- P and f from the same (masked iff constrained) distribution, aux = E*(f*P).sum(),
    # z = logsumexp^2 -- computed here inside the checkpointed region so the gradient exists in
    # every checkpointing mode. Gated on grad being enabled: eval passes and the no-grad outer
    # pass of a reentrant checkpoint skip it entirely, so step-0 parity is untouched.
    inj = _CFG.get("aux_inject")
    if inj is not None and torch.is_grad_enabled():
        k = self.gate.top_k
        pf = probs                                                 # [M, E] float, post-mask
        P = pf.mean(0)
        idx = pf.topk(k, dim=-1).indices
        f = torch.zeros_like(P).scatter_add_(
            0, idx.reshape(-1),
            torch.ones(idx.numel(), device=P.device, dtype=P.dtype)) / pf.shape[0]
        aux = P.shape[0] * (f * P).sum()
        z = (torch.logsumexp(used.float(), dim=-1) ** 2).mean()
        out = _AuxInject.apply(out, aux, inj["aux"])
        out = _AuxInject.apply(out, z, inj["z"])
        AUX_LOG.append((float(aux.detach()), float(z.detach())))

    if shared_out is not None:
        out = out + F.sigmoid(self.shared_expert_gate(hs)) * shared_out

    return out.view(B, S, H) if hidden_states.dim() == 3 else out


def _block_classes():
    """The block classes present in this process, keyed by family."""
    out = {}
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        out["qwen3"] = Qwen3MoeSparseMoeBlock
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        out["qwen3_5"] = Qwen3_5MoeSparseMoeBlock
    except ImportError:
        pass
    return out


def install(model):
    """Patch the block classes the model actually uses and tag layer indices. Class-level,
    so it must run after `import unsloth` finishes applying the zoo patches — last write wins."""
    ntp = getattr(getattr(model.config, "text_config", model.config), "norm_topk_prob", True)
    if ntp is False:
        raise ValueError(
            "norm_topk_prob=False: masking logits before the softmax rescales top-k gate "
            "mass (~0.40 -> 1.0), the artifact that inflated OLMoE results ~10x. Refusing.")
    n = 0
    classes = _block_classes()
    used = set()
    for m in model.modules():
        for fam, cls in classes.items():
            if isinstance(m, cls):
                m._layer_idx = n
                n += 1
                used.add(fam)
    if n == 0:
        raise RuntimeError("no sparse-MoE blocks found on this model; nothing to patch")
    for fam in used:
        _ORIG.setdefault(fam, classes[fam].forward)
        classes[fam].forward = _forward
    return n


def restore():
    for fam, cls in _block_classes().items():
        if fam in _ORIG:
            cls.forward = _ORIG[fam]
