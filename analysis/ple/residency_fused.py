#!/usr/bin/env python3
"""Rolling residency for `transformers-qwen3-moe-fused`.

That library replaces the expert computation with sorted grouped-GEMM Triton kernels and leaves
routing in ordinary PyTorch, which is exactly the split this programme needs: the expert loop is
where the time goes (profiling put `torch.where` + `index_add_` at 89.5% of a layer and the GEMMs at
5.2%), and the router is where the intervention goes.

Integration is simpler than the stock-transformers path. `Qwen3MoeFusedSparseMoeBlock.forward` does
`gate -> softmax -> topk` inline with batch and sequence still in scope, so masking slots in directly
and none of the `_resid_shape` bookkeeping that stock needs is required -- stock hands the router a
flattened [B*S, E] with no way to recover S.

Everything that decides a number is still imported from `residency`: the resident-set scan, _CFG,
and the telemetry. This file only re-expresses where the mask is applied.

Gate-mass note: the mask is applied to the logits before the softmax, so on a model with
`norm_topk_prob=False` this rescales the gate weights exactly as the OLMoE artifact did (~0.40 -> 1.0
top-k mass). Qwen3-30B-A3B-Base sets it True, where renormalisation makes masking equivalent to
restricting the candidate set. `install()` refuses to run on a False model rather than silently
reintroducing the bug that cost this programme its headline result.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402

_CFG = RES._CFG
_ORIG = {}


def _forward(self, hidden_states):
    """Fused block forward with residency masking; body follows the library's own implementation."""
    from qwen3_moe_fused.kernels.indexing import get_expert_offsets_and_idx

    B, S, H = hidden_states.shape
    M = B * S
    hidden_states = hidden_states.view(M, H)
    router_logits = self.gate(hidden_states)                       # [M, E]

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
            mask = scan(lg.float(), _CFG["R"], evict=_CFG["evict"])
        if _CFG.get("collect_telem"):
            RES._accum_telem(mask)
        used = router_logits.masked_fill(~mask.transpose(0, 1).reshape(M, E), float("-inf"))

    routing_weights = F.softmax(used, dim=1, dtype=torch.float32)
    routing_weights, selected = torch.topk(routing_weights, self.num_selected, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    h = hidden_states.unsqueeze(1).expand(M, self.num_selected, H).reshape(M * self.num_selected, H)
    sel = selected.view(M * self.num_selected)
    m_offsets, sort_idx, inv_sort_idx = get_expert_offsets_and_idx(sel, self.num_experts)
    h = h[sort_idx]
    h = F.silu(self.gate_proj(h, m_offsets)) * self.up_proj(h, m_offsets)
    h = self.down_proj(h, m_offsets)
    h = h[inv_sort_idx].view(M, self.num_selected, H)
    out = torch.einsum("beo,be->bo", h, routing_weights).view(B, S, H)
    return out, router_logits


def install(model=None):
    sys.path.insert(0, "/workspace/qwen3-moe-fused")
    from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock as Blk
    if model is not None:
        ntp = getattr(model.config, "norm_topk_prob", None)
        if ntp is False:
            raise ValueError(
                "norm_topk_prob=False: masking logits before the softmax would also rescale the "
                "top-k gate mass (~0.40 -> 1.0), which is the artifact that inflated this "
                "programme's OLMoE results ~10x. Use the gate-mass-preserving path instead.")
    _ORIG.setdefault("blk", Blk.forward)
    Blk.forward = _forward
    n = 0
    if model is not None:
        for m in model.modules():
            if isinstance(m, Blk):
                m._layer_idx = n
                n += 1
    return n


def restore():
    from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock as Blk
    if "blk" in _ORIG:
        Blk.forward = _ORIG["blk"]
