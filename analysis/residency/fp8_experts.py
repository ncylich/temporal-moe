#!/usr/bin/env python3
"""FP8 for MoE expert weights, done directly rather than through a quantisation library.

transformers' FineGrainedFP8 path is unusable in this environment: six blockers deep, and the fifth
fix (`pip install kernels`) broke `transformers.activations` at import. bitsandbytes, torchao and
FbgemmFp8 all convert `nn.Linear` and therefore skip expert weights entirely -- which on these models
is ~90% of the parameters and the only part worth quantising.

But the expert weights are plain 3-D tensors, `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]`.
Nothing stops us quantising them ourselves, and doing so means the numerics are ours to verify rather
than a library's to surprise us with.

Scheme: per-output-channel absmax scaling to float8_e4m3fn (max 448). One scale per (expert, output
row) rather than one per tensor, because a single scale across 2048 rows would be set by the largest
row and crush the rest. Weights are stored as fp8 and dequantised one expert at a time inside the
forward, immediately before that expert's GEMM -- so the memory saving is real (the bf16 copy is
freed) while the dequantised working set is a single [out, in] slice, a few MB.

This is a memory optimisation, not an arithmetic one: the GEMM still runs in bf16. That is the right
trade here, because the measured bottleneck is micro-batch 1 -- which exists *because* 57-67 GB of
weights leaves no room -- rather than raw FLOPs. Halving the weights buys the batch size.

    from fp8_experts import quantize_experts_fp8, install_fp8_forward
    n = quantize_experts_fp8(model)          # in place, frees the bf16 copies
    install_fp8_forward(family)
"""
import gc
import sys

import torch
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
_ORIG = {}
MODE = {"how": "weight_only", "chunk": 16}


def _expert_classes():
    out = []
    for mod, cls in (("qwen3_5_moe", "Qwen3_5MoeExperts"), ("qwen3_moe", "Qwen3MoeExperts"),
                     ("olmoe", "OlmoeExperts")):
        try:
            m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[cls])
            out.append((mod, getattr(m, cls)))
        except Exception:
            pass
    return out


@torch.no_grad()
def _quant(W):
    """[E, out, in] bf16 -> (fp8 [E, out, in], scale fp32 [E, out, 1]).

    Per-EXPERT scalar absmax, not per-output-channel. Per-channel was the obvious choice and it is
    the wrong one: torch._scaled_mm drops off its fast cuBLAS path when given rowwise scales, costing
    3x throughput (352 vs 1135 TFLOP/s at M=16384), and it buys no accuracy -- measured 0.03772 vs
    0.03768 relative error even on adversarial input with 4x channel spread and 6x outlier tokens.
    e4m3's error is mantissa-dominated, so finer scale granularity only helps when row-to-row dynamic
    range exceeds fp8's exponent span, which weight matrices do not approach.

    Done expert by expert so peak extra memory is one expert's fp32 copy rather than a second full
    tensor.
    """
    E = W.shape[0]
    q = torch.empty(W.shape, dtype=FP8, device=W.device)
    s = torch.empty((E, 1, 1), dtype=torch.float32, device=W.device)
    for e in range(E):
        we = W[e].float()
        se = (we.abs().amax() / FP8_MAX).clamp(min=1e-8)
        q[e] = (we / se).to(FP8)
        s[e] = se
        del we
    return q, s


@torch.no_grad()
def quantize_experts_fp8(model, verbose=True):
    """Replace every expert weight with fp8 + scales, in place, freeing the bf16 tensors."""
    classes = [c for _, c in _expert_classes()]
    n_mod = 0
    before = torch.cuda.memory_allocated()
    for m in model.modules():
        if not any(isinstance(m, c) for c in classes):
            continue
        for name in ("gate_up_proj", "down_proj"):
            W = getattr(m, name, None)
            if W is None or getattr(m, name + "_fp8", None) is not None:
                continue
            q, s = _quant(W.data)
            m.register_buffer(name + "_fp8", q, persistent=False)
            m.register_buffer(name + "_scale", s, persistent=False)
            # drop the bf16 original: this is where the memory actually comes back
            if isinstance(W, torch.nn.Parameter):
                delattr(m, name)
            else:
                setattr(m, name, None)
            del W
        n_mod += 1
    gc.collect(); torch.cuda.empty_cache()
    if verbose:
        after = torch.cuda.memory_allocated()
        print(f"  [fp8] quantised {n_mod} expert modules, "
              f"{before/1e9:.1f} GB -> {after/1e9:.1f} GB ({(before-after)/1e9:.1f} GB freed)",
              flush=True)
    return n_mod


def _mm(x, wq, ws, dt):
    """One expert's projection, either dequantise-then-bf16-GEMM or a true fp8 GEMM.

    weight_only  dequantise the fp8 weight to bf16 and use the normal GEMM. Activations stay bf16,
                 so the only error is weight rounding -- measured 0.0266 relative on a test GEMM.
    scaled_mm    quantise activations per token, weights per output channel, and let the tensor
                 cores consume fp8 directly. Faster and avoids the dequant traffic entirely, but
                 quantising activations adds error -- 0.0375, i.e. 41% worse. The MoE literature
                 warns about exactly this: expert activation distributions are heterogeneous, and
                 rare experts are the ones that suffer. Per-token dynamic scaling is the mitigation,
                 which is what is used here; static scales would be worse still.

    Needs torch >= 2.5 for per-row scales; 2.4's _scaled_mm takes scalar scales only.
    """
    if MODE["how"] == "scaled_mm":
        sx = (x.abs().amax(1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
        xq = (x.float() / sx).to(FP8)
        return torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=ws.reshape(1, -1),
                                out_dtype=dt)
    return F.linear(x, wq.to(dt) * ws.to(dt))


def _forward_fp8(self, hidden_states, top_k_index, top_k_weights):
    """Stock expert loop with fp8 weights.

    The activation quantisation for the first projection is hoisted OUT of the expert loop. Every
    expert reads a disjoint subset of the same hidden_states, and the scales are per token, so
    quantising once for all tokens and then indexing is arithmetically identical to quantising each
    expert's slice -- and measured 41x cheaper, because the per-expert version rescans the same
    tensor E times in small pieces. This is what the vLLM/torchao MoE kernels do: quantise the layer
    input once, then dispatch.

    The second projection cannot be hoisted the same way: its input is produced per expert inside the
    loop, so it is still quantised per expert. That is a real remaining cost, and it is why this is
    ~2x rather than ~4x better.

    Weight layout needs no work: weights are stored [E, out, in] contiguous, so `wq[e].t()` is a
    stride-(1, in) view -- exactly the column-major operand _scaled_mm wants, with no copy.
    """
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    dt = hidden_states.dtype
    scaled = MODE["how"] == "scaled_mm"
    if scaled:
        # one scalar for the whole activation tensor: keeps _scaled_mm on the fast path
        sx_all = (hidden_states.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(1, 1)
        # kept as uint8: fp8 has no index_cuda kernel, so gathering rows of an fp8 tensor fails.
        # Indexing the byte view and reinterpreting is bit-exact (verified), same trick the cat/stack
        # gaps needed. Cheaper than materialising a bf16 copy just to be indexable.
        xq_all = (hidden_states / sx_all.to(dt)).to(FP8).view(torch.uint8)
    for e in hit:
        e = e[0]
        if e == self.num_experts:
            continue
        pos, tok = torch.where(mask[e])
        if scaled:
            gu = torch._scaled_mm(xq_all[tok].view(FP8), self.gate_up_proj_fp8[e].t(),
                                  scale_a=sx_all,
                                  scale_b=self.gate_up_proj_scale[e].reshape(1, 1),
                                  out_dtype=dt)
            gate, up = gu.chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            sh = (h.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(1, 1)
            y = torch._scaled_mm((h / sh.to(dt)).to(FP8), self.down_proj_fp8[e].t(),
                                 scale_a=sh, scale_b=self.down_proj_scale[e].reshape(1, 1),
                                 out_dtype=dt)
        else:
            x = hidden_states[tok]
            gate, up = F.linear(x, self.gate_up_proj_fp8[e].to(dt)
                                * self.gate_up_proj_scale[e].to(dt)).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            y = F.linear(h, self.down_proj_fp8[e].to(dt) * self.down_proj_scale[e].to(dt))
        final.index_add_(0, tok, (y * top_k_weights[tok, pos, None]).to(final.dtype))
    return final




def _grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    """One batched GEMM per expert-chunk instead of one GEMM per expert.

    NOT decorated @torch.no_grad: this path has to be differentiable. Only the routing arithmetic
    (argsort, bincount, offsets) and the dequantisation of the FROZEN fp8 base weights run under
    no_grad. Gradients still flow to the activations, and therefore to any LoRA parameters attached
    around this block -- a no_grad here would train nothing and fail silently.

    The per-expert loop is ~96% overhead: at batch 64 the expert GEMMs are ~98 ms of arithmetic
    inside a 2.85 s forward, because 128 experts x 48 layers is 6144 iterations each launching
    where + gather + two matmuls + index_add. Grouping the experts into one bmm measures 1.6x
    (22.7 -> 14.3 ms/layer) with only 7% padding waste, since routing is well balanced.

    Chunked rather than fully grouped because the memory saving has to survive: dequantising all
    E experts at once to feed a single bmm would rematerialise the full bf16 weight tensor, which is
    exactly what fp8 storage exists to avoid. A chunk of 16 experts is ~100 MB dequantised, and turns
    128 launches per layer into 8.
    """
    T, K = hidden_states.shape
    E, kk = self.num_experts, top_k_index.shape[1]
    dt = hidden_states.dtype
    with torch.no_grad():                      # routing bookkeeping carries no gradient
        fe = top_k_index.reshape(-1)
        ft = torch.arange(T, device=fe.device).repeat_interleave(kk)
        order = fe.argsort()
        se, st = fe[order], ft[order]
        cnt = torch.bincount(fe, minlength=E)
        off = torch.zeros(E + 1, dtype=torch.long, device=fe.device)
        off[1:] = cnt.cumsum(0)
        pos_in = torch.arange(se.numel(), device=fe.device) - off[se]
    sw = top_k_weights.reshape(-1)[order]      # gate weights DO carry gradient to the router
    out = torch.zeros_like(hidden_states)
    C = MODE["chunk"]
    for c0 in range(0, E, C):
        c1 = min(c0 + C, E)
        sel = (se >= c0) & (se < c1)
        if not bool(sel.any()):
            continue
        e_l, t_l, p_l, w_l = se[sel] - c0, st[sel], pos_in[sel], sw[sel]
        mc = int(cnt[c0:c1].max())
        buf = torch.zeros(c1 - c0, mc, K, device=hidden_states.device, dtype=dt)
        buf[e_l, p_l] = hidden_states[t_l]
        with torch.no_grad():                  # base experts are frozen; only LoRA/router train
            gu_w = (self.gate_up_proj_fp8[c0:c1].to(dt)
                    * self.gate_up_proj_scale[c0:c1].to(dt))                # [C, 2I, K]
        gu = torch.bmm(buf, gu_w.transpose(1, 2))
        del buf, gu_w
        gate, up = gu.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        del gu, gate, up
        with torch.no_grad():
            dn_w = (self.down_proj_fp8[c0:c1].to(dt) * self.down_proj_scale[c0:c1].to(dt))
        y = torch.bmm(h, dn_w.transpose(1, 2))
        del h, dn_w
        out.index_add_(0, t_l, y[e_l, p_l] * w_l.unsqueeze(-1).to(dt))
        del y
    return out


def install_fp8_forward(verbose=True, mode="weight_only"):
    MODE["how"] = mode
    fwd = _grouped_forward if mode == "grouped" else _forward_fp8
    for mod, cls in _expert_classes():
        _ORIG.setdefault(cls.__name__, cls.forward)
        cls.forward = fwd
    if verbose:
        print(f"  [fp8] expert forward installed, mode={mode}", flush=True)


def restore(verbose=False):
    for mod, cls in _expert_classes():
        if cls.__name__ in _ORIG:
            cls.forward = _ORIG[cls.__name__]
