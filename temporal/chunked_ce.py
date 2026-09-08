"""Chunked fused output-layer + cross-entropy for the language-model head.

The 50k-vocab head is the memory and a large part of the time of the small isoFLOP runs: for one
micro-batch of 131k tokens Megatron materialises the logits in bf16 (13 GB) and fp32 (26 GB) plus
their gradients, 60 to 80 GB of a 90 GB step. This autograd function never materialises the full
logits: the forward walks the tokens in chunks, computes the chunk's logits, the log-sum-exp and the
target logit, and keeps only the per-token loss; the backward recomputes each chunk's logits, forms
softmax minus one-hot scaled by the incoming per-token gradient, and accumulates the hidden-state
and weight gradients chunk by chunk. Peak extra memory is one chunk of fp32 logits (8,192 tokens x
50k = 1.6 GB by default) instead of the whole batch's.

Numerics: the chunk logits come from the same bf16 GEMM as the output layer, the loss is computed in
fp32 as Megatron's fused vocab cross-entropy does, and the weight gradient is accumulated in fp32
across chunks (standard autograd accumulates it in bf16 per micro-batch), so the result matches the
unchunked path to bf16 rounding. Tensor-parallel size must be 1 (asserted by the caller).

Enabled in Megatron's GPTModel.forward by MOE_CHUNKED_CE=1 (recompute in the backward) or
MOE_CHUNKED_CE=2 (gradients computed in the forward, exact for a uniform loss mask, recompute
fallback otherwise); patch in scripts/residency/orchestration/megatron_chunked_ce.patch;
MOE_CHUNKED_CE_TOKENS sets the chunk.
Tests: temporal/tests/test_chunked_ce.py
"""
import os

import torch


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, chunk):
        # hidden [N, h] (bf16 or fp32), weight [V, h], target [N] int64 -> loss [N] fp32
        n = hidden.shape[0]
        loss = torch.empty(n, dtype=torch.float32, device=hidden.device)
        for i in range(0, n, chunk):
            h = hidden[i:i + chunk]
            logits = torch.matmul(h, weight.t()).float()
            t = target[i:i + chunk]
            lse = torch.logsumexp(logits, dim=-1)
            tgt = logits.gather(1, t.unsqueeze(1)).squeeze(1)
            loss[i:i + chunk] = lse - tgt
        ctx.save_for_backward(hidden, weight, target)
        ctx.chunk = chunk
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, target = ctx.saved_tensors
        chunk = ctx.chunk
        n = hidden.shape[0]
        grad_h = torch.empty_like(hidden)
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        grad_loss = grad_loss.to(torch.float32)
        for i in range(0, n, chunk):
            h = hidden[i:i + chunk]
            logits = torch.matmul(h, weight.t()).float()
            p = torch.softmax(logits, dim=-1)
            t = target[i:i + chunk]
            p[torch.arange(p.shape[0], device=p.device), t] -= 1.0
            p.mul_(grad_loss[i:i + chunk].unsqueeze(1))
            pb = p.to(hidden.dtype)
            grad_h[i:i + chunk] = torch.matmul(pb, weight.to(hidden.dtype))
            grad_w += torch.matmul(pb.t(), h).float()
        return grad_h, grad_w.to(weight.dtype), None, None


class _ChunkedLinearCEFwdGrad(torch.autograd.Function):
    """Same loss, gradients computed in the forward pass (one head GEMM plus the two gradient GEMMs
    per chunk, no recompute in the backward). Exact when the upstream per-token gradient is
    uniform, which Megatron's loss gives whenever the loss mask is all ones; the backward checks
    that and otherwise recomputes chunk by chunk like _ChunkedLinearCE."""
    @staticmethod
    def forward(ctx, hidden, weight, target, chunk):
        n = hidden.shape[0]
        loss = torch.empty(n, dtype=torch.float32, device=hidden.device)
        grad_h = torch.empty_like(hidden)
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        wt = weight.t()
        for i in range(0, n, chunk):
            h = hidden[i:i + chunk]
            logits = torch.matmul(h, wt).float()
            t = target[i:i + chunk]
            lse = torch.logsumexp(logits, dim=-1)
            loss[i:i + chunk] = lse - logits.gather(1, t.unsqueeze(1)).squeeze(1)
            logits.sub_(lse.unsqueeze(1)).exp_()                       # softmax, in place
            logits[torch.arange(logits.shape[0], device=logits.device), t] -= 1.0
            pb = logits.to(hidden.dtype)                                # unscaled dL/dlogits
            grad_h[i:i + chunk] = torch.matmul(pb, weight.to(hidden.dtype))
            grad_w += torch.matmul(pb.t(), h).float()
        ctx.save_for_backward(hidden, weight, target, grad_h, grad_w)
        ctx.chunk = chunk
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, target, grad_h, grad_w = ctx.saved_tensors
        g = grad_loss.to(torch.float32)
        gm = g.mean()
        if bool(((g - gm).abs() <= 1e-6 * (gm.abs() + 1e-12)).all()):
            return (grad_h * gm.to(grad_h.dtype)), (grad_w * gm).to(weight.dtype), None, None
        # non-uniform upstream gradient (a loss mask with zeros): exact recompute path
        return _ChunkedLinearCE.backward(_Ctx(hidden, weight, target, ctx.chunk), grad_loss)[:2] + (None, None)


class _Ctx:
    def __init__(self, hidden, weight, target, chunk):
        self.saved_tensors = (hidden, weight, target); self.chunk = chunk


def chunked_linear_cross_entropy(hidden, weight, target, chunk=None):
    """hidden [s, b, h], weight [V, h], target [s, b] (int64, negatives clamped; mask them upstream)
    -> per-token loss [s, b] in fp32. Equivalent to cross_entropy(hidden @ weight.T, target)."""
    chunk = chunk or int(os.environ.get("MOE_CHUNKED_CE_TOKENS", "8192"))
    s, b, h = hidden.shape
    flat = hidden.reshape(s * b, h)
    tgt = target.reshape(s * b).clamp_min(0)
    fn = _ChunkedLinearCEFwdGrad if os.environ.get("MOE_CHUNKED_CE") == "2" else _ChunkedLinearCE
    loss = fn.apply(flat, weight, tgt, chunk)
    return loss.view(s, b)
