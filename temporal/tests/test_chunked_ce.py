#!/usr/bin/env python3
"""chunked_linear_cross_entropy == cross_entropy(hidden @ W.T) in value and in both gradients,
with a chunk smaller than the token count so several chunks are exercised.
Run: $PY -m pytest temporal/tests/test_chunked_ce.py  (or the pytest-free runner used in the repo)
"""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal.chunked_ce import chunked_linear_cross_entropy  # noqa: E402


def _case(dtype, device, s=7, b=3, h=16, v=37, chunk=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(s, b, h, generator=g) * 0.5).to(device=device, dtype=dtype).requires_grad_(True)
    weight = (torch.randn(v, h, generator=g) * 0.1).to(device=device, dtype=dtype).requires_grad_(True)
    target = torch.randint(0, v, (s, b), generator=g).to(device)
    per_token_grad = torch.rand(s, b, generator=g).to(device)          # a loss-mask-like upstream gradient
    # reference: full logits, fp32 cross-entropy, same upstream gradient
    h2 = hidden.detach().clone().requires_grad_(True); w2 = weight.detach().clone().requires_grad_(True)
    logits = torch.matmul(h2.reshape(s * b, h), w2.t()).float()
    ref = F.cross_entropy(logits, target.reshape(-1), reduction="none").view(s, b)
    (ref * per_token_grad).sum().backward()
    out = chunked_linear_cross_entropy(hidden, weight, target, chunk=chunk)
    (out * per_token_grad).sum().backward()
    return ref, out, h2.grad, hidden.grad, w2.grad, weight.grad


def test_fp32_cpu_matches_reference_to_rounding():
    ref, out, gh_ref, gh, gw_ref, gw = _case(torch.float32, "cpu")
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gh, gh_ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gw, gw_ref, atol=1e-5, rtol=1e-4)


def test_output_is_fp32_and_shaped_like_target():
    _, out, _, _, _, _ = _case(torch.float32, "cpu", s=4, b=2)
    assert out.dtype == torch.float32 and out.shape == (4, 2)


def test_bf16_gpu_matches_reference_to_bf16_rounding():
    if not torch.cuda.is_available():
        return
    ref, out, gh_ref, gh, gw_ref, gw = _case(torch.bfloat16, "cuda", s=64, b=4, h=64, v=1000, chunk=50)
    assert torch.allclose(out, ref, atol=2e-2, rtol=1e-2)
    assert torch.allclose(gh.float(), gh_ref.float(), atol=2e-2, rtol=5e-2)
    assert torch.allclose(gw.float(), gw_ref.float(), atol=2e-2, rtol=5e-2)


def test_forward_grad_variant_uniform_and_masked(monkeypatch=None):
    os.environ["MOE_CHUNKED_CE"] = "2"
    try:
        # uniform upstream gradient (all-ones mask / N): stored gradients, exact to rounding
        g = torch.Generator().manual_seed(3)
        s, b, h, v = 6, 2, 16, 29
        hidden = (torch.randn(s, b, h, generator=g)).requires_grad_(True); weight = (torch.randn(v, h, generator=g) * 0.1).requires_grad_(True)
        target = torch.randint(0, v, (s, b), generator=g)
        h2 = hidden.detach().clone().requires_grad_(True); w2 = weight.detach().clone().requires_grad_(True)
        ref = F.cross_entropy(torch.matmul(h2.reshape(-1, h), w2.t()), target.reshape(-1), reduction="none").view(s, b)
        (ref.sum() / (s * b)).backward()
        out = chunked_linear_cross_entropy(hidden, weight, target, chunk=5)
        (out.sum() / (s * b)).backward()
        assert torch.allclose(out, ref, atol=1e-5) and torch.allclose(hidden.grad, h2.grad, atol=1e-5) and torch.allclose(weight.grad, w2.grad, atol=1e-5)
        # masked (non-uniform) upstream gradient: the recompute fallback, still exact
        hidden.grad = None; weight.grad = None; h2.grad = None; w2.grad = None
        mask = (torch.rand(s, b, generator=g) > 0.3).float()
        ref = F.cross_entropy(torch.matmul(h2.reshape(-1, h), w2.t()), target.reshape(-1), reduction="none").view(s, b)
        (ref * mask).sum().backward()
        out = chunked_linear_cross_entropy(hidden, weight, target, chunk=5)
        (out * mask).sum().backward()
        assert torch.allclose(hidden.grad, h2.grad, atol=1e-5) and torch.allclose(weight.grad, w2.grad, atol=1e-5)
    finally:
        os.environ.pop("MOE_CHUNKED_CE", None)


def test_negative_targets_are_clamped_not_crashed():
    hidden = torch.randn(3, 2, 8, requires_grad=True); weight = torch.randn(11, 8, requires_grad=True)
    target = torch.tensor([[0, -1], [5, 3], [-1, 10]])
    out = chunked_linear_cross_entropy(hidden, weight, target, chunk=4)
    assert torch.isfinite(out).all()
