#!/usr/bin/env python3
"""TDD specs for the temporal-coherence auxiliary loss (BCE).

Pure-function tests for `coherence_bce_loss` — no Megatron, no GPU, CPU torch only.
Run: .venv/bin/python -m pytest scripts/phase0/test_coherence_loss.py

Semantics under test:
  loss = BCE(sigmoid(logits), target), target = final resident/used set (multi-hot, detached).
  Independent per-expert sigmoids (NOT softmax) -> set-membership pull, no distribution clone.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_router import coherence_bce_loss


def _mask(*rows):
    """Build a [T,1,E] bool mask from python lists (one row per token)."""
    return torch.tensor([[r] for r in rows], dtype=torch.bool)


def test_zero_when_confident_and_aligned():
    # logits huge-positive on the resident set, huge-negative elsewhere -> BCE ~ 0.
    m = _mask([1, 1, 0, 0], [0, 0, 1, 1])
    logits = torch.where(m, 20.0, -20.0)
    assert coherence_bce_loss(logits, m).item() < 1e-6


def test_large_when_anti_aligned():
    # logits confident but on the WRONG experts -> large loss (>> aligned case).
    m = _mask([1, 1, 0, 0], [0, 0, 1, 1])
    aligned = coherence_bce_loss(torch.where(m, 20.0, -20.0), m).item()
    anti = coherence_bce_loss(torch.where(m, -20.0, 20.0), m).item()
    assert anti > 10.0 and anti > aligned + 10.0


def test_gradient_pushes_resident_up_and_others_down():
    # BCEWithLogits grad wrt logit = (sigmoid(z) - y)/N: <0 on resident (y=1, pushes z UP under
    # gradient descent), >0 on non-resident (y=0, pushes z DOWN). This is the retention mechanism.
    m = _mask([1, 1, 0, 0], [0, 0, 1, 1])
    z = torch.zeros(2, 1, 4, requires_grad=True)
    coherence_bce_loss(z, m).backward()
    assert (z.grad[m] < 0).all()      # resident experts: gradient descent raises their logits
    assert (z.grad[~m] > 0).all()     # non-resident: lowered


def test_target_is_detached_matches_analytic_gradient():
    # If the target were part of the graph the gradient would differ; assert it equals the exact
    # constant-target formula (sigmoid(z) - y)/N -> confirms detachment.
    torch.manual_seed(0)
    m = torch.rand(3, 2, 5) > 0.5
    z = torch.randn(3, 2, 5, requires_grad=True)
    coherence_bce_loss(z, m).backward()
    expected = (torch.sigmoid(z.detach()) - m.float()) / z.numel()
    assert torch.allclose(z.grad, expected, atol=1e-6)


def test_scalar_over_seq_batch():
    torch.manual_seed(1)
    m = torch.rand(2048, 8, 64) > 0.9
    z = torch.randn(2048, 8, 64)
    out = coherence_bce_loss(z, m)
    assert out.dim() == 0 and torch.isfinite(out)


def test_higher_retention_lowers_loss():
    # A router whose free preference already matches the resident set (coherent) scores lower than
    # one that keeps wanting non-resident experts (churny) — the loss rewards retention.
    m = _mask([1, 1, 0, 0], [1, 1, 0, 0])
    coherent = torch.where(m, 3.0, -3.0)                          # prefers resident every token
    churny = torch.tensor([[[3.0, 3.0, -3.0, -3.0]], [[-3.0, -3.0, 3.0, 3.0]]])  # flips away
    assert coherence_bce_loss(coherent, m).item() < coherence_bce_loss(churny, m).item()
