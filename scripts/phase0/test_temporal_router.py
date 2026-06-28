#!/usr/bin/env python3
"""TDD specs for the rolling-residency selection (temporal MoE).

Pure-function tests for `compute_resident_mask` — no Megatron, no GPU, CPU torch only.
Run: .venv/bin/python -m pytest scripts/phase0/test_temporal_router.py
(or:  python3 -m pytest scripts/phase0/test_temporal_router.py)

Semantics under test (K = k, use-then-swap, deployment-faithful):
  - t=0 cold fill: R_0 = top-k(logits[0])  ("first token picks all experts")
  - mask[t] = R_t  (token t is served by the set available to it)
  - R_{t+1} = swap(R_t, logits[t]): nominate the best NON-resident expert; swap it in ONLY if it
    beats the worst resident (equivalently: R_t != global top-k); evict the LRU resident
    (oldest last-refresh; cold-fill refresh ranks by ascending logit, nominations are newest).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_router import compute_resident_mask


def _sets(mask, b=0):
    """List over t of the resident expert-index set for batch element b."""
    return [frozenset(torch.nonzero(mask[t, b]).flatten().tolist()) for t in range(mask.shape[0])]


def test_shape_and_count_invariant():
    torch.manual_seed(0)
    logits = torch.randn(7, 3, 10)
    k = 4
    mask = compute_resident_mask(logits, k)
    assert mask.shape == logits.shape
    assert mask.dtype == torch.bool
    # exactly k resident per (seq, batch) token
    assert torch.equal(mask.sum(dim=-1), torch.full((7, 3), k))


def test_cold_fill_is_first_token_topk():
    logits = torch.zeros(1, 1, 5)
    logits[0, 0] = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2])
    mask = compute_resident_mask(logits, k=2)
    assert _sets(mask)[0] == frozenset({1, 3})  # top-2 are idx 1 (0.9) and idx 3 (0.7)


def test_at_most_one_swap_per_step():
    torch.manual_seed(1)
    logits = torch.randn(20, 2, 8)
    mask = compute_resident_mask(logits, k=3)
    for b in range(2):
        s = _sets(mask, b)
        for t in range(len(s) - 1):
            # symmetric difference <= 2  <=>  at most one in, one out
            assert len(s[t] ^ s[t + 1]) <= 2


def test_noop_when_preference_stays_resident():
    # identical logits every step -> R_0 = top-k is always the global top-k -> never any swap.
    base = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2])
    logits = base.view(1, 1, 5).expand(6, 1, 5).contiguous()
    mask = compute_resident_mask(logits, k=2)
    s = _sets(mask)
    assert all(r == frozenset({1, 3}) for r in s)  # constant, no churn


def test_swap_evicts_lru_not_lowest_logit():
    # Hand-traced sequence where LRU (oldest refresh) and "lowest current logit" disagree,
    # so this pins that eviction is by refresh-recency, NOT by current logit.
    logits = torch.zeros(3, 1, 5)
    logits[0, 0] = torch.tensor([0.9, 0.8, 0.0, 0.0, 0.0])   # R_0 = {0,1}; refresh: idx1 older, idx0 newer
    logits[1, 0] = torch.tensor([0.1, 0.95, 0.9, 0.0, 0.0])  # wants idx2 (non-resident, 0.9 > worst 0.1)
    logits[2, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    s = _sets(compute_resident_mask(logits, k=2))
    assert s[0] == frozenset({0, 1})
    assert s[1] == frozenset({0, 1})           # served by R_0 (1-token prefetch lag)
    # idx1 evicted (oldest refresh) even though its current logit 0.95 is the HIGHER of the two;
    # a lowest-logit policy would have evicted idx0 (0.1) -> {1,2}. LRU gives {0,2}.
    assert s[2] == frozenset({0, 2})


def test_min_logit_evicts_lowest_resident_logit():
    # Same sequence, but evict="min_logit" removes idx0 (current logit 0.1, the lowest resident)
    # instead of the LRU idx1 -> {1,2} rather than {0,2}. Pins the eviction knob.
    logits = torch.zeros(3, 1, 5)
    logits[0, 0] = torch.tensor([0.9, 0.8, 0.0, 0.0, 0.0])
    logits[1, 0] = torch.tensor([0.1, 0.95, 0.9, 0.0, 0.0])
    logits[2, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    s = _sets(compute_resident_mask(logits, k=2, evict="min_logit"))
    assert s[0] == frozenset({0, 1})
    assert s[1] == frozenset({0, 1})
    assert s[2] == frozenset({1, 2})


def test_batch_elements_are_independent():
    a = torch.zeros(3, 1, 5)
    a[0, 0] = torch.tensor([0.9, 0.8, 0.0, 0.0, 0.0])
    a[1, 0] = torch.tensor([0.1, 0.95, 0.9, 0.0, 0.0])
    a[2, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    b = torch.zeros(3, 1, 5)
    b[0, 0] = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2])  # different sequence
    b[1, 0] = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2])
    b[2, 0] = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2])
    solo_a = _sets(compute_resident_mask(a, k=2))
    solo_b = _sets(compute_resident_mask(b, k=2))
    joint = compute_resident_mask(torch.cat([a, b], dim=1), k=2)
    assert _sets(joint, 0) == solo_a
    assert _sets(joint, 1) == solo_b


def test_deterministic():
    torch.manual_seed(2)
    logits = torch.randn(12, 4, 9)
    m1 = compute_resident_mask(logits, k=3)
    m2 = compute_resident_mask(logits, k=3)
    assert torch.equal(m1, m2)
