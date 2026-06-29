#!/usr/bin/env python3
"""TDD specs for the rolling-residency selection (temporal MoE).

Pure-function tests for `compute_resident_mask` — no Megatron, no GPU, CPU torch only.
Run: .venv/bin/python -m pytest scripts/phase0/test_temporal_router.py
(or:  python3 -m pytest scripts/phase0/test_temporal_router.py)

Semantics under test (K = k, swap-then-use — a token pulls in one expert and uses it the SAME step):
  - t=0 cold fill: R_0 = top-k(logits[0])  ("first token picks all experts")
  - for t >= 1: R_t = swap(R_{t-1}, logits[t]): nominate the best NON-resident expert; swap it in
    ONLY if it beats the worst resident (equivalently: R_{t-1} != global top-k); evict per `evict`
    (lru = oldest last-refresh; min_logit = lowest current logit). mask[t] = R_t (the set t uses).
"""
import os, sys
import torch
import pytest

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


def test_swap_then_use_serves_new_expert_same_step():
    # A token that wants a non-resident expert pulls it in and USES it on the same step (no lag):
    # idx2 is wanted at t=1 and must appear in mask[1], not a step later.
    logits = torch.zeros(2, 1, 5)
    logits[0, 0] = torch.tensor([0.9, 0.8, 0.0, 0.0, 0.0])   # R_0 = {0,1}
    logits[1, 0] = torch.tensor([0.1, 0.95, 0.9, 0.0, 0.0])  # top-2 = {1,2}; idx2 is non-resident
    assert 2 in _sets(compute_resident_mask(logits, k=2))[1]


def test_lru_and_min_logit_diverge_then_reconverge():
    # Trace where LRU and least-wanted GENUINELY disagree, and where LRU does the counter-intuitive
    # thing: it evicts the resident with the HIGHEST current logit (idx1=10) purely because it is the
    # oldest, while min_logit keeps it. This is exactly the asymmetry the policies are meant to test.
    logits = torch.zeros(3, 1, 4)
    logits[0, 0] = torch.tensor([10.0, 9.0, 0.0, 0.0])  # cold fill R_0={0,1}; idx1 oldest, idx0 newest
    logits[1, 0] = torch.tensor([1.0, 10.0, 9.0, 0.0])  # idx0 now low (1), idx1 high (10); pull in idx2 (9)
    logits[2, 0] = torch.tensor([1.0, 10.0, 9.0, 0.0])
    lru = _sets(compute_resident_mask(logits, k=2, evict="lru"))
    minl = _sets(compute_resident_mask(logits, k=2, evict="min_logit"))
    assert lru[0] == minl[0] == frozenset({0, 1})
    # t=1: min_logit keeps idx1 (logit 10, most wanted) -> {1,2};
    #      LRU evicts idx1 *despite* its logit 10 (it is the oldest) -> {0,2}.  They differ.
    assert minl[1] == frozenset({1, 2})
    assert lru[1] == frozenset({0, 2})
    assert lru[1] != minl[1]
    # t=2: LRU spends an extra swap to undo its mistake; both end at {1,2}.
    assert lru[2] == minl[2] == frozenset({1, 2})


def test_policies_actually_differ_on_a_longer_sequence():
    # Guard against the two policies silently collapsing to identical behavior.
    torch.manual_seed(7)
    logits = torch.randn(40, 2, 12)
    lru = compute_resident_mask(logits, k=4, evict="lru")
    minl = compute_resident_mask(logits, k=4, evict="min_logit")
    assert not torch.equal(lru, minl)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU fast path needs a GPU")
def test_accel_matches_reference_on_gpu():
    # The default accelerator (compute_resident_mask_accel -> Triton single-launch scan) must equal
    # the eager reference bit-for-bit at the production shape and both eviction policies, including
    # integer-valued logits with deliberate ties (the tie-breaking regime the CPU tests use).
    # Guards the GPU-only path the CPU tests above can't reach.
    import temporal_router as tr
    tr._scan_path = None; tr._graph_cache.clear()      # exercise the fast path from a clean state
    torch.manual_seed(3)
    for evict in ("lru", "min_logit"):
        for S, B, E, k in [(2048, 32, 64, 6), (2048, 32, 64, 5), (50, 4, 12, 3), (40, 2, 12, 4)]:
            logits = torch.randn(S, B, E, device="cuda")
            ref = compute_resident_mask(logits, k, evict)
            acc = tr.compute_resident_mask_accel(logits, k, evict)
            assert torch.equal(acc, ref), f"accel != ref for evict={evict} shape={(S,B,E)} k={k}"
            # integer logits with heavy ties -> stresses argmax/argmin/topk tie-breaking
            tied = torch.randint(0, 4, (S, B, E), device="cuda").float()
            ref_t = compute_resident_mask(tied, k, evict)
            acc_t = tr.compute_resident_mask_accel(tied, k, evict)
            assert torch.equal(acc_t, ref_t), f"accel != ref (ties) evict={evict} shape={(S,B,E)} k={k}"
    assert tr._scan_path == "triton", f"triton path did not engage (path={tr._scan_path})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU fast path needs a GPU")
def test_graph_path_matches_reference_on_gpu():
    # The alternate CUDA-graph fast path (TEMPORAL_SCAN=graph) must also stay bit-exact.
    import os, temporal_router as tr
    tr._scan_path = None; tr._graph_cache.clear()
    os.environ["TEMPORAL_SCAN"] = "graph"
    try:
        torch.manual_seed(4)
        for evict in ("lru", "min_logit"):
            for S, B, E, k in [(2048, 32, 64, 6), (50, 4, 12, 3)]:
                logits = torch.randn(S, B, E, device="cuda")
                ref = compute_resident_mask(logits, k, evict)
                acc = tr.compute_resident_mask_accel(logits, k, evict)
                assert torch.equal(acc, ref), f"graph != ref evict={evict} shape={(S,B,E)} k={k}"
        assert tr._scan_path == "cuda-graph", f"graph path did not engage (path={tr._scan_path})"
    finally:
        os.environ.pop("TEMPORAL_SCAN", None)
        tr._scan_path = None; tr._graph_cache.clear()
