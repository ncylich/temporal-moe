#!/usr/bin/env python3
"""TDD specs for the rolling-residency selection (temporal MoE) — CORE mechanism.

Pure-function tests for `compute_resident_mask` (+ the GPU accel scan, the tau/ema_beta trigger
knobs, and the residency-set-size R knob) — no Megatron, GPU only where marked, CPU torch otherwise.
Run: $PY -m pytest temporal/tests/test_temporal_router.py

Experimental (default-off, negative-result) scoring/loss knobs are tested separately in
test_ablation_mechanisms.py.

Semantics under test (K = k, swap-then-use — a token pulls in one expert and uses it the SAME step):
  - t=0 cold fill: R_0 = top-k(logits[0])  ("first token picks all experts")
  - for t >= 1: R_t = swap(R_{t-1}, logits[t]): nominate the best NON-resident expert; swap it in
    ONLY if it beats the worst resident (equivalently: R_{t-1} != global top-k); evict per `evict`
    (lru = oldest last-refresh; min_logit = lowest current logit). mask[t] = R_t (the set t uses).
"""
import os, sys
import torch
import pytest

# repo root on path -> import the `temporal` package and (for cross-checks) probe_replay.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal import temporal_router as tr
from temporal.temporal_router import compute_resident_mask, banner_knobs


def _sets(mask, b=0):
    """List over t of the resident expert-index set for batch element b."""
    return [frozenset(torch.nonzero(mask[t, b]).flatten().tolist()) for t in range(mask.shape[0])]


def _ar1_logits_t(S=160, B=3, E=16, seed=11):
    torch.manual_seed(seed)
    base = torch.randn(B, E)
    lg = torch.empty(S, B, E)
    lg[0] = base
    for t in range(1, S):
        base = 0.95 * base + 0.3 * torch.randn(B, E)
        lg[t] = base
    return lg


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


# ---------------------------------------------------------------------------
# Trigger-shaping knobs (tau hysteresis + EMA smoothing) — B1 eval-time path.
# ---------------------------------------------------------------------------
def test_knob_defaults_are_shipped_identity():
    """tau=0, ema_beta=1 must reproduce the shipped mask bit-exactly (both evict policies)."""
    lg = _ar1_logits_t()
    for evict in ("lru", "min_logit"):
        a = compute_resident_mask(lg, 4, evict)
        b = compute_resident_mask(lg, 4, evict, tau=0.0, ema_beta=1.0)
        assert torch.equal(a, b)


def test_tau_monotonically_reduces_swaps_and_freezes_at_large_tau():
    lg = _ar1_logits_t()
    prev = float("inf")
    for tau in (0.0, 0.5, 1.0, 2.0, 8.0, 1e9):
        m = compute_resident_mask(lg, 4, "min_logit", tau=tau)
        swaps = (m[1:] != m[:-1]).any(-1).float().mean().item()
        assert swaps <= prev + 1e-9
        prev = swaps
    assert swaps == 0.0                      # tau=1e9: cold-fill set never changes


def test_ema_beta_coldfill_matches_raw_and_is_causal():
    lg = _ar1_logits_t()
    m = compute_resident_mask(lg, 4, "min_logit", ema_beta=0.25)
    raw0 = torch.zeros_like(m[0])
    raw0.scatter_(1, lg[0].topk(4, -1).indices, True)
    assert torch.equal(m[0], raw0)           # EMA is identity at t=0
    lg2 = lg.clone(); lg2[120:] += 10.0
    m2 = compute_resident_mask(lg2, 4, "min_logit", ema_beta=0.25)
    assert torch.equal(m[:120], m2[:120])    # future logits cannot affect past masks


def test_knobs_match_probe_replay_policy():
    """The eval-time knob must implement EXACTLY the policy the A1 replay selected: same swap
    stream as probe_replay.replay(_ema(raw, beta), k, tau=tau) on identical inputs."""
    pytest.importorskip("numpy")
    import numpy as np
    try:
        import probe_replay as pr
    except Exception:
        pytest.skip("probe_replay (plot_probe deps) unavailable in this environment")
    lg = _ar1_logits_t(seed=13)
    tau, beta, k = 1.0, 0.25, 4
    m = compute_resident_mask(lg, k, "min_logit", tau=tau, ema_beta=beta)
    swaps_router = (m[1:] != m[:-1]).any(-1).numpy()                      # [S-1, B]
    lg_np = lg.numpy().astype(np.float32)
    out = pr.replay(pr._ema(lg_np, beta), k, evict="min_logit", tau=tau, eval_lg=lg_np)
    swaps_replay = out["swaps"][1:]                                       # [S-1, B]
    assert (swaps_router == swaps_replay).mean() > 0.995                  # float-tie tolerance


# ---------------------------------------------------------------------------
# R-knob — residency-set size R >= k decoupled from top-k (de-lexicalization dose).
# ---------------------------------------------------------------------------
def test_residency_R_mask_size_and_swap_budget():
    """With set size R > k the mask must hold exactly R residents per token, cold-fill = top-R of
    the first token, and the <=1-swap/token budget must hold on the R-set (symmetric difference
    of consecutive sets <= 2)."""
    lg = _ar1_logits_t(seed=71)                            # E=16
    R = 8
    m = compute_resident_mask(lg, R, "min_logit")
    assert torch.equal(m.sum(dim=-1), torch.full(m.shape[:2], R))
    cold = torch.zeros_like(m[0])
    cold.scatter_(1, lg[0].topk(R, -1).indices, True)
    assert torch.equal(m[0], cold)
    for b in range(m.shape[1]):
        s = _sets(m, b)
        assert all(len(s[t] ^ s[t + 1]) <= 2 for t in range(len(s) - 1))


def test_residency_R_equals_E_is_unconstrained():
    """R = E must make every expert always resident (all-True mask): the cold fill takes the whole
    pool and the swap trigger never fires (no non-resident nominee) -> masked routing == full MoE."""
    lg = _ar1_logits_t(seed=72)
    E = lg.shape[-1]
    m = compute_resident_mask(lg, E, "min_logit")
    assert m.all()


def test_residency_R_banner():
    os.environ.pop("TEMPORAL_RESIDENCY_R", None)
    assert "residency_R" not in banner_knobs()
    os.environ["TEMPORAL_RESIDENCY_R"] = "36"
    try:
        assert "residency_R=36" in banner_knobs()
    finally:
        os.environ.pop("TEMPORAL_RESIDENCY_R", None)


# ---------------------------------------------------------------------------
# Eviction as a temporal filter (docs/research/mechanism/lru-as-convolution.md).
#
# `lru` refreshes only on ADMISSION, so it is a FIFO queue over the admission stream — a box
# kernel of width k. The two derived policies widen or reshape that kernel: `lrd` also refreshes
# on demand (textbook LRU), `ema` evicts on a causal EMA of the demand indicator.
# ---------------------------------------------------------------------------
def _admissions(mask, b=0):
    """(t, expert) admissions read off a resident-set trace."""
    s = _sets(mask, b)
    return [(t, next(iter(s[t] - s[t - 1]))) for t in range(1, len(s)) if s[t] - s[t - 1]]


def test_lru_is_a_box_kernel_of_width_k_over_the_admission_stream():
    """THE structural claim: under `lru` the resident set is exactly the k most recent admissions
    (cold fill counting as k pseudo-admissions ordered by logit). Equivalently, residency is the
    box-kernel convolution of the admission stream — the set is a pure function of admission ORDER
    and carries no information about the scores."""
    lg = _ar1_logits_t(S=200, B=3, E=20, seed=31)
    k = 5
    m = compute_resident_mask(lg, k, "lru")
    for b in range(lg.shape[1]):
        s = _sets(m, b)
        # cold-fill pseudo-admissions: the k experts of R_0, oldest (lowest logit) first
        order = lg[0, b].topk(k, -1).indices.tolist()[::-1]
        for t, e in _admissions(m, b):
            order.append(e)
            assert s[t] == frozenset(order[-k:]), f"b={b} t={t}: not the last-{k} admissions"


def test_lru_evicts_at_age_exactly_k_admissions():
    """Corollary of the box kernel: every expert's residency lasts exactly k admission events —
    residency time is allocated uniformly, independent of how strongly the expert is wanted."""
    lg = _ar1_logits_t(S=200, B=2, E=20, seed=32)
    k = 5
    m = compute_resident_mask(lg, k, "lru")
    for b in range(lg.shape[1]):
        s = _sets(m, b)
        # cold fill = k pseudo-admissions: highest logit newest (clock 0), lowest oldest (-(k-1)).
        born = {e: -i for i, e in enumerate(lg[0, b].topk(k, -1).indices.tolist())}
        clock, ages = 0, []
        for t in range(1, len(s)):
            adm = s[t] - s[t - 1]
            if adm:
                clock += 1                       # eviction and admission are the same event
            for e in s[t - 1] - s[t]:
                ages.append(clock - born[e])
            for e in adm:
                born[e] = clock
        assert ages, "no evictions in the trace"
        assert set(ages) == {k}, f"ages should all be {k}, got {sorted(set(ages))}"


def test_new_evict_policies_hold_the_core_invariants():
    """`lrd` and `ema` are drop-in eviction kernels: same mask shape, exactly k residents, the
    same cold fill, and the same <=1-swap/token budget as the shipped pair."""
    lg = _ar1_logits_t(S=120, B=3, E=16, seed=33)
    k = 4
    cold = torch.zeros_like(lg[0], dtype=torch.bool)
    cold.scatter_(1, lg[0].topk(k, -1).indices, True)
    for evict in ("lrd", "ema"):
        m = compute_resident_mask(lg, k, evict)
        assert m.shape == lg.shape and m.dtype == torch.bool
        assert torch.equal(m.sum(-1), torch.full(m.shape[:2], k))
        assert torch.equal(m[0], cold)
        for b in range(lg.shape[1]):
            s = _sets(m, b)
            assert all(len(s[t] ^ s[t + 1]) <= 2 for t in range(len(s) - 1))


def test_new_evict_policies_differ_from_the_shipped_pair():
    """Guard against a new kernel silently collapsing onto `lru` or `min_logit`."""
    lg = _ar1_logits_t(S=160, B=2, E=16, seed=34)
    k = 4
    ms = {e: compute_resident_mask(lg, k, e) for e in ("lru", "min_logit", "lrd", "ema")}
    for a, b in [("lrd", "lru"), ("lrd", "min_logit"), ("ema", "lru"), ("ema", "min_logit"),
                 ("ema", "lrd")]:
        assert not torch.equal(ms[a], ms[b]), f"{a} collapsed onto {b}"


def test_lrd_refreshes_on_demand_where_lru_does_not():
    """The one-line difference, traced: an expert that is still in the token's unconstrained top-k
    is protected by `lrd` and evicted by `lru` (which only counts admission age)."""
    #                          e0     e1    e2   e3
    lg = torch.tensor([[[10.0,  9.0,  0.0, 0.0]],     # cold fill R_0 = {0,1}; e1 the older slot
                       [[1.0,  10.0,  9.0, 0.0]]])    # admit e2; e1 still demanded, e0 is not
    lru = _sets(compute_resident_mask(lg, k=2, evict="lru"))
    lrd = _sets(compute_resident_mask(lg, k=2, evict="lrd"))
    minl = _sets(compute_resident_mask(lg, k=2, evict="min_logit"))
    assert lru[0] == lrd[0] == frozenset({0, 1})
    # `lru` counts admission age only, so it drops e1 — the expert this very token wants most.
    assert lru[1] == frozenset({0, 2})
    # `lrd` saw e1 in the token's top-k, refreshed it, and drops the stale e0 instead. Here that
    # agrees with the quality-greedy policy, which is the point: refresh-on-demand recovers most
    # of what admission-order recency throws away.
    assert lrd[1] == frozenset({1, 2}) == minl[1]


def test_ema_gamma_interpolates_toward_the_instantaneous_kernel():
    """The width knob is real: gamma -> 1 makes the eviction key the current token's demand alone
    (a width-1 kernel), which is a different policy from a long-memory gamma."""
    lg = _ar1_logits_t(S=160, B=2, E=16, seed=35)
    k = 4
    wide = compute_resident_mask(lg, k, "ema", evict_gamma=0.03125)
    narrow = compute_resident_mask(lg, k, "ema", evict_gamma=0.999)
    assert not torch.equal(wide, narrow)


def test_unknown_evict_policy_raises():
    lg = _ar1_logits_t(S=8, B=1, E=8, seed=36)
    with pytest.raises(AssertionError):
        compute_resident_mask(lg, 3, "belady")


def test_accel_never_silently_downgrades_a_reference_only_policy():
    """compute_resident_mask_accel must route `lrd`/`ema` to the reference. The fast paths take a
    single use_lru boolean, so an unguarded call would run min_logit under an lrd label."""
    lg = _ar1_logits_t(S=60, B=2, E=16, seed=37)
    k = 4
    for evict in ("lrd", "ema"):
        acc = tr.compute_resident_mask_accel(lg, k, evict=evict)
        assert torch.equal(acc, compute_resident_mask(lg, k, evict))
        assert not torch.equal(acc, compute_resident_mask(lg, k, "min_logit"))
    tr._scan_path = None


def test_evict_gamma_banner():
    for var in ("TEMPORAL_EVICT", "TEMPORAL_EVICT_GAMMA"):
        os.environ.pop(var, None)
    assert "evict_gamma" not in banner_knobs()
    os.environ["TEMPORAL_EVICT"] = "ema"
    os.environ["TEMPORAL_EVICT_GAMMA"] = "0.0625"
    try:
        assert "evict_gamma=0.0625" in banner_knobs()
    finally:
        for var in ("TEMPORAL_EVICT", "TEMPORAL_EVICT_GAMMA"):
            os.environ.pop(var, None)


def test_new_evict_policies_match_probe_replay():
    """The router and the offline replay harness must implement the SAME eviction kernels — the
    replay is what produces the published policy tables, so a divergence would silently make the
    two halves of any eviction finding incomparable."""
    pytest.importorskip("numpy")
    import numpy as np
    try:
        import probe_replay as pr
    except Exception:
        pytest.skip("probe_replay (plot_probe deps) unavailable in this environment")
    lg = _ar1_logits_t(S=200, B=3, E=16, seed=41)
    lg_np = lg.numpy().astype(np.float32)
    k = 4
    for evict in ("lru", "min_logit", "lrd", "ema"):
        m = compute_resident_mask(lg, k, evict).numpy()
        swaps_router = (m[1:] != m[:-1]).any(-1)
        swaps_replay = pr.replay(lg_np, k, evict=evict)["swaps"][1:]
        assert (swaps_router == swaps_replay).mean() > 0.99, f"swap streams differ for {evict}"
