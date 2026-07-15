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
from temporal_router import (compute_resident_mask, auxfree_trigger_scores,
                             anticipatory_target, anticipatory_bce_loss,
                             momentum_shaped_scores, bursty_window_loss,
                             nomination_head_logits, head_trigger_bonus,
                             head_selection_active, banner_knobs,
                             head_centered_bonus, gate_momentum_scores,
                             centered_demand_labels)


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


# ---------------------------------------------------------------------------
# Trigger-shaping knobs (tau hysteresis + EMA smoothing) — B1 eval-time path.
# ---------------------------------------------------------------------------
def _ar1_logits_t(S=160, B=3, E=16, seed=11):
    torch.manual_seed(seed)
    base = torch.randn(B, E)
    lg = torch.empty(S, B, E)
    lg[0] = base
    for t in range(1, S):
        base = 0.95 * base + 0.3 * torch.randn(B, E)
        lg[t] = base
    return lg


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
# Aux-free (DeepSeek-V3) trigger basis — alignment program Track A.
# ---------------------------------------------------------------------------
def test_auxfree_trigger_zero_bias_is_identity():
    """With expert_bias == 0, sigmoid is elementwise monotone, so the induced resident mask must
    be identical to the raw-logit mask (all per-token comparisons preserved)."""
    lg = _ar1_logits_t(seed=17)
    zero = torch.zeros(lg.shape[-1])
    trig = auxfree_trigger_scores(lg, zero).to(lg.dtype)
    a = compute_resident_mask(lg, 4, "min_logit")
    b = compute_resident_mask(trig, 4, "min_logit")
    assert torch.equal(a, b)


def test_auxfree_trigger_bias_shifts_selection():
    """A strongly positive bias on one expert must pull it into residency more often; a strongly
    negative bias must push it out (the controller's lever actually reaches the trigger)."""
    lg = _ar1_logits_t(seed=18)
    E = lg.shape[-1]
    base = compute_resident_mask(auxfree_trigger_scores(lg, torch.zeros(E)).to(lg.dtype), 4, "min_logit")
    up = torch.zeros(E); up[3] = 5.0
    down = torch.zeros(E); down[3] = -5.0
    m_up = compute_resident_mask(auxfree_trigger_scores(lg, up).to(lg.dtype), 4, "min_logit")
    m_dn = compute_resident_mask(auxfree_trigger_scores(lg, down).to(lg.dtype), 4, "min_logit")
    assert m_up[..., 3].float().mean() > base[..., 3].float().mean()
    assert m_dn[..., 3].float().mean() < base[..., 3].float().mean()


# ---------------------------------------------------------------------------
# Anticipatory loss (alignment program Track B) — discounted future-demand target.
# ---------------------------------------------------------------------------
def test_anticipatory_target_hand_example():
    """S=3, E=3, k=1, gamma=0.5. Demands: t0->e0, t1->e1, t2->e2.
    y2 = m2; y1 = m1 + .5*y2; y0 = m0 + .5*y1. target = .5*y."""
    lg = torch.full((3, 1, 3), -10.0)
    lg[0, 0, 0] = 10; lg[1, 0, 1] = 10; lg[2, 0, 2] = 10
    tgt, valid = anticipatory_target(lg, k=1, gamma=0.5)
    exp = torch.tensor([  # (1-gamma)*y
        [0.5, 0.25, 0.125],
        [0.0, 0.5, 0.25],
        [0.0, 0.0, 0.5],
    ])
    assert torch.allclose(tgt[:, 0, :], exp, atol=1e-6)
    assert valid[0] and not valid[1] and not valid[2]   # tail = 1/(1-0.5) = 2 -> last two masked


def test_anticipatory_tail_mask():
    lg = torch.randn(20, 2, 8)
    tgt, valid = anticipatory_target(lg, k=2, gamma=0.5)
    assert valid[:18].all() and not valid[18:].any()      # 1/(1-0.5)=2 tail positions masked
    tgt9, valid9 = anticipatory_target(lg, k=2, gamma=0.9)
    assert valid9[:10].all() and not valid9[10:].any()    # 1/(1-0.9)=10 tail positions masked
    assert (tgt >= 0).all() and (tgt <= 1 + 1e-6).all()
    assert (tgt9 >= 0).all() and (tgt9 <= 1 + 1e-6).all()


def test_anticipatory_target_uses_only_present_and_future():
    lg = torch.randn(50, 2, 8)
    lg2 = lg.clone(); lg2[:20] += 100.0                   # perturb the PAST
    t1, _ = anticipatory_target(lg, 2, 0.5)
    t2, _ = anticipatory_target(lg2, 2, 0.5)
    assert torch.allclose(t1[20:], t2[20:])               # future-of-perturbation unchanged


def test_anticipatory_loss_direction():
    """Loss must be lower when logits already rank the future-demanded experts highly."""
    lg = torch.randn(30, 2, 8)
    tgt, valid = anticipatory_target(lg, 2, 0.5)
    aligned = (tgt * 8 - 4)                               # logits proportional to target
    shuffled = aligned.flip(-1)
    l_aligned = anticipatory_bce_loss(aligned, tgt, valid)
    l_shuffled = anticipatory_bce_loss(shuffled, tgt, valid)
    assert l_aligned < l_shuffled


# ---------------------------------------------------------------------------
# Demand-momentum trigger shaping (Track A rung A3 — Karen's formulation).
# ---------------------------------------------------------------------------
def test_momentum_zero_beta_identity_and_coldfill():
    lg = _ar1_logits_t(seed=21)
    probs = torch.softmax(lg.float(), -1)
    shaped = momentum_shaped_scores(lg, probs, beta_m=0.0, gamma_m=0.125)
    assert torch.allclose(shaped, lg, atol=1e-5)          # beta=0 is a no-op
    shaped2 = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.125)
    assert torch.allclose(shaped2[0], lg[0], atol=1e-5)   # no bonus at t=0 (cold fill preserved)


def test_momentum_causal_and_matches_replay_semantics():
    """Perturbing future logits must not change past shaped scores; and with base == probs the
    torch implementation must match probe_replay._momentum_scores numerically."""
    lg = _ar1_logits_t(seed=22)
    probs = torch.softmax(lg.float(), -1)
    a = momentum_shaped_scores(lg, probs, 1.0, 0.125)
    lg2 = lg.clone(); lg2[100:] += 10.0
    probs2 = torch.softmax(lg2.float(), -1)
    b = momentum_shaped_scores(lg2, probs2, 1.0, 0.125)
    assert torch.allclose(a[:100], b[:100], atol=1e-5)
    pytest.importorskip("numpy")
    import numpy as np
    try:
        import probe_replay as pr
    except Exception:
        pytest.skip("probe_replay deps unavailable")
    ours = momentum_shaped_scores(probs, probs, 1.0, 0.125).numpy()
    theirs = pr._momentum_scores(lg.numpy().astype(np.float32), 1.0, 0.125)
    assert np.allclose(ours, theirs, atol=1e-4)


def test_momentum_favors_persistent_demand():
    """An expert with steady moderate demand should outscore a one-shot spike under momentum."""
    S, E = 60, 8
    lg = torch.zeros(S, 1, E)
    lg[:, 0, 0] = 2.0          # persistent expert 0
    lg[30, 0, 1] = 2.5         # one-token spike on expert 1
    probs = torch.softmax(lg.float(), -1)
    shaped = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.25)
    assert shaped[31, 0, 0] > shaped[31, 0, 1]            # right after the spike, persistence wins


def test_double_momentum_alpha_zero_is_single_momentum():
    """alpha_m=0 must reproduce the original single-momentum rung exactly (regression guard)."""
    lg = _ar1_logits_t(seed=23)
    probs = torch.softmax(lg.float(), -1)
    single = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.125)
    double0 = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.125,
                                     alpha_m=0.0, gamma_q=0.015625)
    assert torch.allclose(single, double0, atol=1e-6)


def test_double_momentum_chronic_demand_cancels():
    """Karen's full formulation (alpha=beta): a chronically hot expert's bonus must decay toward
    zero (M and Q converge), while a NEWLY hot expert right after onset keeps a large positive
    bonus (M rises fast, Q lags). Chronic expert 0 hot from t=0; expert 1 turns hot at t=200."""
    S, E = 400, 8
    lg = torch.zeros(S, 1, E)
    lg[:, 0, 0] = 3.0            # chronic
    lg[200:, 0, 1] = 3.0         # newly hot at t=200
    probs = torch.softmax(lg.float(), -1)
    shaped = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.125,
                                    alpha_m=1.0, gamma_q=0.015625)
    bonus = (shaped.float() - lg.float())[:, 0, :]
    # shortly after onset: fresh expert's bonus dwarfs the chronic expert's
    assert bonus[215, 1] > bonus[215, 0] + 0.05
    # chronic expert late in its steady state: bonus ~0 (permanence is not rewarded)
    assert abs(bonus[199, 0].item()) < 0.02
    # cold fill: no bonus at t=0
    assert torch.allclose(shaped[0], lg[0], atol=1e-6)


def test_double_momentum_demotes_recently_cooled():
    """A formerly-hot expert that just went cold should get a NEGATIVE bonus (M decays fast,
    slow Q remembers) — stale residents are actively demoted, not just no longer boosted."""
    S, E = 400, 8
    lg = torch.zeros(S, 1, E)
    lg[:200, 0, 0] = 3.0         # hot for 200 tokens, then cold
    probs = torch.softmax(lg.float(), -1)
    shaped = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.125,
                                    alpha_m=1.0, gamma_q=0.015625)
    bonus = (shaped.float() - lg.float())[:, 0, :]
    assert bonus[240, 0] < -0.02


def test_logratio_momentum_burst_vs_chronic_vs_cooled():
    """LG1 log-ratio momentum: a chronically hot expert's bonus ~0; a rarely-used expert in a
    fresh burst gets a large POSITIVE bonus; a just-cooled expert goes negative; beta=0 no-op."""
    S, E = 400, 8
    lg = torch.zeros(S, 1, E)
    lg[:, 0, 0] = 3.0            # chronic expert 0
    lg[200:, 0, 1] = 3.0         # fresh burst on expert 1 at t=200
    lg[:150, 0, 2] = 3.0         # expert 2 hot early, cooled at t=150
    probs = torch.softmax(lg.float(), -1)
    shaped = momentum_shaped_scores(lg, probs, beta_m=1.0, gamma_m=0.125,
                                    gamma_q=0.015625, mode="logratio")
    bonus = (shaped.float() - lg.float())[:, 0, :]
    # t=149: expert 0's demand has been constant since t=0 -> M==Q -> bonus cancels exactly.
    # (After t=150 expert 2 cools, expert 0's demand share RISES — a genuine mini-burst — so its
    # bonus is legitimately positive for a while; chronic-cancellation is a steady-state claim.)
    assert abs(bonus[149, 0].item()) < 0.05                 # chronic ~ cancels
    assert bonus[215, 1] > bonus[215, 0] + 0.3              # fresh burst >> (near-)chronic
    assert bonus[215, 1] > 0.3                              # and large positive in logit units
    assert bonus[199, 2] < -0.3                             # cooled -> demoted
    ident = momentum_shaped_scores(lg, probs, beta_m=0.0, gamma_m=0.125, mode="logratio")
    assert torch.allclose(ident, lg, atol=1e-5)
    assert torch.allclose(shaped[0], lg[0], atol=1e-6)      # cold fill preserved


def test_logratio_momentum_matches_replay_mirror():
    """torch logratio implementation must match probe_replay._momentum_scores(mode='logratio')."""
    lg = _ar1_logits_t(seed=25)
    probs = torch.softmax(lg.float(), -1)
    ours = momentum_shaped_scores(lg, probs, 1.0, 0.125, gamma_q=0.015625, mode="logratio")
    pytest.importorskip("numpy")
    import numpy as np
    try:
        import probe_replay as pr
    except Exception:
        pytest.skip("probe_replay deps unavailable")
    theirs = pr._momentum_scores(lg.numpy().astype(np.float32), 1.0, 0.125,
                                 gamma_q=0.015625, mode="logratio")
    assert np.allclose(ours.numpy(), theirs, atol=1e-4)


def test_double_momentum_causal():
    """Perturbing future logits must not change past shaped scores through the Q path either."""
    lg = _ar1_logits_t(seed=24)
    probs = torch.softmax(lg.float(), -1)
    a = momentum_shaped_scores(lg, probs, 1.0, 0.125, alpha_m=1.0, gamma_q=0.015625)
    lg2 = lg.clone(); lg2[100:] += 10.0
    probs2 = torch.softmax(lg2.float(), -1)
    b = momentum_shaped_scores(lg2, probs2, 1.0, 0.125, alpha_m=1.0, gamma_q=0.015625)
    assert torch.allclose(a[:100], b[:100], atol=1e-5)


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
# H3 — centered-target head labels (popularity unlearnable by construction).
# ---------------------------------------------------------------------------
def test_centered_labels_popularity_unlearnable():
    """A CONSTANTLY-hot expert (target == its own baseline) and a dead expert must both get
    ~all-zero labels — static popularity carries no positive signal; only above-own-baseline
    transients are positive class."""
    S, B, E = 300, 1, 4
    tgt = torch.zeros(S, B, E)
    tgt[:, 0, 0] = 0.8                                      # chronically hot expert 0
    tgt[100:110, 0, 1] = 0.6                                # burst on expert 1 at t=100
    labels = centered_demand_labels(tgt, gamma_c=0.015625)
    assert labels[:, 0, 0].sum().item() == 0.0              # chronic: never above own baseline
    assert labels[:, 0, 2].sum().item() == 0.0              # dead: never positive
    assert labels[100:110, 0, 1].mean().item() > 0.9        # burst onset IS the positive class
    assert labels[150:, 0, 1].sum().item() == 0.0           # after burst: back to zero


def test_centered_labels_causal_and_balanced_across_popularity():
    """Future targets must not affect past labels; and experts with very different STATIC demand
    levels must get similar positive-label base rates on fluctuating streams (the classifier
    cannot sort experts by popularity from the labels)."""
    torch.manual_seed(61)
    S, B, E = 600, 1, 4
    tgt = torch.rand(S, B, E) * 0.2
    tgt[:, 0, 0] += 0.7                                     # popular expert, same fluctuations
    labels = centered_demand_labels(tgt, gamma_c=0.015625)
    rates = labels[50:, 0, :].mean(dim=0)
    assert (rates.max() - rates.min()).item() < 0.1         # base rates ~equal despite popularity
    tgt2 = tgt.clone(); tgt2[400:] += 0.5
    l2 = centered_demand_labels(tgt2, gamma_c=0.015625)
    assert torch.equal(labels[:400], l2[:400])              # causal


def test_centered_labels_banner():
    for k in ("HEAD_LAMBDA", "HEAD_BETA", "HEAD_TARGET_CENTER", "HEAD_CENTER"):
        os.environ.pop(k, None)
    os.environ["HEAD_LAMBDA"] = "1.0"; os.environ["HEAD_TARGET_CENTER"] = "1"
    try:
        assert "target_center=1, gamma_c=0.015625" in banner_knobs()
    finally:
        for k in ("HEAD_LAMBDA", "HEAD_TARGET_CENTER"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# K2 — gate momentum (fast activator-inhibitor dynamics in the router scores).
# ---------------------------------------------------------------------------
def test_gate_momentum_identity_and_coldfill():
    """beta=0 (and beta<0) must return logits unchanged; t=0 must carry no bonus."""
    torch.manual_seed(51)
    lg = torch.randn(40, 2, 8)
    assert gate_momentum_scores(lg, 0.0) is lg
    shaped = gate_momentum_scores(lg, 1.0)
    assert torch.allclose(shaped[0], lg[0], atol=1e-6)     # cold fill preserved


def test_gate_momentum_self_excitation_persistence():
    """The activator: a one-token spike on an otherwise-flat stream must leave the spiked expert
    preferred for several FOLLOWING tokens (shaped argmax persists ~1/gamma_m) — this is the
    burst-forming dynamic the trigger-side momentum could not create."""
    S, B, E = 60, 1, 8
    lg = torch.zeros(S, B, E)
    lg[5, 0, 3] = 3.0                                       # single-token spike on expert 3
    shaped = gate_momentum_scores(lg, beta=1.0, gamma_m=0.125, gamma_q=0.015625)
    top = shaped[:, 0, :].argmax(-1)
    assert (top[6:12] == 3).all()                           # spike persists as a run afterwards
    raw_top = lg[:, 0, :].argmax(-1)
    assert not (raw_top[6:12] == 3).all()                   # ...which raw logits do NOT have


def test_gate_momentum_burst_self_extinguishes():
    """The inhibitor: when demand for one expert TURNS ON and stays on, the bonus must peak
    shortly after onset and decay as slow Q catches up (M/Q -> 1, ln -> 0) — sustained demand is
    not chronically rewarded, so a burst hands off rather than pins. And an expert hot from t=0
    (M0 == Q0) gets NO cold-start kick at all."""
    S, B, E = 500, 1, 8
    lg = torch.zeros(S, B, E)
    lg[100:, 0, 0] = 4.0                                    # expert 0 turns hot at t=100, stays
    lg[:, 0, 1] = 4.0                                       # expert 1 hot from t=0 (M0=Q0)
    shaped = gate_momentum_scores(lg, beta=1.0, gamma_m=0.125, gamma_q=0.015625)
    bonus = (shaped - lg)[:, 0, :]
    early = bonus[112, 0].item()                            # ~1/gamma_m after onset
    late = bonus[499, 0].item()
    assert early > 0.5                                      # activator fires after onset
    assert late < 0.25 * early                              # inhibitor extinguishes the run
    assert bonus[:99, 1].abs().max() < 0.05                 # from-t0 hot: no cold-start kick
    assert abs(bonus[499, 1].item()) < 0.1                  # ~zero chronic bonus (no pinning)
    # (e1 DOES get a negative transient at t=100 — its demand share genuinely fell; correct.)


def test_gate_momentum_causal_and_zero_dc():
    """Future logits must not affect past shaped scores; and on a symmetric rotating-burst stream
    (every expert takes equal turns being hot) the per-expert time-average of the bonus must be
    ~0 and ~equal across experts — nothing for gradient descent to rectify into popularity."""
    lg = _ar1_logits_t(S=400, seed=52)
    a = gate_momentum_scores(lg, 1.0)
    lg2 = lg.clone(); lg2[300:] += 10.0
    b = gate_momentum_scores(lg2, 1.0)
    assert torch.allclose(a[:300], b[:300], atol=1e-5)
    E2, RUN = 8, 50
    S2 = RUN * E2 * 4                                       # 4 full rotations of 8 experts
    rot = torch.zeros(S2, 1, E2)
    for w in range(S2 // RUN):
        rot[w * RUN:(w + 1) * RUN, 0, w % E2] = 4.0
    shaped = gate_momentum_scores(rot, 1.0, gamma_m=0.125, gamma_q=0.015625)
    bonus = (shaped - rot).float()[:, 0, :]
    dc = bonus[RUN * E2:].mean(dim=0)                       # per-expert DC, init rotation skipped
    # The rectification-relevant quantity is the RELATIVE DC across experts (what W could
    # integrate into popularity): it must vanish. A COMMON shift (Jensen asymmetry of the ln,
    # here ~-0.2 for all experts equally) is popularity-neutral by symmetry.
    assert (dc.max() - dc.min()).item() < 0.02              # zero RELATIVE DC — nothing to rectify
    assert dc.mean().abs() < 0.5                            # common shift bounded
    assert bonus.abs().max() > 0.5                          # while the TRANSIENT bonus is large


def test_gate_momentum_grad_reaches_logits_identically():
    """The bonus is detached state: gradients w.r.t. the shaped logits must flow to the raw
    logits EXACTLY as identity (no gradient path through M/Q), so W trains through the shift."""
    lg = torch.randn(30, 2, 8, requires_grad=True)
    shaped = gate_momentum_scores(lg, 1.0)
    shaped.sum().backward()
    assert torch.allclose(lg.grad, torch.ones_like(lg))     # d(shaped)/d(logits) == I


def test_gate_momentum_banner():
    for k in ("TEMPORAL_MOM_BETA", "TEMPORAL_MOM_APPLY", "TEMPORAL_MOM_MODE"):
        os.environ.pop(k, None)
    os.environ["TEMPORAL_MOM_BETA"] = "0.5"; os.environ["TEMPORAL_MOM_APPLY"] = "gates"
    try:
        s = banner_knobs()
        assert "momentum(beta=0.5, gamma=0.125, apply=gates, gamma_q=0.015625)" in s
    finally:
        for k in ("TEMPORAL_MOM_BETA", "TEMPORAL_MOM_APPLY"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Bursty window loss (LG2 — windowed concentration + untouched global balance).
# ---------------------------------------------------------------------------
def test_bursty_loss_prefers_windowed_concentration():
    """Rotating per-window bursts (one hot expert per window, different across windows) must
    score LOWER than a uniform stream — the loss rewards local concentration."""
    S, B, E, W = 128, 2, 8, 32
    uniform = torch.zeros(S, B, E)
    bursts = torch.zeros(S, B, E)
    for w in range(S // W):
        bursts[w * W:(w + 1) * W, :, w % E] = 6.0     # window w concentrated on expert w%E
    assert bursty_window_loss(bursts, W) < bursty_window_loss(uniform, W) - 0.5


def test_bursty_loss_indifferent_to_global_rotation():
    """The LOCAL term must not prefer global collapse over rotation: rotating bursts and a
    single-expert-always stream are both fully window-concentrated -> near-equal loss. Global
    diversity is enforced by the untouched standard aux, not (anti-)rewarded here."""
    S, B, E, W = 128, 1, 8, 32
    rotate = torch.zeros(S, B, E)
    fixed = torch.zeros(S, B, E)
    for w in range(S // W):
        rotate[w * W:(w + 1) * W, :, w % E] = 6.0
    fixed[:, :, 0] = 6.0
    a = bursty_window_loss(rotate, W).item()
    b = bursty_window_loss(fixed, W).item()
    assert abs(a - b) < 1e-4


# ---------------------------------------------------------------------------
# Decoupled stop-grad nomination head (local-global program, mechanism H1).
# ---------------------------------------------------------------------------
def test_head_stopgrad_trunk_and_router_grads_exactly_zero():
    """THE property that makes the head Goodhart-proof: backward of the head BCE loss must leave
    the trunk hidden state and the router weight with EXACTLY zero (absent) gradient — only the
    head weight learns. This is the same graph temporal_forward builds (target from raw router
    logits via anticipatory_target, head on detached hidden)."""
    import torch.nn.functional as F
    torch.manual_seed(31)
    S, B, E, D = 12, 2, 8, 16
    hidden = torch.randn(S, B, D, requires_grad=True)
    router_w = torch.randn(E, D, requires_grad=True)
    head_w = torch.randn(E, D, requires_grad=True)
    logits = F.linear(hidden, router_w)                   # trunk -> router logits (differentiable)
    head_lg = nomination_head_logits(hidden, head_w)      # trunk DETACHED inside
    tgt, valid = anticipatory_target(logits, k=2, gamma=0.5)
    loss = anticipatory_bce_loss(head_lg, tgt, valid)
    loss.backward()
    assert head_w.grad is not None and head_w.grad.abs().sum() > 0    # the head DOES learn
    assert hidden.grad is None or hidden.grad.abs().sum().item() == 0.0
    assert router_w.grad is None or router_w.grad.abs().sum().item() == 0.0


def test_head_target_correctness_vs_anticipatory_target():
    """The head's supervision must be EXACTLY anticipatory_target's output: a head whose sigmoid
    equals the target achieves the BCE floor (loss strictly below any perturbation), and the
    hand example's target values are hit."""
    torch.manual_seed(32)
    lg = torch.randn(30, 2, 8)
    tgt, valid = anticipatory_target(lg, k=2, gamma=0.5)
    ideal = torch.logit(tgt.clamp(1e-4, 1 - 1e-4))        # sigmoid(ideal) == target
    l_ideal = anticipatory_bce_loss(ideal, tgt, valid)
    for seed in range(3):
        torch.manual_seed(100 + seed)
        l_pert = anticipatory_bce_loss(ideal + torch.randn_like(ideal), tgt, valid)
        assert l_ideal < l_pert
    # a linear head trained by SGD on a fixed input must reduce this loss (trainability)
    S, B, E, D = 40, 2, 8, 16
    torch.manual_seed(33)
    hidden = torch.randn(S, B, D)
    rlogits = torch.randn(S, B, E)
    tgt2, valid2 = anticipatory_target(rlogits, k=2, gamma=0.5)
    w = torch.zeros(E, D, requires_grad=True)
    opt = torch.optim.SGD([w], lr=1.0)
    first = None
    for _ in range(50):
        loss = anticipatory_bce_loss(nomination_head_logits(hidden, w), tgt2, valid2)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first - 0.01


def test_head_zscore_bonus_not_uniform_and_selection_reaches_trigger():
    """The bonus must be a real per-expert signal, not a per-token-uniform shift: centered (~0
    mean per token), unit scale after z-scoring, finite/zero on a constant head output, beta=0
    no-op — and a strong head preference must actually change the resident mask."""
    torch.manual_seed(34)
    hl = torch.randn(20, 3, 8)
    b = head_trigger_bonus(hl, beta=1.0)
    assert b.shape == hl.shape
    assert torch.allclose(b.mean(-1), torch.zeros(20, 3), atol=1e-5)   # uniform part removed
    assert b.std(-1, unbiased=False).min() > 0.9                       # z-scored, NOT uniform
    u = head_trigger_bonus(torch.zeros(5, 2, 8), 1.0)                  # constant head output
    assert torch.isfinite(u).all() and u.abs().max() < 1e-3
    assert head_trigger_bonus(hl, 0.0).abs().max().item() == 0.0       # beta=0 no-op
    # selection actually shifts: head strongly favoring expert 5 pulls it into residency
    lg = _ar1_logits_t(seed=35, E=8)
    fav = torch.full((lg.shape[0], lg.shape[1], 8), -3.0); fav[..., 5] = 3.0
    base = compute_resident_mask(lg, 3, "min_logit")
    shaped = compute_resident_mask(lg + head_trigger_bonus(fav, 2.0).to(lg.dtype), 3, "min_logit")
    assert shaped[..., 5].float().mean() > base[..., 5].float().mean()


def test_head_warmup_gating():
    """Selection use activates at exactly warmup_frac * train_iters; unknown train_iters is
    fail-safe inactive; warmup 0 is immediately active."""
    assert not head_selection_active(0, 1000, 0.25)
    assert not head_selection_active(249, 1000, 0.25)
    assert head_selection_active(250, 1000, 0.25)
    assert head_selection_active(999, 1000, 0.25)
    assert not head_selection_active(10, 0, 0.25)         # train_iters unknown -> stay off
    assert head_selection_active(0, 1000, 0.0)


def test_head_banner():
    """install()'s banner must print head(lambda=…, beta=…, gamma=…, warmup=…) when the head is
    enabled and omit it when not."""
    for k in ("HEAD_LAMBDA", "HEAD_BETA", "HEAD_GAMMA", "HEAD_WARMUP_FRAC"):
        os.environ.pop(k, None)
    assert "head(" not in banner_knobs()
    os.environ["HEAD_LAMBDA"] = "1.0"; os.environ["HEAD_BETA"] = "1.0"
    try:
        assert "head(lambda=1.0, beta=1.0, gamma=0.5, warmup=0.25)" in banner_knobs()
        os.environ["HEAD_GAMMA"] = "0.9"; os.environ["HEAD_WARMUP_FRAC"] = "0.1"
        assert "head(lambda=1.0, beta=1.0, gamma=0.9, warmup=0.1)" in banner_knobs()
    finally:
        for k in ("HEAD_LAMBDA", "HEAD_BETA", "HEAD_GAMMA", "HEAD_WARMUP_FRAC"):
            os.environ.pop(k, None)


def test_head_centered_bonus_chronic_bias_cancels():
    """H2's whole point: an expert the head CHRONICALLY predicts hot must see its bonus decay to
    ~0 (zero time-average per expert), while the plain z-score bonus keeps rewarding it forever."""
    S, B, E = 600, 1, 8
    hl = torch.zeros(S, B, E)
    hl[:, 0, 0] = 4.0                                     # chronic head preference for expert 0
    plain = head_trigger_bonus(hl, 1.0)
    cent = head_centered_bonus(hl, 1.0, gamma_c=0.015625)
    assert plain[599, 0, 0] > 1.0                         # uncentered: standing bonus persists
    assert abs(cent[599, 0, 0].item()) < 0.15             # centered: chronic bonus ~ gone
    assert torch.allclose(cent[0], plain[0], atol=1e-6)   # t=0: C=0, identical to uncentered


def test_head_centered_bonus_transient_survives_and_cooled_demoted():
    """Transient anticipation (a fresh burst) must pass through near-full strength right after
    onset; a just-cooled chronic expert must go NEGATIVE (actively demoted, mirroring logratio)."""
    S, B, E = 600, 1, 8
    hl = torch.zeros(S, B, E)
    hl[300:, 0, 1] = 4.0                                  # burst onset at t=300 on expert 1
    hl[:300, 0, 2] = 4.0                                  # expert 2 hot early, cools at t=300
    cent = head_centered_bonus(hl, 1.0, gamma_c=0.015625)
    plain = head_trigger_bonus(hl, 1.0)
    assert cent[305, 0, 1] > 0.8 * plain[305, 0, 1]       # burst survives centering ~undamped
    assert cent[310, 0, 2] < -0.3                         # cooled chronic expert demoted
    # causality: perturbing the future must not change past bonuses
    hl2 = hl.clone(); hl2[400:] += 3.0
    a = head_centered_bonus(hl, 1.0); b = head_centered_bonus(hl2, 1.0)
    assert torch.allclose(a[:400], b[:400], atol=1e-6)


def test_head_centered_bonus_beta_and_banner():
    """beta scales linearly / beta=0 is a no-op; banner gains center=1, gamma_c when HEAD_CENTER=1."""
    torch.manual_seed(41)
    hl = torch.randn(30, 2, 8)
    b1 = head_centered_bonus(hl, 1.0)
    b2 = head_centered_bonus(hl, 2.0)
    assert torch.allclose(b2, 2.0 * b1, atol=1e-5)
    assert head_centered_bonus(hl, 0.0).abs().max().item() == 0.0
    for k in ("HEAD_LAMBDA", "HEAD_BETA", "HEAD_CENTER", "HEAD_GAMMA_C", "HEAD_FORCE_ACTIVE"):
        os.environ.pop(k, None)
    os.environ["HEAD_LAMBDA"] = "1.0"; os.environ["HEAD_BETA"] = "1.0"
    os.environ["HEAD_CENTER"] = "1"
    try:
        s = banner_knobs()
        assert "center=1, gamma_c=0.015625" in s and "head(lambda=1.0" in s
        os.environ["HEAD_FORCE_ACTIVE"] = "1"
        assert "force_active=1" in banner_knobs()
    finally:
        for k in ("HEAD_LAMBDA", "HEAD_BETA", "HEAD_CENTER", "HEAD_GAMMA_C", "HEAD_FORCE_ACTIVE"):
            os.environ.pop(k, None)


def test_bursty_loss_tail_and_grad():
    """Tail tokens beyond the last full window are dropped (S not a multiple of W works; S < W
    returns a connected zero), and gradients flow to the logits."""
    lg = torch.randn(70, 2, 8, requires_grad=True)
    loss = bursty_window_loss(lg, 32)                  # 2 full windows, 6-token tail dropped
    loss.backward()
    assert lg.grad is not None and lg.grad[:64].abs().sum() > 0
    assert lg.grad[64:].abs().sum() == 0               # tail contributes nothing
    tiny = bursty_window_loss(torch.randn(10, 1, 8, requires_grad=True), 32)
    assert tiny.item() == 0.0
