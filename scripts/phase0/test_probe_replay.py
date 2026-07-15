#!/usr/bin/env python3
"""Unit tests for the probe-replay policies (scripts/phase0/probe_replay.py).

Pure-numpy, no logs, no GPU. Run:
  .venv/bin/python -m pytest scripts/phase0/test_probe_replay.py

Covers the replay semantics that back experiments E4/E5/E7:
  - Belady (offline-optimal eviction) on a hand-computable 5-token example, and its coverage-bound
    property (Belady set-coverage >= shipped min_logit, since offline-optimal cannot be worse).
  - EMA beta=1.0 identity (E7 harness sanity check): smoothing with beta=1 is a no-op, so the
    replayed swap/coverage stream must be bit-identical to the un-smoothed baseline.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_replay as pr


# ---------------------------------------------------------------------------
# Belady — hand-computable 5-token example (E=4 experts, k=K=2, one sequence).
#
# t0 [9,8,1,0] -> cold fill R0 = top-2 = {0,1}
# t1 [1,2,9,0] -> demand {1,2}; expert2 non-resident beats worst resident (expert0) -> swap in 2.
#                 min_logit evicts expert0 (lowest logit=1).
#                 Belady evicts the resident whose NEXT demand is farthest: expert0 next demanded
#                 at t2, expert1 not until t4 -> Belady evicts expert1.
# t2 [9,0,8,1] -> demand {0,2}: min_logit (resident {1,2}) misses expert0; Belady ({0,2}) hits both.
# t3 [9,1,8,2] -> demand {0,2}
# t4 [1,9,2,8] -> demand {1,3}
# ---------------------------------------------------------------------------
def _hand_logits():
    lg = np.array([[9, 8, 1, 0],
                   [1, 2, 9, 0],
                   [9, 0, 8, 1],
                   [9, 1, 8, 2],
                   [1, 9, 2, 8]], np.float32)
    return lg[:, None, :]                      # [S=5, B=1, E=4]


def test_belady_evicts_farthest_next_use_and_diverges_from_min_logit():
    lg = _hand_logits()
    ml = pr.replay(lg, k=2, evict="min_logit", record_swaps=True)
    be = pr.replay(lg, k=2, evict="belady", record_swaps=True)
    # both swap in expert 2 at t1
    assert ml["nominee"][1, 0] == 2 and be["nominee"][1, 0] == 2
    # min_logit evicts the lowest-logit resident (expert 0); Belady evicts the farthest-future
    # resident (expert 1). This is the crux of the hand example.
    assert ml["evicted"][1, 0] == 0
    assert be["evicted"][1, 0] == 1
    assert ml["evicted"][1, 0] != be["evicted"][1, 0]


def test_belady_hand_coverage_values():
    # Pre-swap set-coverage per token (t=0 is the cold fill = 1.0 by definition).
    lg = _hand_logits()
    ml = pr.replay(lg, k=2, evict="min_logit")
    be = pr.replay(lg, k=2, evict="belady")
    # t2 demand {0,2}: min_logit resident on entry {1,2} -> 1/2; Belady resident {0,2} -> 2/2.
    assert ml["setcov"][2, 0] == pytest.approx(0.5)
    assert be["setcov"][2, 0] == pytest.approx(1.0)
    # Belady's total coverage must be >= min_logit's (offline-optimal eviction cannot be worse).
    assert be["setcov"].mean() >= ml["setcov"].mean()


def test_belady_never_worse_than_min_logit_random():
    # Property check on random streams: Belady set-coverage >= min_logit at K=k, cap-1.
    rng = np.random.default_rng(0)
    for _ in range(20):
        lg = rng.standard_normal((60, 3, 10)).astype(np.float32)
        ml = pr.replay(lg, k=3, evict="min_logit")["setcov"].mean()
        be = pr.replay(lg, k=3, evict="belady")["setcov"].mean()
        assert be >= ml - 1e-6


# ---------------------------------------------------------------------------
# EMA beta=1.0 identity (E7 harness sanity check).
# ---------------------------------------------------------------------------
def test_ema_beta_one_is_identity():
    rng = np.random.default_rng(1)
    lg = rng.standard_normal((100, 2, 8)).astype(np.float32)
    assert np.array_equal(pr._ema(lg, 1.0), lg)          # smoothing with beta=1 is a no-op


def test_ema_beta_one_replay_matches_baseline_exactly():
    rng = np.random.default_rng(2)
    lg = rng.standard_normal((200, 4, 12)).astype(np.float32)
    base = pr.replay(lg, k=4, evict="min_logit")
    ema1 = pr.replay(pr._ema(lg, 1.0), k=4, evict="min_logit")
    assert np.array_equal(base["swaps"], ema1["swaps"])
    assert np.array_equal(base["setcov"], ema1["setcov"])
    assert np.array_equal(base["masscov"], ema1["masscov"])


def test_ema_smoothing_reduces_swaps():
    # Slow-feature preview: smoothing the logit stream should not increase the swap rate.
    rng = np.random.default_rng(3)
    lg = rng.standard_normal((300, 2, 16)).astype(np.float32)
    sr1 = pr.replay(lg, k=4, evict="min_logit")["swaps"][1:].mean()
    sr_smooth = pr.replay(pr._ema(lg, 0.1), k=4, evict="min_logit")["swaps"][1:].mean()
    assert sr_smooth <= sr1


# ---------------------------------------------------------------------------
# Replay invariants shared with the shipped policy.
# ---------------------------------------------------------------------------
def test_tau_monotonically_reduces_swaps():
    rng = np.random.default_rng(4)
    lg = rng.standard_normal((300, 2, 12)).astype(np.float32)
    prev = 1.01
    for tau in (0.0, 0.5, 1.0, 2.0, 4.0):
        sr = pr.replay(lg, k=3, evict="min_logit", tau=tau)["swaps"][1:].mean()
        assert sr <= prev + 1e-9
        prev = sr


def test_coverage_in_unit_interval_and_cold_fill_full():
    rng = np.random.default_rng(5)
    lg = rng.standard_normal((50, 3, 9)).astype(np.float32)
    for evict in ("min_logit", "lru", "belady"):
        out = pr.replay(lg, k=3, evict=evict)
        assert (out["setcov"] >= -1e-6).all() and (out["setcov"] <= 1 + 1e-6).all()
        assert np.allclose(out["setcov"][0], 1.0)        # cold fill serves its own top-k


# ---------------------------------------------------------------------------
# A1/A2 additions: raw-demand scoring (eval_lg) + additive-momentum trigger.
# ---------------------------------------------------------------------------
def _ar1_logits(S=200, B=4, E=24, seed=7):
    """Correlated (AR(1)) logits so smoothing/momentum have real temporal structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(B, E)).astype(np.float32)
    lg = np.empty((S, B, E), np.float32)
    lg[0] = base
    for t in range(1, S):
        base = 0.95 * base + 0.3 * rng.normal(size=(B, E)).astype(np.float32)
        lg[t] = base
    return lg


def test_eval_lg_identity_when_trigger_is_raw():
    """beta=1 (no smoothing), tau=0, eval_lg=raw must be bit-identical to the plain baseline."""
    lg = _ar1_logits()
    b0 = pr.replay(lg, k=6, evict="min_logit")
    b1 = pr.replay(pr._ema(lg, 1.0), k=6, evict="min_logit", tau=0.0, eval_lg=lg)
    for key in ("setcov", "masscov", "swaps"):
        assert np.array_equal(b0[key], b1[key])


def test_eval_lg_raw_scoring_leq_self_scoring_and_swaps_invariant():
    """A shaped trigger scores itself optimistically; raw-demand scoring must not exceed it,
    and the swap stream (a pure function of the trigger) must be identical under either scoring."""
    lg = _ar1_logits()
    sm = pr._ema(lg, 0.1)
    self_scored = pr.replay(sm, k=6, evict="min_logit")
    raw_scored = pr.replay(sm, k=6, evict="min_logit", eval_lg=lg)
    assert raw_scored["setcov"][1:].mean() <= self_scored["setcov"][1:].mean() + 1e-6
    assert np.array_equal(self_scored["swaps"], raw_scored["swaps"])


def test_momentum_scores_causal_and_coldfill_matches_raw():
    """_momentum_scores must be causal (future logits cannot affect past scores) and its t=0
    top-k must equal the raw top-k (softmax is monotone; no bonus at t=0)."""
    lg = _ar1_logits()
    sc_a = pr._momentum_scores(lg, beta_m=1.0, gamma_m=0.125)
    lg2 = lg.copy(); lg2[150:] += 10.0
    sc_b = pr._momentum_scores(lg2, beta_m=1.0, gamma_m=0.125)
    assert np.allclose(sc_a[:150], sc_b[:150])
    assert np.isfinite(sc_a).all()
    raw_top0 = np.argsort(-lg[0], 1)[:, :6]
    mom_top0 = np.argsort(-sc_a[0], 1)[:, :6]
    assert np.array_equal(np.sort(raw_top0, 1), np.sort(mom_top0, 1))


def test_momentum_end_to_end_sane_and_smoothing_cuts_swaps():
    """Momentum-shaped replay stays in range; EMA smoothing on correlated logits cuts swap rate."""
    lg = _ar1_logits()
    out = pr.replay(pr._momentum_scores(lg, 1.0, 0.125), k=6, evict="min_logit", eval_lg=lg)
    for key in ("setcov", "masscov"):
        assert (out[key] >= -1e-6).all() and (out[key] <= 1 + 1e-6).all()
    base_sr = pr.replay(lg, k=6, evict="min_logit")["swaps"][1:].mean()
    smooth_sr = pr.replay(pr._ema(lg, 0.1), k=6, evict="min_logit")["swaps"][1:].mean()
    assert smooth_sr < base_sr
