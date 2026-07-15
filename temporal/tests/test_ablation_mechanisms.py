#!/usr/bin/env python3
"""TDD specs for the EXPERIMENTAL ablation mechanisms of the temporal MoE router.

Pure-function tests for the default-off, negative-result scoring/loss knobs living in
temporal/ablation_mechanisms.py: aux-free (sigmoid+bias) trigger, demand momentum (Karen A3 /
double-momentum A3q / logratio LG1), K2 gate momentum, the decoupled stop-grad nomination head
(H1/H2/H3), the anticipatory (Track B) loss, the bursty-window (LG2) loss, and the temporal-
coherence BCE loss (the latter absorbed from the former scripts/phase0/test_coherence_loss.py).

No Megatron, no GPU — CPU torch only.
Run: .venv/bin/python -m pytest scripts/phase0/temporal/tests/test_ablation_mechanisms.py
"""
import os, sys
import torch
import pytest

# scripts/phase0 on path -> import the `temporal` package and (for cross-checks) probe_replay.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal.ablation_mechanisms import (auxfree_trigger_scores,
                                          anticipatory_target, anticipatory_bce_loss,
                                          momentum_shaped_scores, bursty_window_loss,
                                          nomination_head_logits, head_trigger_bonus,
                                          head_selection_active, head_centered_bonus,
                                          gate_momentum_scores, centered_demand_labels,
                                          coherence_bce_loss)
from temporal.temporal_router import compute_resident_mask, banner_knobs


def _ar1_logits_t(S=160, B=3, E=16, seed=11):
    torch.manual_seed(seed)
    base = torch.randn(B, E)
    lg = torch.empty(S, B, E)
    lg[0] = base
    for t in range(1, S):
        base = 0.95 * base + 0.3 * torch.randn(B, E)
        lg[t] = base
    return lg


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


# ---------------------------------------------------------------------------
# Temporal-coherence auxiliary loss (BCE) — absorbed from test_coherence_loss.py.
#
# Semantics under test:
#   loss = BCE(sigmoid(logits), target), target = final resident/used set (multi-hot, detached).
#   Independent per-expert sigmoids (NOT softmax) -> set-membership pull, no distribution clone.
# ---------------------------------------------------------------------------
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
