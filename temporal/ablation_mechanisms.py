#!/usr/bin/env python3
"""Experimental ablation mechanisms for the temporal (rolling-residency) MoE router.

NEGATIVE-RESULT ABLATIONS — kept ONLY for reproducibility. Every mechanism in this module is a
non-default, flag-gated (default-off) scoring/loss knob that produced a documented negative or
neutral result (see results/ablations/*.csv and results/ablations/FINDINGS.md). NONE of them is part
of the shipped rolling-residency router (that lives in temporal/temporal_router.py). They are
retained so the ablation CSVs remain regenerable and the negative findings stay auditable — do not
mistake anything here for the production mechanism.

Contents (all pure functions; unit-tested in temporal/tests/test_ablation_mechanisms.py):
  - auxfree_trigger_scores      aux-loss-free sigmoid+bias selection basis (DeepSeek-V3 recipe).
  - momentum_shaped_scores      "Karen" demand-momentum trigger shaping (A3 / A3q / logratio LG1).
  - gate_momentum_scores        K2 fast activator-inhibitor demand dynamics in the router gates.
  - anticipatory_target /       discounted future-demand target + soft-BCE (Track B / nomination head).
    anticipatory_bce_loss
  - nomination_head_logits /    decoupled stop-grad nomination head (H1) and its selection bonuses
    head_trigger_bonus /        (plain z-score H1, temporally-centered H2) + warmup gate + H3
    head_centered_bonus /       centered labels.
    head_selection_active /
    centered_demand_labels
  - bursty_window_loss          LG2 windowed-concentration (burstiness) loss.
  - coherence_bce_loss          temporal-coherence behaviour-cloning BCE loss.

The core router (temporal_router.py) reaches these lazily through `ablation_mechanisms` at its
flag-gated branch points; with every knob off (the default) none of this module runs.
"""
import torch
import torch.nn.functional as F

__all__ = [
    "auxfree_trigger_scores",
    "momentum_shaped_scores",
    "gate_momentum_scores",
    "anticipatory_target",
    "anticipatory_bce_loss",
    "nomination_head_logits",
    "head_trigger_bonus",
    "centered_demand_labels",
    "head_centered_bonus",
    "head_selection_active",
    "bursty_window_loss",
    "coherence_bce_loss",
]


def momentum_shaped_scores(base: torch.Tensor, probs: torch.Tensor,
                           beta_m: float, gamma_m: float,
                           alpha_m: float = 0.0, gamma_q: float = 0.015625,
                           mode: str = "add") -> torch.Tensor:
    """Karen's demand-momentum selection shaping (alignment program Track A, rung A3).

    score_t = base_t + beta_m * M_{t-1} - alpha_m * Q_{t-1}
      M_t = (1-gamma_m)*M_{t-1} + gamma_m * probs_t     (fast EMA, ~1/gamma_m tokens)
      Q_t = (1-gamma_q)*Q_{t-1} + gamma_q * probs_t     (slow EMA, ~1/gamma_q tokens)
    where `base` is the trigger's selection basis (raw logits, or sigmoid+bias under aux-free) and
    `probs` is the softmax of the RAW logits — the router's own demand history, never the cache
    state. Causal (t-1 state only); no bonus at t=0 (cold fill == base ranking). Momentum is
    residency-independent, so this is a pure pre-pass; feed the result into
    compute_resident_mask unchanged. Mirrors probe_replay._momentum_scores. Pure; unit-tested.

    alpha_m=0 is the original single-momentum rung (A3, uncentered). alpha_m>0 is Karen's FULL
    double-momentum formulation (re-tune rung A3q, the anti-pinning fix): with alpha_m == beta_m a
    chronically hot expert's M and Q converge and its bonus cancels to ~0 — momentum rewards
    RECENT demand only, so permanence must be continually re-earned (a formerly-hot expert even
    goes negative until Q decays). Targets the pinning failure of single momentum at s2@1e17
    (max-residency 85.3%). Note a per-token-uniform shift would be a no-op here (the residency
    trigger only compares scores within a token); Q varies per expert, so this is not one.

    mode="logratio" (local-consistency/global-diversity program, rung LG1): bonus becomes
      beta_m * ln((M_{t-1}+eps)/(Q_{t-1}+eps)),  eps = 1/E
    — popularity-NORMALIZED momentum. A chronically popular expert has M~Q -> bonus ~0 however
    hot it is; a rarely-used expert in a fresh burst has M>>Q -> large positive bonus (favoring
    global diversity at exactly the moments of local demand); a just-cooled expert goes negative.
    Zero time-average per expert, and the log keeps the bonus in O(beta) logit-scale units
    (additive M-Q was scale-inert on raw logits — mom-plain trained negative). alpha_m is ignored
    in this mode.
    """
    with torch.no_grad():
        S = base.shape[0]
        out = base.clone().float()
        M = probs[0].float()
        Q = probs[0].float()
        eps = 1.0 / base.shape[-1]
        for t in range(1, S):
            if mode == "logratio":
                bonus = beta_m * torch.log((M + eps) / (Q + eps))
            else:
                bonus = beta_m * M - alpha_m * Q
            out[t] = base[t].float() + bonus
            M = (1.0 - gamma_m) * M + gamma_m * probs[t].float()
            Q = (1.0 - gamma_q) * Q + gamma_q * probs[t].float()
        return out.to(base.dtype)


def gate_momentum_scores(logits: torch.Tensor, beta: float,
                         gamma_m: float = 0.125, gamma_q: float = 0.015625) -> torch.Tensor:
    """K2: fast demand dynamics IN THE ROUTER'S GATES (local-global program, rung K2).

    logits'_t = logits_t + beta * ln((M_{t-1}+eps)/(Q_{t-1}+eps)),  eps = 1/E
      M_t = (1-gamma_m)*M_{t-1} + gamma_m * softmax(logits'_t)   (fast, ~1/gamma_m tokens)
      Q_t = (1-gamma_q)*Q_{t-1} + gamma_q * softmax(logits'_t)   (slow, ~1/gamma_q tokens)

    Karen's zero-DC logratio momentum moved from the residency TRIGGER into the scores the router
    actually routes on — an activator-inhibitor system with built-in timescale separation: M is
    self-excitation (a demanded expert transiently easier to demand again -> bursts form), Q is
    the slow local inhibitor (a sustained run raises Q until the bonus self-extinguishes -> the
    burst must end -> forced rotation). The limit cycle lives in inference-time STATE, not in
    learned parameters: the bonus is detached (gradients reach W only through the raw-logit term)
    and has ~zero time-average per expert (logratio), so slow optimization cannot rectify it into
    a static popularity table (the failure mode of every prior mechanism). M/Q update on the
    softmax of the SHAPED logits — the demand that actually happened — so the inhibitor sees the
    amplified bursts it must extinguish (self-consistent closed loop). Causal (t-1 state only);
    t=0 has no bonus (cold fill preserved). Demand-referential only — never reads cache state.
    Pure function; unit-tested.
    """
    if beta <= 0:
        return logits
    with torch.no_grad():
        S, B, E = logits.shape
        eps = 1.0 / E
        bonus = torch.zeros(S, B, E, dtype=torch.float32, device=logits.device)
        p = torch.softmax(logits[0].float(), dim=-1)
        M = p.clone()
        Q = p.clone()
        for t in range(1, S):
            b_t = beta * torch.log((M + eps) / (Q + eps))
            bonus[t] = b_t
            p = torch.softmax(logits[t].float() + b_t, dim=-1)
            M = (1.0 - gamma_m) * M + gamma_m * p
            Q = (1.0 - gamma_q) * Q + gamma_q * p
    return logits + bonus.to(logits.dtype)


def anticipatory_target(logits: torch.Tensor, k: int, gamma: float):
    """Discounted future-demand target for the anticipatory loss (alignment program Track B).

    y_t(e) = sum_{j>=0} gamma^j * 1[e in top-k(t+j)]   (reverse scan: y_t = m_t + gamma*y_{t+1})
    where m_t is the UNCONSTRAINED top-k multi-hot of the raw logits (detached). The j=0 term is
    included so the loss optimum stays near the LM optimum (minimal gate distortion). Returns
    (target, valid): target = (1-gamma)*y in [0,1] (soft-BCE labels); valid[t] masks out the
    sequence tail (last ~1/(1-gamma) tokens, whose discounted sum is incomplete). Known
    approximation: the scan runs across packed-document boundaries (EOD-aware targets would need
    token ids in the router; boundary-adjacent tokens are <2% of the batch per probe experiment
    E8, so the contamination is noise-level). Pure function; unit-tested.
    """
    with torch.no_grad():
        S, B, E = logits.shape
        _, idx = logits.topk(k, dim=-1)
        m = torch.zeros(S, B, E, dtype=torch.float32, device=logits.device)
        m.scatter_(2, idx, 1.0)
        y = torch.empty_like(m)
        y[S - 1] = m[S - 1]
        for t in range(S - 2, -1, -1):
            y[t] = m[t] + gamma * y[t + 1]
        target = (1.0 - gamma) * y
        tail = min(S, max(1, int(round(1.0 / (1.0 - gamma)))))
        valid = torch.ones(S, dtype=torch.bool, device=logits.device)
        if tail < S:
            valid[S - tail:] = False
        else:
            valid[:] = False
        return target, valid


def anticipatory_bce_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    """Soft-BCE of raw router logits against the (detached) discounted future-demand target,
    averaged over valid (non-tail) positions only. Scalar."""
    lg = logits[valid]
    tg = target[valid].detach().to(lg.dtype)
    return F.binary_cross_entropy_with_logits(lg, tg)


# ---------------------------------------------------------------------------
# Decoupled stop-grad nomination head (local-global program, mechanism H1).
#
# A per-MoE-layer linear head W_f in R^{d x E} reads the SAME hidden state the
# router reads, but DETACHED — its BCE loss (vs the discounted future-demand
# target from anticipatory_target) can move ONLY the head weights, never the
# trunk or the router. Selection use: after a warmup, beta * zscore_E(sigmoid(
# head)) is added to the residency-trigger scores (z-scoring matters: a
# per-token-uniform shift is a no-op for the trigger, a per-expert z-score is
# not). Why this can win where momentum couldn't: content-based anticipation
# (E5 replay: +20-30pt coverage headroom for future-demand nomination) with the
# Goodhart solution unreachable — the loss cannot alter the demand process.
# Env knobs: HEAD_LAMBDA, HEAD_BETA, HEAD_GAMMA (0.5), HEAD_WARMUP_FRAC (0.25).
# ---------------------------------------------------------------------------
def nomination_head_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Nomination-head logits: W_f @ h with the trunk DETACHED (stop-grad).

    hidden: [seq, batch, d] — the router's input (post-jitter). Detached here, so no gradient
    from any head loss can ever reach the trunk. fp32 compute for BCE stability. Differentiable
    w.r.t. `weight` only. Pure function; unit-tested.
    """
    return F.linear(hidden.detach().float(), weight.float())


def head_trigger_bonus(head_logits: torch.Tensor, beta: float) -> torch.Tensor:
    """Selection bonus: beta * zscore_E(sigmoid(head_logits)), detached.

    z-score across the EXPERT dim per token: the residency trigger only compares scores within a
    token, so a per-token-uniform shift would be a no-op — centering removes the uniform part and
    the std normalization puts the bonus in O(beta) units regardless of head confidence scale.
    A constant head output yields ~0 bonus (eps guards the 0/0). Pure; unit-tested.
    """
    with torch.no_grad():
        p = torch.sigmoid(head_logits.float())
        mu = p.mean(dim=-1, keepdim=True)
        sd = p.std(dim=-1, keepdim=True, unbiased=False)
        return beta * (p - mu) / (sd + 1e-6)


def centered_demand_labels(target: torch.Tensor, gamma_c: float = 0.015625) -> torch.Tensor:
    """H3: binary head labels 'future demand ABOVE this expert's OWN running baseline'.

    target: [S,B,E] discounted future demand from anticipatory_target (detached, in [0,1]).
    label_t(e) = 1[target_t(e) > base_{t}(e)] with base a slow causal per-expert EMA of the
    target (gamma_c, ~64 tokens; init base = target[0]).

    Why: the H1 head learned a static popularity table because the raw BCE target is dominated
    by its stationary marginal. Centering each expert's label on its OWN baseline makes
    popularity literally unlearnable-as-positive: a CONSTANTLY-hot expert's target equals its
    baseline -> labels ~all 0; a dead expert -> all 0; ONLY above-own-baseline transients (burst
    onsets) are positive class. The head either learns genuine anticipation of demand anomalies
    or it learns nothing. Mirrors the Q2 predictability probe's logistic formulation (AUC
    0.70-0.87 on temporal substrates). Pure; unit-tested.
    """
    with torch.no_grad():
        S = target.shape[0]
        labels = torch.zeros_like(target)
        base = target[0].clone()
        for t in range(S):
            labels[t] = (target[t] > base).float()
            base = (1.0 - gamma_c) * base + gamma_c * target[t]
        return labels


def head_centered_bonus(head_logits: torch.Tensor, beta: float,
                        gamma_c: float = 0.015625) -> torch.Tensor:
    """H2: per-expert temporally-CENTERED head bonus — beta * (z_t - C_{t-1}),
    C_t = (1-gamma_c)*C_{t-1} + gamma_c*z_t, z = zscore_E(sigmoid(head_logits)).

    The plain head bonus (head_trigger_bonus) removes per-token-uniform shifts but NOT per-expert
    chronic bias: an expert the head chronically predicts hot gets a standing positive bonus —
    the concentration channel (g3 head cell: eff 183.9 -> 102.6). Subtracting each expert's own
    slow causal EMA makes the bonus ZERO TIME-AVERAGE PER EXPERT by construction: chronic
    favoritism cannot survive; only transient (bursty) anticipation — the local-cohesion signal —
    passes through. Causal (C uses t-1 state only; C_{-1}=0, so t=0 equals the uncentered bonus).
    Cross-token loop mirrors momentum_shaped_scores (same cost class, no_grad).
    Pure; unit-tested.
    """
    with torch.no_grad():
        z = head_trigger_bonus(head_logits, 1.0)
        out = torch.empty_like(z)
        C = torch.zeros_like(z[0])
        for t in range(z.shape[0]):
            out[t] = z[t] - C
            C = (1.0 - gamma_c) * C + gamma_c * z[t]
        return beta * out


def head_selection_active(curr_iteration: int, train_iters: int, warmup_frac: float) -> bool:
    """Warmup gate for the head's SELECTION use (the BCE loss trains from step 0 regardless).

    Active from iteration >= warmup_frac * train_iters. Unknown train_iters (<=0) -> inactive
    (fail safe: an untrained head must not shape selection). Pure; unit-tested.
    """
    if train_iters <= 0:
        return False
    return curr_iteration >= int(round(warmup_frac * train_iters))


def bursty_window_loss(logits: torch.Tensor, window: int) -> torch.Tensor:
    """Windowed-concentration loss (local-consistency/global-diversity program, rung LG2).

    Split the sequence into windows of `window` tokens; within each window compute the mean router
    demand p̄ = mean_t softmax(logits_t) and penalize its entropy H(p̄). Minimizing this makes
    demand CONCENTRATE within short windows (local consistency / burstiness); the standard global
    aux-loss (left untouched) keeps the cross-batch marginal flat, vetoing the global-collapse
    shortcut. Target equilibrium: bursty rotation — few experts per window, different experts
    across windows. Tail tokens beyond the last full window are dropped. Differentiable through
    softmax; scalar. Pure function; unit-tested.
    """
    S, B, E = logits.shape
    n = S // window
    if n == 0:
        return logits.sum() * 0.0
    p = torch.softmax(logits[: n * window].float(), dim=-1)
    pbar = p.view(n, window, B, E).mean(dim=1)              # [n, B, E]
    ent = -(pbar * torch.log(pbar + 1e-9)).sum(dim=-1)      # [n, B]
    return ent.mean()


def auxfree_trigger_scores(logits: torch.Tensor, expert_bias: torch.Tensor) -> torch.Tensor:
    """Selection basis for the residency trigger under aux-loss-free routing (DeepSeek-V3 recipe).

    Megatron selects experts by sigmoid(logits) + expert_bias (bias in SELECTION only; gates stay
    unbiased). The rolling-residency trigger must rank experts on the SAME basis, or the temporal
    selection would silently diverge from the paradigm it runs under. Sigmoid is elementwise
    monotone, so with expert_bias == 0 this reduces to the raw-logit ranking (identity of the
    induced mask). Pure function; unit-tested.
    """
    return torch.sigmoid(logits.float()) + expert_bias.to(torch.float32)


# ---------------------------------------------------------------------------
# Temporal-coherence auxiliary loss (BCE).
#
# Behaviour-cloning of the residency-masked policy into the raw router: for each
# token, treat the FINAL used set (the resident mask this token ran) as a multi-hot
# target and BCE the router's per-expert logits toward it. Independent sigmoids
# (not softmax) => a set-MEMBERSHIP pull, not a distribution clone: it aligns the
# router with what it used without forcing identical magnitudes. Raising the
# resident experts' logits this token makes next token's demand more likely to
# already sit resident -> fewer swaps -> higher retention. Target is detached.
# ---------------------------------------------------------------------------
def coherence_bce_loss(logits, resident_mask):
    """BCE aligning raw router logits with the final resident (used) set (detached target).

    logits: [seq, batch, E] raw router logits (differentiable, pre residency mask).
    resident_mask: [seq, batch, E] bool — the set each token actually used.
    Returns a scalar (mean BCE over all experts).
    """
    return F.binary_cross_entropy_with_logits(logits, resident_mask.to(logits.dtype).detach())

def cosmoes_block_loss(logits: torch.Tensor, block: int) -> torch.Tensor:
    """CoSMoEs block-selection penalty (Huber et al., arXiv:2503.00245), reimplemented.

    BASELINE_METHODS_COMPARISON.md baseline #1. Appendix E currently argues against CoSMoEs
    by quoting their reported table rather than running their method, which that document
    calls the most attackable move in the paper. This makes it runnable on our own isoFLOP
    sweep, so the comparison is a measurement instead of a citation.

    The method: cut the sequence into fixed-length blocks and penalise how many DISTINCT
    experts get used inside a block. Nothing is forbidden -- switching is merely made
    expensive -- so the model learns to stay with a smaller set within each block. Their own
    paper reports the trade directly: fewer expert changes, and a measurable quality drop.

    Implementation. "Number of distinct experts in a block" is a counting statistic and not
    differentiable, so use the used-at-least-once surrogate: treating each token's routing
    probability as an independent chance of touching expert e, the chance the block touches
    it at all is 1 - prod_t (1 - p_te), and summing that over experts gives a smooth
    estimate of the distinct-expert count.

    The obvious cheaper surrogate, max_t p_te, was tried first and rejected: it only tracks
    distinctness when routing is PEAKED. Under flat routing every expert's max is 1/E, so
    the sum is identically 1 no matter how many experts the block really touches -- the
    penalty goes silent in exactly the regime it is meant to punish. The product form gives
    ~3.3 on that case against a true expectation of E(1-(1-1/E)^block) = 3.3, and still
    gives ~1 for a block that uses one expert and ~block for one that rotates every token.
    Computed in log space, since the product underflows for long blocks.

    Contrast with bursty_window_loss, its nearest sibling here: that penalises the ENTROPY
    of a window's mean demand, which concentrates demand while leaving the support free to
    rotate; this penalises the SUPPORT SIZE itself. The distinction is the point of the
    comparison -- CoSMoEs shrinks how many experts a block touches without ever bounding
    the resident set, so it still has to provision memory for the experts it might miss,
    which is why it cannot serve at R = k by construction and we can.

    Args:
        logits: [seq, batch, num_experts] raw router logits, seq-first as Megatron routes.
        block: block length in tokens. Tail tokens beyond the last full block are dropped,
            matching bursty_window_loss's convention.
    Returns:
        Scalar, mean over blocks and batch of the soft distinct-expert count. Differentiable.
    """
    S, B, E = logits.shape
    n = S // block
    if n == 0:
        return logits.sum() * 0.0                       # no full block: inert, keeps graph
    p = F.softmax(logits[: n * block].float(), dim=-1)  # [n*block, B, E]
    p = p.view(n, block, B, E)
    # 1 - prod_t (1 - p), in log space: prod = exp(sum log1p(-p))
    log_unused = torch.log1p(-p.clamp(max=1 - 1e-6)).sum(dim=1)   # [n, B, E]
    soft_used = -torch.expm1(log_unused)                          # 1 - exp(log_unused)
    return soft_used.sum(dim=-1).mean()                 # soft |distinct experts| per block
