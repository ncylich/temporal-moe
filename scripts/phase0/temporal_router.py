#!/usr/bin/env python3
"""Temporal MoE — rolling-residency routing (proof of concept).

A trained-from-scratch MoE variant that keeps only K = k routed experts resident per layer and
streams one expert at a time (evicting the LRU or least-wanted resident — see `evict` below),
so the resident footprint is a small fraction of all E experts.
See docs/research/temporal-moe.md (§2 "rolling residency").

The whole architecture is:
  1. `compute_resident_mask` — the pure, unit-tested rolling-residency selection (this is the only
     novel logic; tests in test_temporal_router.py).
  2. `temporal_forward` — a ~6-line replacement for Megatron's `TopKRouter.forward` that restricts
     each token's selection to its resident set by masking non-resident logits to -inf, then calls
     the UNMODIFIED `routing()` so z-loss, aux-loss and top-k are reused verbatim.
  3. `install` — monkeypatches `TopKRouter.forward` (same approach as expert_load.py).

The Megatron import lives inside `install()` so this module (and its tests) import with plain torch
and no Megatron present.
"""
import os
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                           # triton missing -> eager/graph fallback
    _HAS_TRITON = False


def compute_resident_mask(logits: torch.Tensor, k: int, evict: str = "lru",
                          tau: float = 0.0, ema_beta: float = 1.0) -> torch.Tensor:
    """Rolling-residency expert selection.

    Args:
        logits: [seq, batch, num_experts] router logits (seq-first, as Megatron's router sees them).
        k: resident-set size (= top-k; K = k for this PoC).
        evict: which resident to remove when a swap happens (experiment knob, same swap *trigger*):
            "lru"       — oldest last-refresh time (cache-style; protects just-loaded experts from
                          immediate re-eviction → less thrash; score-neutral w.r.t. the aux loss).
            "min_logit" — lowest current logit, i.e. the same "worst resident" the swap trigger
                          compares against (most consistent; quality-greedy; simpler).
        tau: hysteresis margin (logit units): swap fires only if the best non-resident beats the
            worst resident by MORE than tau. tau=0 == shipped behavior.
        ema_beta: causal EMA smoothing of the TRIGGER stream (weight on the current token;
            trig[t] = (1-ema_beta)*trig[t-1] + ema_beta*logits[t]; ema_beta=1 == shipped, no
            smoothing). Shapes only WHICH experts are resident — the caller applies the returned
            mask to the RAW logits, so gates/routing always see raw scores (matches the canonical
            raw-demand-scored A1 replay semantics in probe_replay.py).

    Returns:
        Boolean mask [seq, batch, num_experts] with exactly `k` True per (seq, batch) token: the
        experts resident — and therefore usable — for that token.

    Policy (swap-then-use): a token pulls in one expert and uses it the SAME step (no prefetch lag),
    so the token always gets its top-k experts that are within +1 swap of the current set.
        R_0 = top-k(logits[0])                      # cold fill: first token picks all k experts
        for t >= 1:  R_t = swap(R_{t-1}, logits[t]):
            nominee = argmax over NON-resident logits[t]
            swap in nominee iff it beats the worst resident (i.e. R_{t-1} != global top-k),
            evicting the resident chosen by `evict`.
        mask[t] = R_t                               # the post-swap set the token actually uses
    Refresh times (for "lru"): cold-fill experts rank by ascending logit (lowest = oldest); each
    nomination is the newest. All resident refresh times stay distinct, so eviction is deterministic.
    """
    assert evict in ("lru", "min_logit"), f"unknown evict policy {evict!r}"
    use_lru = evict == "lru"
    S, B, E = logits.shape
    dev = logits.device
    NEG, POS = float("-inf"), float("inf")

    trig = logits                                           # trigger stream (== raw when ema_beta=1)
    if ema_beta < 1.0:
        trig = torch.empty_like(logits)
        trig[0] = logits[0]
        for _t in range(1, S):
            trig[_t] = (1.0 - ema_beta) * trig[_t - 1] + ema_beta * logits[_t]

    refresh = torch.full((B, E), NEG, device=dev)           # last-refresh time per expert ("lru" only)
    out = torch.zeros(S, B, E, dtype=torch.bool, device=dev)

    # --- t=0 cold fill: R_0 = top-k(trig[0]) (== top-k(logits[0]): EMA is identity at t=0) ---
    resident = torch.zeros(B, E, dtype=torch.bool, device=dev)
    _, top_i = trig[0].topk(k, dim=-1)                      # [B,k], descending by logit
    resident.scatter_(1, top_i, True)
    # highest logit -> newest (largest refresh); lowest of the k -> oldest (0).
    rank_refresh = torch.arange(k - 1, -1, -1, device=dev).float().expand(B, k)
    refresh.scatter_(1, top_i, rank_refresh)
    out[0] = resident

    for t in range(1, S):
        lt = trig[t]                                        # token t pulls in one expert and uses it
        nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)      # best non-resident [B]
        worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)       # worst resident   [B]
        do_swap = (nom_val > worst_val + tau).unsqueeze(-1)             # [B,1]; tau=0 == shipped
        evict_key = refresh if use_lru else lt
        evict_i = evict_key.masked_fill(~resident, POS).argmin(dim=-1)  # resident to remove [B]
        evicted = F.one_hot(evict_i, E).bool() & do_swap               # [B,E]
        nominee = F.one_hot(nom_i, E).bool() & do_swap                 # [B,E]
        resident = (resident & ~evicted) | nominee
        refresh = refresh.masked_fill(nominee, float(k + t))           # newest (read only when "lru")
        out[t] = resident

    return out


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


def coherence_bce_loss(logits, resident_mask):
    """BCE aligning raw router logits with the final resident (used) set (detached target).

    logits: [seq, batch, E] raw router logits (differentiable, pre residency mask).
    resident_mask: [seq, batch, E] bool — the set each token actually used.
    Returns a scalar (mean BCE over all experts).
    """
    return F.binary_cross_entropy_with_logits(logits, resident_mask.to(logits.dtype).detach())


# ---------------------------------------------------------------------------
# GPU acceleration (additive — `compute_resident_mask` above is unchanged and
# remains the canonical, unit-tested reference + the fallback).
#
# The sequence scan launches ~6 tiny CUDA kernels per step × 2048 steps × every
# MoE layer × every micro-batch, which is purely kernel-launch-bound (~10× slower
# than the baseline router). `_step` is the per-step body factored out so it can be
# captured once into a CUDA graph and replayed, collapsing the launch storm (~7×
# faster, measured). The captured body is numerically identical to the reference
# loop; `compute_resident_mask_accel` enforces that at runtime (one-time equality
# check vs `compute_resident_mask`) and RAISES HARD on any mismatch (no fallback).
# ---------------------------------------------------------------------------
def _step(lt, resident, refresh, tval, use_lru):
    """One rolling-residency update step (the body of compute_resident_mask's loop)."""
    E = lt.shape[-1]
    NEG, POS = float("-inf"), float("inf")
    nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)
    worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)
    do_swap = (nom_val > worst_val).unsqueeze(-1)
    evict_key = refresh if use_lru else lt
    evict_i = evict_key.masked_fill(~resident, POS).argmin(dim=-1)
    evicted = F.one_hot(evict_i, E).bool() & do_swap
    nominee = F.one_hot(nom_i, E).bool() & do_swap
    resident = (resident & ~evicted) | nominee
    refresh = torch.where(nominee, tval, refresh)
    return resident, refresh


_graph_cache = {}
_scan_path = None   # "cuda-graph" | "eager" — logged once


def _graph_scan(logits, k, use_lru):
    """CUDA-graph fast path: capture `_step` once per (B,E,policy) and replay over the sequence."""
    S, B, E = logits.shape
    dev, dt = logits.device, logits.dtype
    key = (B, E, use_lru, dt)
    if key not in _graph_cache:
        lt_s = torch.zeros(B, E, device=dev, dtype=dt)
        res_s = torch.zeros(B, E, dtype=torch.bool, device=dev)
        ref_s = torch.zeros(B, E, device=dev, dtype=dt)
        tval_s = torch.zeros((), device=dev, dtype=dt)
        torch.cuda.synchronize()
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                r, rf = _step(lt_s, res_s, ref_s, tval_s, use_lru)
                res_s.copy_(r); ref_s.copy_(rf)
        torch.cuda.current_stream().wait_stream(warm)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            r, rf = _step(lt_s, res_s, ref_s, tval_s, use_lru)
            res_s.copy_(r); ref_s.copy_(rf)
        _graph_cache[key] = (lt_s, res_s, ref_s, tval_s, g)
    lt_s, res_s, ref_s, tval_s, g = _graph_cache[key]

    out = torch.zeros(S, B, E, dtype=torch.bool, device=dev)
    res_s.zero_(); ref_s.fill_(float("-inf"))
    _, top_i = logits[0].topk(k, dim=-1)
    res_s.scatter_(1, top_i, True)
    rank = torch.arange(k - 1, -1, -1, device=dev, dtype=dt).expand(B, k)
    ref_s.scatter_(1, top_i, rank)
    out[0].copy_(res_s)
    for t in range(1, S):
        lt_s.copy_(logits[t]); tval_s.fill_(float(k + t))
        g.replay()
        out[t].copy_(res_s)
    return out


# ---------------------------------------------------------------------------
# Triton fast path (default) — the whole sequential scan in ONE kernel launch.
#
# The CUDA-graph path above still launches ~4 kernels/step × 2048 steps (the
# replay is GPU-side but every captured op is its own dispatch), ~91 ms/call.
# This fuses the entire t=1..S-1 residency update into a single kernel: grid of
# B programs (one per batch element), each holding the E-wide resident/refresh
# state in registers and looping the steps in-kernel. One launch total -> ~1 ms
# at the production shape ([2048,32,64], k=6), a further ~90× over the graph.
#
# The t=0 cold fill (torch.topk) is done in torch so its tie-breaking matches the
# reference exactly; the kernel reproduces torch's first-index argmax/argmin tie
# semantics via "max value, then min index achieving it". All comparisons are in
# fp32: upcasting the logits is exact and order-preserving, so the boolean mask is
# bit-identical to the reference for any input dtype. Verified == reference once at
# runtime by compute_resident_mask_accel, which raises hard on any mismatch.
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _scan_kernel(logits_ptr, res0_ptr, ref0_ptr, out_ptr,
                     S, B, k, E, use_lru: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        e = tl.arange(0, BLOCK)
        valid = e < E                                        # BLOCK = next pow2 >= E; mask pad lanes
        NEG, POS = float("-inf"), float("inf")
        resident = tl.load(res0_ptr + b * E + e, mask=valid, other=0) != 0
        refresh = tl.load(ref0_ptr + b * E + e, mask=valid, other=0.0)
        tl.store(out_ptr + b * E + e, resident.to(tl.int8), mask=valid)   # out[0] = cold fill
        for t in range(1, S):
            base = t * B * E + b * E + e
            lt = tl.load(logits_ptr + base, mask=valid, other=0.0).to(tl.float32)
            # nominee = argmax over NON-resident logits (first index on ties)
            masked_nom = tl.where(valid & (resident == 0), lt, NEG)
            nom_val = tl.max(masked_nom, 0)
            nom_i = tl.min(tl.where(masked_nom == nom_val, e, BLOCK), 0)
            # worst resident logit (the swap trigger)
            worst_val = tl.min(tl.where(resident, lt, POS), 0)
            do_swap = nom_val > worst_val
            # evict: lru -> oldest refresh; min_logit -> lowest current logit (first index on ties)
            if use_lru:
                evict_key = refresh
            else:
                evict_key = lt
            masked_ev = tl.where(resident, evict_key, POS)
            ev_val = tl.min(masked_ev, 0)
            evict_i = tl.min(tl.where(masked_ev == ev_val, e, BLOCK), 0)
            is_evict = (e == evict_i) & do_swap
            is_nom = (e == nom_i) & do_swap
            resident = (resident & (is_evict == 0)) | is_nom
            refresh = tl.where(is_nom, (k + t).to(tl.float32), refresh)   # newest (read only when lru)
            tl.store(out_ptr + base, resident.to(tl.int8), mask=valid)


def _triton_scan(logits, k, use_lru):
    """Single-launch Triton fast path: cold fill in torch, full t>=1 scan in one kernel."""
    if not _HAS_TRITON:
        raise RuntimeError("triton unavailable")
    S, B, E = logits.shape
    dev = logits.device
    logits = logits.contiguous()
    resident0 = torch.zeros(B, E, dtype=torch.bool, device=dev)
    refresh0 = torch.full((B, E), float("-inf"), device=dev, dtype=torch.float32)
    _, top_i = logits[0].topk(k, dim=-1)
    resident0.scatter_(1, top_i, True)
    rank = torch.arange(k - 1, -1, -1, device=dev, dtype=torch.float32).expand(B, k)
    refresh0.scatter_(1, top_i, rank)
    out = torch.empty(S, B, E, dtype=torch.int8, device=dev)
    BLOCK = 1 << (E - 1).bit_length()
    _scan_kernel[(B,)](logits, resident0.to(torch.int8), refresh0, out,
                       S, B, k, E, use_lru, BLOCK, num_warps=1)
    return out.to(torch.bool)


def compute_resident_mask_accel(logits, k, evict="lru", tau=0.0, ema_beta=1.0):
    """Resident mask via a GPU fast path; identical result to compute_resident_mask.

    On CUDA, runs the Triton single-launch scan (TEMPORAL_SCAN default "triton"; "graph" selects the
    CUDA-graph path, "eager" forces the reference). The fast path is verified bit-exact against the
    reference once at startup; a mismatch (or any kernel/capture error) RAISES HARD — we deliberately
    do NOT silently fall back to eager, so a correctness bug crashes the run loudly instead of
    degrading to the ~10x-slower path unnoticed across a multi-hour sweep.
    """
    global _scan_path
    mode = os.environ.get("TEMPORAL_SCAN", "triton")
    if tau != 0.0 or ema_beta < 1.0:                        # shaped trigger: reference only
        # The Triton/CUDA-graph kernels implement the shipped (tau=0, ema_beta=1) semantics;
        # running them here would silently ignore the knobs. Eval-only workloads tolerate eager.
        if _scan_path != "eager-shaped":
            _scan_path = "eager-shaped"
            print(f"[temporal] scan path: eager (shaped trigger: tau={tau}, ema_beta={ema_beta})")
        with torch.no_grad():
            return compute_resident_mask(logits, k, evict, tau=tau, ema_beta=ema_beta)
    if not logits.is_cuda or mode == "eager":               # CPU (tests) or explicit opt-out
        if _scan_path is None:
            _scan_path = "eager"
            print("[temporal] scan path: eager")
        with torch.no_grad():
            return compute_resident_mask(logits, k, evict)
    # GPU fast path — correct or crash; no fallback.
    with torch.no_grad():
        out, pathname = ((_graph_scan(logits, k, evict == "lru"), "cuda-graph") if mode == "graph"
                         else (_triton_scan(logits, k, evict == "lru"), "triton"))
        if _scan_path is None:                              # one-time bit-exactness gate (hard)
            ref = compute_resident_mask(logits, k, evict)
            if not torch.equal(out, ref):
                bad = (out != ref).any(dim=-1).sum().item()
                raise RuntimeError(
                    f"[temporal] FAST PATH '{pathname}' DISAGREES WITH REFERENCE on {bad} tokens — "
                    f"aborting (kernel bug; do not trust results). Set TEMPORAL_SCAN=eager to bypass.")
            _scan_path = pathname
            print(f"[temporal] scan path: {pathname} (verified == reference)")
    return out


def temporal_forward(self, input: torch.Tensor):
    """Drop-in replacement for TopKRouter.forward: restrict selection to the resident set.

    Masking non-resident experts to -inf and calling the unmodified self.routing() keeps z-loss,
    aux-loss and the top-k/dispatch path byte-for-byte identical (they just see masked logits).

    If TEMPORAL_COHERENCE_LAMBDA>0, add the BCE coherence loss (train only): its
    gradient is injected onto the raw logits via MoEAuxLossAutoScaler (same mechanism as z-loss),
    pulling the router toward its own resident set so future tokens swap less.
    """
    input = self.apply_input_jitter(input)
    logits = self.gating(input)                             # [seq, batch, num_experts]
    k = self.config.moe_router_topk
    mom_beta = float(os.environ.get("TEMPORAL_MOM_BETA", "0"))
    mom_apply = os.environ.get("TEMPORAL_MOM_APPLY", "trigger")
    if mom_beta > 0 and mom_apply == "gates":
        # K2: shape the scores the router ROUTES on (demand dynamics), not just the trigger.
        # Bonus is detached inside; downstream (trigger, losses, mask, routing) sees the shaped
        # stream as "the" logits — the demand process itself is now bursty-rotating.
        logits = gate_momentum_scores(
            logits, mom_beta,
            gamma_m=float(os.environ.get("TEMPORAL_MOM_GAMMA", "0.125")),
            gamma_q=float(os.environ.get("TEMPORAL_MOM_GAMMA_Q", "0.015625")))
    auxfree = bool(getattr(self, "enable_expert_bias", False)) and getattr(self, "expert_bias", None) is not None
    trig = auxfree_trigger_scores(logits, self.expert_bias).to(logits.dtype) if auxfree else logits
    if mom_beta > 0 and mom_apply == "trigger":
        probs = torch.softmax(logits.float(), dim=-1)
        trig = momentum_shaped_scores(trig, probs, mom_beta,
                                      float(os.environ.get("TEMPORAL_MOM_GAMMA", "0.125")),
                                      alpha_m=float(os.environ.get("TEMPORAL_MOM_ALPHA", "0")),
                                      gamma_q=float(os.environ.get("TEMPORAL_MOM_GAMMA_Q", "0.015625")),
                                      mode=os.environ.get("TEMPORAL_MOM_MODE", "add"))
    head_lam = float(os.environ.get("HEAD_LAMBDA", "0"))
    head_beta = float(os.environ.get("HEAD_BETA", "0"))
    head_logits = None
    if (head_lam > 0 or head_beta > 0) and getattr(self, "nom_head_weight", None) is not None:
        head_logits = nomination_head_logits(input, self.nom_head_weight)
        if head_beta > 0:
            # HEAD_FORCE_ACTIVE=1: eval-only screens on a trained checkpoint (fresh process has
            # no curr_iteration, which would silently disable the bonus). Never set in training.
            if os.environ.get("HEAD_FORCE_ACTIVE", "0") == "1":
                active = True
            else:
                from megatron.training import get_args
                targs = get_args()
                active = head_selection_active(int(getattr(targs, "curr_iteration", 0) or 0),
                                               int(getattr(targs, "train_iters", 0) or 0),
                                               float(os.environ.get("HEAD_WARMUP_FRAC", "0.25")))
            if active:
                if os.environ.get("HEAD_CENTER", "0") == "1":
                    bonus = head_centered_bonus(head_logits, head_beta,
                                                float(os.environ.get("HEAD_GAMMA_C", "0.015625")))
                else:
                    bonus = head_trigger_bonus(head_logits, head_beta)
                trig = (trig.float() + bonus).to(trig.dtype)
    # R-knob (de-lexicalization dose): residency-set size R >= k, decoupled from top-k. The cache
    # holds R experts (cold fill = top-R, same <=1 swap/token trigger/evict on the R-set) and the
    # router selects top-k AMONG residents. R=k (default) == shipped maximal constraint; R=E ==
    # unconstrained full MoE (mask all-True, masked_fill a no-op). Zero FLOP change at any R.
    resid_R = int(os.environ.get("TEMPORAL_RESIDENCY_R", "0")) or k
    mask = compute_resident_mask_accel(
        trig, resid_R, evict=os.environ.get("TEMPORAL_EVICT", "lru"),
        tau=float(os.environ.get("TEMPORAL_RHO", "0")),
        ema_beta=float(os.environ.get("TEMPORAL_EMA_BETA", "1.0")))
    lam = float(os.environ.get("TEMPORAL_COHERENCE_LAMBDA", "0"))
    if lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        coh = coherence_bce_loss(logits, mask)
        logits = MoEAuxLossAutoScaler.apply(logits, lam * coh)   # inject grad onto raw logits
        save_to_aux_losses_tracker("coherence_loss", coh.detach(),
                                   self.layer_number, self.config.num_layers)
    bw_lam = float(os.environ.get("BURSTY_LAMBDA", "0"))
    if bw_lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        bw = bursty_window_loss(logits, int(os.environ.get("BURSTY_WINDOW", "32")))
        logits = MoEAuxLossAutoScaler.apply(logits, bw_lam * bw)
        save_to_aux_losses_tracker("bursty_loss", bw.detach(),
                                   self.layer_number, self.config.num_layers)
    ant_lam = float(os.environ.get("ANTICIPATORY_LAMBDA", "0"))
    if ant_lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        gamma = float(os.environ.get("ANTICIPATORY_GAMMA", "0.5"))
        tgt, valid = anticipatory_target(logits, k, gamma)
        ant = anticipatory_bce_loss(logits, tgt, valid)
        logits = MoEAuxLossAutoScaler.apply(logits, ant_lam * ant)
        save_to_aux_losses_tracker("anticipatory_loss", ant.detach(),
                                   self.layer_number, self.config.num_layers)
    head_bce = None
    if head_lam > 0 and self.training and head_logits is not None:
        # BCE(head(h.detach()), discounted-future-demand) — target from the RAW (pre-mask) logits,
        # detached inside anticipatory_target. The loss graph touches ONLY nom_head_weight.
        tgt, valid = anticipatory_target(logits, k, float(os.environ.get("HEAD_GAMMA", "0.5")))
        if os.environ.get("HEAD_TARGET_CENTER", "0") == "1":
            # H3: labels centered on each expert's own baseline — popularity unlearnable.
            tgt = centered_demand_labels(tgt, float(os.environ.get("HEAD_GAMMA_C", "0.015625")))
        head_bce = anticipatory_bce_loss(head_logits, tgt, valid)
    logits = logits.masked_fill(~mask, float("-inf"))       # only resident experts are selectable
    if auxfree:
        # Selection over the k unmasked residents is a no-op top-k, but sigmoid(-inf)=0 plus a
        # positive bias could in principle outrank a low-scoring resident inside Megatron's
        # biased selection — zero the bias for this call to make the residency invariant
        # unconditional. The bias still governed the trigger above (where the paradigm acts),
        # and the per-step bias UPDATE uses the routing_map (residents) — the correct load signal.
        saved_bias = self.expert_bias
        self.expert_bias = torch.zeros_like(saved_bias)
        try:
            probs, routing_map = self.routing(logits)
        finally:
            self.expert_bias = saved_bias
    else:
        probs, routing_map = self.routing(logits)
    if head_bce is not None:
        # Attach the head loss to the model loss via the aux-loss autoscaler hooked on the
        # routing OUTPUT (gate probs) — deliberately NOT on the router logits (the Track-B
        # Goodhart path). The scaler's backward passes the probs gradient through UNCHANGED and
        # kicks off backward on `head_bce`, whose graph is detached from the trunk — so the only
        # parameters that can move are the head weights. Tracked as "head_bce_loss" for logging.
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        probs = MoEAuxLossAutoScaler.apply(probs, head_lam * head_bce)
        save_to_aux_losses_tracker("head_bce_loss", head_bce.detach(),
                                   self.layer_number, self.config.num_layers)
    return probs, routing_map


def banner_knobs() -> str:
    """The env-knob suffix of the install banner (pure — reads only os.environ; unit-tested)."""
    tau = os.environ.get("TEMPORAL_RHO", "0"); beta = os.environ.get("TEMPORAL_EMA_BETA", "1.0")
    knobs = f", tau={tau}, ema_beta={beta}" if (float(tau) != 0.0 or float(beta) < 1.0) else ""
    if int(os.environ.get("TEMPORAL_RESIDENCY_R", "0")) > 0:
        knobs += f", residency_R={os.environ.get('TEMPORAL_RESIDENCY_R')}"
    if os.environ.get("AUXFREE", "0") == "1":
        knobs += ", auxfree-trigger=sigmoid+bias"
    if float(os.environ.get("TEMPORAL_MOM_BETA", "0")) > 0:
        dm = ""
        if os.environ.get("TEMPORAL_MOM_APPLY", "trigger") == "gates":
            dm = f", apply=gates, gamma_q={os.environ.get('TEMPORAL_MOM_GAMMA_Q', '0.015625')}"
        elif os.environ.get("TEMPORAL_MOM_MODE", "add") == "logratio":
            dm = f", mode=logratio, gamma_q={os.environ.get('TEMPORAL_MOM_GAMMA_Q', '0.015625')}"
        elif float(os.environ.get("TEMPORAL_MOM_ALPHA", "0")) > 0:
            dm = (f", alpha={os.environ.get('TEMPORAL_MOM_ALPHA')}, "
                  f"gamma_q={os.environ.get('TEMPORAL_MOM_GAMMA_Q', '0.015625')}")
        knobs += (f", momentum(beta={os.environ.get('TEMPORAL_MOM_BETA')}, "
                  f"gamma={os.environ.get('TEMPORAL_MOM_GAMMA', '0.125')}{dm})")
    if float(os.environ.get("BURSTY_LAMBDA", "0")) > 0:
        knobs += (f", bursty(lambda={os.environ.get('BURSTY_LAMBDA')}, "
                  f"window={os.environ.get('BURSTY_WINDOW', '32')})")
    if float(os.environ.get("ANTICIPATORY_LAMBDA", "0")) > 0:
        knobs += (f", anticipatory(lambda={os.environ.get('ANTICIPATORY_LAMBDA')}, "
                  f"gamma={os.environ.get('ANTICIPATORY_GAMMA', '0.5')})")
    if float(os.environ.get("HEAD_LAMBDA", "0")) > 0 or float(os.environ.get("HEAD_BETA", "0")) > 0:
        hc = ""
        if os.environ.get("HEAD_TARGET_CENTER", "0") == "1":
            hc += f", target_center=1, gamma_c={os.environ.get('HEAD_GAMMA_C', '0.015625')}"
        if os.environ.get("HEAD_CENTER", "0") == "1":
            hc += f", center=1, gamma_c={os.environ.get('HEAD_GAMMA_C', '0.015625')}"
        if os.environ.get("HEAD_FORCE_ACTIVE", "0") == "1":
            hc += ", force_active=1"
        knobs += (f", head(lambda={os.environ.get('HEAD_LAMBDA', '0')}, "
                  f"beta={os.environ.get('HEAD_BETA', '0')}, "
                  f"gamma={os.environ.get('HEAD_GAMMA', '0.5')}, "
                  f"warmup={os.environ.get('HEAD_WARMUP_FRAC', '0.25')}{hc})")
    return knobs


def _head_patched_init(orig_init):
    """Wrap TopKRouter.__init__ to register the nomination head W_f in R^{E x d}.

    Registered as a Parameter at construction time (before DDP wrap / optimizer build), same
    init + dtype + sequence_parallel treatment as the router's own gate weight. Changes the
    checkpoint shape — head cells are trained from scratch only.
    """
    def _init(self, config, *a, **kw):
        orig_init(self, config, *a, **kw)
        self.nom_head_weight = torch.nn.Parameter(
            torch.empty((config.num_moe_experts, config.hidden_size), dtype=torch.float32))
        if config.perform_initialization:
            config.init_method(self.nom_head_weight)
        self.nom_head_weight.data = self.nom_head_weight.data.to(dtype=config.params_dtype)
        setattr(self.nom_head_weight, 'sequence_parallel', config.sequence_parallel)
    return _init


def install():
    """Monkeypatch TopKRouter.forward (call once at startup, before model build)."""
    from megatron.core.transformer.moe.router import TopKRouter
    TopKRouter.forward = temporal_forward
    if float(os.environ.get("HEAD_LAMBDA", "0")) > 0 or float(os.environ.get("HEAD_BETA", "0")) > 0:
        TopKRouter.__init__ = _head_patched_init(TopKRouter.__init__)
    print(f"[temporal] rolling-residency router installed "
          f"(evict={os.environ.get('TEMPORAL_EVICT', 'lru')}{banner_knobs()})")
