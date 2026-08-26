#!/usr/bin/env python3
"""Temporal MoE — rolling-residency routing (proof of concept).

A trained-from-scratch MoE variant that keeps only K = k routed experts resident per layer and
streams one expert at a time (evicting the LRU or least-wanted resident — see `evict` below),
so the resident footprint is a small fraction of all E experts.
See docs/research/temporal-moe.md (§2 "rolling residency").

The whole architecture is:
  1. `compute_resident_mask` — the pure, unit-tested rolling-residency selection (this is the only
     novel logic; tests in temporal/tests/test_temporal_router.py).
  2. `temporal_forward` — a ~6-line replacement for Megatron's `TopKRouter.forward` that restricts
     each token's selection to its resident set by masking non-resident logits to -inf, then calls
     the UNMODIFIED `routing()` so z-loss, aux-loss and top-k are reused verbatim.
  3. `install` — monkeypatches `TopKRouter.forward` (same approach as expert_load.py).

The Megatron import lives inside `install()` so this module (and its tests) import with plain torch
and no Megatron present.

Experimental (default-off, negative-result) scoring/loss knobs — aux-free trigger, momentum,
logratio momentum, coherence/anticipatory/bursty losses, nomination head — live in the sibling
module `ablation_mechanisms`; the core reaches them only at the flag-gated branch points in
`temporal_forward`. With every knob off (the default) that module never runs.
"""
import os
import torch
import torch.nn.functional as F

from . import ablation_mechanisms as ab

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                           # triton missing -> eager/graph fallback
    _HAS_TRITON = False


def compute_resident_mask(logits: torch.Tensor, k: int, evict: str = "lru",
                          tau: float = 0.0, ema_beta: float = 1.0,
                          swaps: int = 1,
                          init_resident: torch.Tensor = None,
                          init_refresh: torch.Tensor = None,
                          t0: int = 0,
                          return_state: dict = None) -> torch.Tensor:
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
    # swaps: budget of resident-set changes per token (the swap-axis knob; 1 == shipped).
    # Each sub-swap repeats the same trigger: best non-resident vs worst resident. The greedy
    # exchange converges to top-k(logits[t]) in <= k sub-swaps, so swaps >= k reproduces the
    # unconstrained per-token top-k exactly -- the analytic anchor the tests check.
    assert evict in ("lru", "min_logit"), f"unknown evict policy {evict!r}"
    assert swaps >= 1
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

    # --- RESUME: continue a walk already in progress -------------------------
    # With init_resident given, this call is the tail of a longer sequence, so
    # there is no cold fill: token 0 of this chunk is an ordinary swap step
    # against the incoming set. The resident set is path-dependent, so resuming
    # from anything other than the true S_t is a different run wearing the same
    # prefix -- that is why the state is passed in rather than re-derived.
    # test_resume_residency.py holds splice == continuous run to BIT equality.
    if return_state is not None and not isinstance(return_state, dict):
        raise TypeError("return_state must be a dict to receive the final state")
    if init_resident is not None:
        resident = init_resident.clone()
        refresh = (init_refresh.clone() if init_refresh is not None
                   else torch.full((B, E), NEG, device=dev))
        assert resident.shape == (B, E), \
            f"init_resident {tuple(resident.shape)} != (B,E) {(B, E)}"
        assert bool((resident.sum(-1) == k).all()), \
            "init_resident must hold exactly k experts per row"
        for t in range(S):                                  # every token is a swap step
            lt = trig[t]
            for _sw in range(swaps):
                nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)
                worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)
                do_swap = (nom_val > worst_val + tau).unsqueeze(-1)
                evict_key = refresh if use_lru else lt
                evict_i = evict_key.masked_fill(~resident, POS).argmin(dim=-1)
                evicted = F.one_hot(evict_i, E).bool() & do_swap
                nominee = F.one_hot(nom_i, E).bool() & do_swap
                resident = (resident & ~evicted) | nominee
                # absolute position: LRU compares refresh times across the whole
                # walk, so a resumed chunk must keep counting from t0, not restart
                refresh = refresh.masked_fill(nominee, float(k + t0 + t))
            out[t] = resident
        if return_state is not None:
            return_state["resident"], return_state["refresh"] = resident, refresh
        return out

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
        for _sw in range(swaps):
            nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)  # best non-resident [B]
            worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)   # worst resident   [B]
            do_swap = (nom_val > worst_val + tau).unsqueeze(-1)         # [B,1]; tau=0 == shipped
            evict_key = refresh if use_lru else lt
            evict_i = evict_key.masked_fill(~resident, POS).argmin(dim=-1)  # resident to remove
            evicted = F.one_hot(evict_i, E).bool() & do_swap            # [B,E]
            nominee = F.one_hot(nom_i, E).bool() & do_swap              # [B,E]
            resident = (resident & ~evicted) | nominee
            refresh = refresh.masked_fill(nominee, float(k + t))        # newest ("lru" only)
        out[t] = resident

    if return_state is not None:      # hand-off point for a resumed continuation
        return_state["resident"], return_state["refresh"] = resident, refresh
    return out


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
    do_swap = (nom_val > worst_val + _RHO).unsqueeze(-1)   # _RHO=0 -> published rule
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
# semantics either via "max value, then min index achieving it" or via Triton's
# tie_break_left fused reduction (same result; see the kernel). All comparisons are
# in fp32: upcasting the logits is exact and order-preserving, so the boolean mask is
# bit-identical to the reference for any input dtype. Verified == reference once at
# runtime by compute_resident_mask_accel, which raises hard on any mismatch.
#
# The loop is latency-bound, not throughput-bound: one program per batch element and a
# strictly sequential recurrence over S, so per-token cost is a dependent chain of a
# global load plus warp-shuffle reductions. It is tuned on that basis -- prefetch to hide
# the load, and a reduction shape/num_warps picked per BLOCK. temporal/bench_scan.py is
# the benchmark those choices come from; re-run it before changing any of them.
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _scan_kernel(logits_ptr, res0_ptr, ref0_ptr, out_ptr,
                     S, B, k, E, use_lru: tl.constexpr, BLOCK: tl.constexpr,
                     NSWAPS: tl.constexpr):
        b = tl.program_id(0)
        e = tl.arange(0, BLOCK)
        valid = e < E                                        # BLOCK = next pow2 >= E; mask pad lanes
        NEG, POS = float("-inf"), float("inf")
        resident = tl.load(res0_ptr + b * E + e, mask=valid, other=0) != 0
        refresh = tl.load(ref0_ptr + b * E + e, mask=valid, other=0.0)
        tl.store(out_ptr + b * E + e, resident.to(tl.int8), mask=valid)   # out[0] = cold fill
        row = b * E + e
        # 2-deep software pipeline. Only the DECISION is sequential: logits[t] is known before the
        # loop starts and its address never depends on the resident state, so the loads for t+1/t+2
        # are issued two iterations early and their global-memory latency is hidden behind the
        # reductions. Worth up to 1.75x on its own (see temporal/bench_scan.py).
        p1 = tl.load(logits_ptr + (1 * B * E + row), mask=valid & (1 < S), other=0.0).to(tl.float32)
        p2 = tl.load(logits_ptr + (2 * B * E + row), mask=valid & (2 < S), other=0.0).to(tl.float32)
        for t in range(1, S):
            lt = p1
            p1 = p2
            p2 = tl.load(logits_ptr + ((t + 2) * B * E + row), mask=valid & (t + 2 < S),
                         other=0.0).to(tl.float32)
            # Two formulations of the same three selections. Which one wins is a measured property
            # of the reduction width, not a preference: at BLOCK >= 256 one fused value+index
            # reduction beats two dependent ones (~1.25x), and at BLOCK <= 128 it loses (~4%),
            # because the fused combiner shuffles twice per level. Both give torch's first-index
            # tie-breaking, so the mask is identical either way.
            # NSWAPS sub-swaps per token (swap-axis budget; 1 == shipped). static_range unrolls,
            # so NSWAPS=1 compiles to exactly the previous kernel body.
            for _sw in tl.static_range(NSWAPS):
                if BLOCK >= 256:
                    nom_val, nom_i = tl.max(tl.where(valid & (resident == 0), lt, NEG), 0,
                                            return_indices=True)
                    if use_lru:
                        worst_val = tl.min(tl.where(resident, lt, POS), 0)
                        _, evict_i = tl.min(tl.where(resident, refresh, POS), 0,
                                            return_indices=True)
                    else:                   # min_logit: the evict key IS lt, so one reduction does
                        worst_val, evict_i = tl.min(tl.where(resident, lt, POS), 0,  # both
                                                    return_indices=True)
                else:
                    # nominee = argmax over NON-resident logits (first index on ties)
                    masked_nom = tl.where(valid & (resident == 0), lt, NEG)
                    nom_val = tl.max(masked_nom, 0)
                    nom_i = tl.min(tl.where(masked_nom == nom_val, e, BLOCK), 0)
                    # worst resident logit (the swap trigger)
                    worst_val = tl.min(tl.where(resident, lt, POS), 0)
                    # evict: lru -> oldest refresh; min_logit -> lowest logit (first index ties)
                    if use_lru:
                        evict_key = refresh
                    else:
                        evict_key = lt
                    masked_ev = tl.where(resident, evict_key, POS)
                    ev_val = tl.min(masked_ev, 0)
                    evict_i = tl.min(tl.where(masked_ev == ev_val, e, BLOCK), 0)
                do_swap = nom_val > worst_val
                is_evict = (e == evict_i) & do_swap
                is_nom = (e == nom_i) & do_swap
                resident = (resident & (is_evict == 0)) | is_nom
                refresh = tl.where(is_nom, (k + t).to(tl.float32), refresh)  # newest (lru only)
            tl.store(out_ptr + (t * B * E + row), resident.to(tl.int8), mask=valid)


def _scan_num_warps(BLOCK, use_lru):
    """num_warps for the scan, chosen by measurement (temporal/bench_scan.py, H100).

    One program per batch element, so this is the only parallelism knob; B is irrelevant to the
    per-token cost (measured identical at B=1/2/8). More warps shorten the per-thread serial part
    of each reduction but add a shared-memory cross-warp step, and the crossover sits at BLOCK=256.
    Going wide below that is severe (num_warps=4 at BLOCK=64 is 0.46x).
    """
    if BLOCK <= 128:
        return 1
    return 2 if use_lru else 4


def _triton_scan(logits, k, use_lru, swaps=1):
    """Single-launch Triton fast path: cold fill in torch, full t>=1 scan in one kernel."""
    if not _HAS_TRITON:
        raise RuntimeError("triton unavailable")
    S, B, E = logits.shape
    dev = logits.device
    logits = logits.contiguous()
    resident0 = torch.zeros(B, E, dtype=torch.int8, device=dev)   # int8: what the kernel loads
    refresh0 = torch.full((B, E), float("-inf"), device=dev, dtype=torch.float32)
    _, top_i = logits[0].topk(k, dim=-1)
    resident0.scatter_(1, top_i, 1)
    rank = torch.arange(k - 1, -1, -1, device=dev, dtype=torch.float32).expand(B, k)
    refresh0.scatter_(1, top_i, rank)
    out = torch.empty(S, B, E, dtype=torch.int8, device=dev)
    BLOCK = 1 << (E - 1).bit_length()
    _scan_kernel[(B,)](logits, resident0, refresh0, out, S, B, k, E, use_lru, BLOCK, swaps,
                       num_warps=_scan_num_warps(BLOCK, use_lru))
    # every element is written by exactly one program with a 0/1 int8, so this reinterprets the
    # buffer as bool for free -- .to(torch.bool) would cost an extra alloc + launch + full copy.
    return out.view(torch.bool)


def compute_resident_mask_padded(logits, k, starts, evict="lru", swaps=1):
    """Pad-aware scan for LEFT-padded batches: column b's real content begins at starts[b].

    Pure wrapper: each column is rolled so its first real token sits at t=0, the existing
    (verified) scan runs unchanged - so the cold-fill lands on the first REAL token exactly
    as in an unbatched call - and the mask is rolled back. Positions before starts[b] are
    padding; their returned mask rows are whatever the roll wraps around (their router
    outputs are never attended to or scored). Bit-equivalent to scanning each column's
    content alone.
    """
    S, B, E = logits.shape
    starts = starts.to(logits.device).long()                       # [B]
    ar = torch.arange(S, device=logits.device)
    fwd_idx = (ar[:, None] + starts[None, :]) % S                  # [S, B]
    shifted = logits.gather(0, fwd_idx[:, :, None].expand(S, B, E))
    mask = compute_resident_mask_accel(shifted, k, evict=evict, swaps=swaps)
    back_idx = (ar[:, None] - starts[None, :]) % S                 # [S, B]
    return mask.gather(0, back_idx[:, :, None].expand(S, B, E))


def compute_resident_mask_accel(logits, k, evict="lru", tau=0.0, ema_beta=1.0, swaps=1):
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
            return compute_resident_mask(logits, k, evict, tau=tau, ema_beta=ema_beta, swaps=swaps)
    if not logits.is_cuda or mode == "eager":               # CPU (tests) or explicit opt-out
        if _scan_path is None:
            _scan_path = "eager"
            print("[temporal] scan path: eager")
        with torch.no_grad():
            return compute_resident_mask(logits, k, evict, swaps=swaps)
    if swaps != 1 and mode == "graph":
        raise RuntimeError("swaps>1 is implemented for the triton scan only; unset TEMPORAL_SCAN")
    # GPU fast path — correct or crash; no fallback.
    with torch.no_grad():
        out, pathname = ((_graph_scan(logits, k, evict == "lru"), "cuda-graph") if mode == "graph"
                         else (_triton_scan(logits, k, evict == "lru", swaps), "triton"))
        if _scan_path is None:                              # one-time bit-exactness gate (hard)
            ref = compute_resident_mask(logits, k, evict, swaps=swaps)
            if not torch.equal(out, ref):
                bad = (out != ref).any(dim=-1).sum().item()
                raise RuntimeError(
                    f"[temporal] FAST PATH '{pathname}' DISAGREES WITH REFERENCE on {bad} tokens — "
                    f"aborting (kernel bug; do not trust results). Set TEMPORAL_SCAN=eager to bypass.")
            _scan_path = pathname
            print(f"[temporal] scan path: {pathname} (verified == reference)")
    return out


# ---------------------------------------------------------------------------
# Decode-time single-step fast path (serving, S == 1).
#
# The scan fast paths above all take a whole sequence [S,B,E]. Generation through
# a KV cache sees ONE token per forward, so decode_state.step calls the eager
# `_step` once per MoE layer per token: ~16 tiny kernels on a [B,E] tensor,
# ~700 launches per decode step on a 36-layer model, which measured 5.5 ms of
# pure launch overhead per step and costs constrained arms 3-6x throughput
# against free ones (worst on the SMALLEST model -- the signature of a fixed
# per-step cost). This captures the same `_step` into a CUDA graph keyed by
# (B,E,dtype,policy) and replays it: 1 graph launch + 2 state copies per layer.
#
# Identical by construction, not by reimplementation: the graph records the very
# ops the reference issues, in order. `step_accel` still gates on a bit-exactness
# check against eager the first time each shape is used, and RAISES HARD on any
# mismatch (house rule: a correctness bug crashes loudly rather than degrading).
# ---------------------------------------------------------------------------
_step_graph_cache = {}
_step_path = None


# Cache-conditional experts (Skliar et al., arXiv:2412.00099), serving side.
# Their method adds a fixed bonus to experts already in memory so close calls break
# toward what is already loaded. In this scoring rule that is exactly a swap deadband:
# evict only when the best non-resident logit beats the worst resident one by more than
# RHO. 0.0 is our published min_logit rule, bit-identical -- the comparison is unchanged
# when the bonus is zero, so every prior row stands. Read once at import so the captured
# CUDA graph and the eager reference always agree (the bit-exactness gate in step_accel
# checks this). The same env name is used by the training-side router for the same
# concept; the two paths never run in one process.
_RHO = float(os.environ.get("TEMPORAL_RHO", "0"))


def _minlogit_step(lt, resident):
    """`_step` specialized to min_logit eviction (use_lru=False), refresh dropped.

    For min_logit the evict key IS the logit tensor, and refresh is only ever read
    by the LRU branch. decode carries tval=0 and refresh starts at zeros, so
    `torch.where(nominee, 0, refresh)` is the identity on a zero tensor: refresh
    is provably dead state. Dropping it removes two per-call device copies and one
    kernel from the captured body. Equality with `_step` is asserted in step_accel."""
    E = lt.shape[-1]
    NEG, POS = float("-inf"), float("inf")
    nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)
    worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)
    do_swap = (nom_val > worst_val + _RHO).unsqueeze(-1)   # _RHO=0 -> published rule
    evict_i = lt.masked_fill(~resident, POS).argmin(dim=-1)
    evicted = F.one_hot(evict_i, E).bool() & do_swap
    nominee = F.one_hot(nom_i, E).bool() & do_swap
    return (resident & ~evicted) | nominee


def _graph_step(lt, resident):
    """One min_logit `_step` via a replayed CUDA graph. Returns the new resident
    mask (a view of the graph's static output buffer -- callers must consume or
    clone it before the next replay of the same shape)."""
    B, E = lt.shape
    key = (B, E, lt.dtype)
    g = _step_graph_cache.get(key)
    if g is None:
        lt_s = torch.zeros(B, E, device=lt.device, dtype=lt.dtype)
        res_s = torch.zeros(B, E, dtype=torch.bool, device=lt.device)
        torch.cuda.synchronize()
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                res_s.copy_(_minlogit_step(lt_s, res_s))
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res_s.copy_(_minlogit_step(lt_s, res_s))
        g = (lt_s, res_s, graph)
        _step_graph_cache[key] = g
    lt_s, res_s, graph = g
    lt_s.copy_(lt); res_s.copy_(resident)
    graph.replay()
    return res_s


def step_accel(lt, resident, refresh, use_lru=False):
    """Decode-time `_step` with the CUDA-graph fast path (TEMPORAL_DECODE=graph,
    the default on CUDA; "eager" forces the reference). tval is zeros: min_logit
    eviction ignores refresh, and the LRU variant is not served from here."""
    global _step_path
    # DEFAULT GRAPH as of 2026-08-24: validated bit-exact and a real end-to-end
    # win on three architecturally distinct models (OLMoE, qwen3.5-35B,
    # gpt-oss-20b/MXFP4) -- baseline run twice for the noise floor, fast path
    # once, generations 100% textually identical in every case, wall-clock
    # -13% to -18%. TODO.md section 6 has the full protocol and numbers. Every
    # row in the grid before this date was produced on the eager path; opt back
    # into that with TEMPORAL_DECODE=eager if a future comparison needs it.
    mode = os.environ.get("TEMPORAL_DECODE", "graph")
    tval = torch.zeros((), device=lt.device, dtype=lt.dtype)
    if not lt.is_cuda or mode == "eager" or use_lru:
        if _step_path is None:
            _step_path = "eager"
            print("[temporal] decode step path: eager")
        return _step(lt, resident, refresh, tval, use_lru)
    if _step_path is None:                       # one-time bit-exactness gate (hard)
        probe = _graph_step(lt, resident).clone()
        ref, _ = _step(lt, resident, refresh, tval, use_lru)
        if not torch.equal(probe, ref):
            bad = (probe != ref).any(dim=-1).sum().item()
            raise RuntimeError(
                "[temporal] DECODE FAST PATH DISAGREES WITH REFERENCE on "
                f"{bad} rows — aborting (do not trust results). "
                "Set TEMPORAL_DECODE=eager to bypass.")
        _step_path = "cuda-graph"
        print("[temporal] decode step path: cuda-graph (verified == reference)")
    # refresh is dead state under min_logit (see _minlogit_step): passed through
    return _graph_step(lt, resident).clone(), refresh


def step_accel_mask(lt, resident):
    """min_logit `_step` returning the resident mask alone (no refresh), for the
    slotted serving walker. Same env switch and same one-time hard equality gate
    as `step_accel`; returns a fresh tensor (safe to index_copy_ into a bank)."""
    global _step_path
    if not lt.is_cuda or os.environ.get("TEMPORAL_DECODE", "graph") == "eager":
        if _step_path is None:
            _step_path = "eager"
            print("[temporal] decode step path: eager")
        return _minlogit_step(lt, resident)
    if _step_path is None:                       # one-time bit-exactness gate (hard)
        probe = _graph_step(lt, resident).clone()
        ref = _minlogit_step(lt, resident)
        if not torch.equal(probe, ref):
            raise RuntimeError(
                "[temporal] DECODE FAST PATH DISAGREES WITH REFERENCE on "
                f"{(probe != ref).any(dim=-1).sum().item()} rows — aborting.")
        _step_path = "cuda-graph"
        print("[temporal] decode step path: cuda-graph (verified == reference)")
    return _graph_step(lt, resident).clone()


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
        logits = ab.gate_momentum_scores(
            logits, mom_beta,
            gamma_m=float(os.environ.get("TEMPORAL_MOM_GAMMA", "0.125")),
            gamma_q=float(os.environ.get("TEMPORAL_MOM_GAMMA_Q", "0.015625")))
    auxfree = bool(getattr(self, "enable_expert_bias", False)) and getattr(self, "expert_bias", None) is not None
    trig = ab.auxfree_trigger_scores(logits, self.expert_bias).to(logits.dtype) if auxfree else logits
    if mom_beta > 0 and mom_apply == "trigger":
        probs = torch.softmax(logits.float(), dim=-1)
        trig = ab.momentum_shaped_scores(trig, probs, mom_beta,
                                         float(os.environ.get("TEMPORAL_MOM_GAMMA", "0.125")),
                                         alpha_m=float(os.environ.get("TEMPORAL_MOM_ALPHA", "0")),
                                         gamma_q=float(os.environ.get("TEMPORAL_MOM_GAMMA_Q", "0.015625")),
                                         mode=os.environ.get("TEMPORAL_MOM_MODE", "add"))
    head_lam = float(os.environ.get("HEAD_LAMBDA", "0"))
    head_beta = float(os.environ.get("HEAD_BETA", "0"))
    head_logits = None
    if (head_lam > 0 or head_beta > 0) and getattr(self, "nom_head_weight", None) is not None:
        head_logits = ab.nomination_head_logits(input, self.nom_head_weight)
        if head_beta > 0:
            # HEAD_FORCE_ACTIVE=1: eval-only screens on a trained checkpoint (fresh process has
            # no curr_iteration, which would silently disable the bonus). Never set in training.
            if os.environ.get("HEAD_FORCE_ACTIVE", "0") == "1":
                active = True
            else:
                from megatron.training import get_args
                targs = get_args()
                active = ab.head_selection_active(int(getattr(targs, "curr_iteration", 0) or 0),
                                                  int(getattr(targs, "train_iters", 0) or 0),
                                                  float(os.environ.get("HEAD_WARMUP_FRAC", "0.25")))
            if active:
                if os.environ.get("HEAD_CENTER", "0") == "1":
                    bonus = ab.head_centered_bonus(head_logits, head_beta,
                                                   float(os.environ.get("HEAD_GAMMA_C", "0.015625")))
                else:
                    bonus = ab.head_trigger_bonus(head_logits, head_beta)
                trig = (trig.float() + bonus).to(trig.dtype)
    # R-knob (de-lexicalization dose): residency-set size R >= k, decoupled from top-k. The cache
    # holds R experts (cold fill = top-R, same <=1 swap/token trigger/evict on the R-set) and the
    # router selects top-k AMONG residents. R=k (default) == shipped maximal constraint; R=E ==
    # unconstrained full MoE (mask all-True, masked_fill a no-op). Zero FLOP change at any R.
    resid_R = int(os.environ.get("TEMPORAL_RESIDENCY_R", "0")) or k
    # Per-layer schedule, for the cross-regime swap sweep (X2 / test C3) and any prefix schedule.
    # TEMPORAL_R_SCHEDULE is a comma-separated list of <layer>:<R> overriding the global R at those
    # layers only; R may be the literal 'E' for unconstrained. Layers are Megatron's layer_number,
    # matching the `layer` column of every mechinterp CSV. Examples:
    #   TEMPORAL_R_SCHEDULE=4:E                  free layer 4, constrain the rest  (unmask one layer)
    #   TEMPORAL_RESIDENCY_R=0 with R=E default and 4:6   constrain layer 4 only   (impose one layer)
    # FLOPs are unchanged at any R, at any layer, so every arm stays compute-matched.
    sched = os.environ.get("TEMPORAL_R_SCHEDULE", "").strip()
    if sched:
        E = logits.shape[-1]
        ln = int(getattr(self, "layer_number", -1))
        for item in sched.split(","):
            if not item.strip():
                continue
            lay, _, val = item.partition(":")
            if int(lay) == ln:
                resid_R = E if val.strip().upper() == "E" else int(val)
                break
    # N1 sham control. TEMPORAL_SHAM=random replaces the residency mask with a resident set drawn
    # uniformly at random per token: the same R experts are eligible, but the choice carries no
    # lexical information and no temporal dynamics. It answers whether the per-layer cost profile is
    # about routing at all, or is positional sensitivity that any perturbation of that layer would
    # show. Deliberately bypasses compute_resident_mask_accel rather than adding a policy to it: the
    # fast path asserts bit-exactness against the reference scan, and a sham has no reference.
    #
    # Seeded from the layer number and a fixed base so repeated evaluations are identical -- C3's
    # whole value is that it carries no seed noise, and a sham arm has to inherit that.
    sham = os.environ.get("TEMPORAL_SHAM", "")
    if sham:
        E = logits.shape[-1]
        if resid_R >= E:
            mask = torch.ones_like(logits, dtype=torch.bool)
        elif sham == "noise":
            # Magnitude-matched sham (N1, round 2). The random-resident sham above has no size knob:
            # its damage is fixed by R and lands 2.11x the real constraint's, so comparing endpoint
            # ratios between them assumes the profile scales linearly in perturbation size. This adds
            # Gaussian noise to the router logits instead and imposes no residency at all, giving a
            # continuous sigma that can be calibrated until the mean CE penalty matches the real
            # constraint's. It carries no lexical information and no temporal dynamics, which is what
            # a sham has to establish.
            sigma = float(os.environ.get("TEMPORAL_SHAM_SIGMA", "0"))
            # TEMPORAL_SHAM_LAYER restricts the noise to one layer, mirroring impose_one so the two
            # profiles are comparable arm for arm. Empty means every MoE layer.
            _only = os.environ.get("TEMPORAL_SHAM_LAYER", "")
            if _only and int(_only) != int(getattr(self, "layer_number", 0)):
                sigma = 0.0
            g = torch.Generator(device=logits.device)
            g.manual_seed(int(os.environ.get("TEMPORAL_SHAM_SEED", "1234"))
                          + 1009 * int(getattr(self, "layer_number", 0)))
            noise = torch.randn(logits.shape, generator=g, device=logits.device,
                                dtype=torch.float32).to(logits.dtype)
            logits = logits + sigma * noise
            mask = torch.ones_like(logits, dtype=torch.bool)
        elif sham == "random":
            g = torch.Generator(device=logits.device)
            g.manual_seed(int(os.environ.get("TEMPORAL_SHAM_SEED", "1234"))
                          + 1009 * int(getattr(self, "layer_number", 0)))
            r = torch.rand(logits.shape, generator=g, device=logits.device, dtype=torch.float32)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, r.topk(resid_R, dim=-1).indices, True)
        else:
            raise ValueError(f"unknown TEMPORAL_SHAM={sham!r} (expected 'random' or 'noise')")
    else:
        mask = compute_resident_mask_accel(
            trig, resid_R, evict=os.environ.get("TEMPORAL_EVICT", "lru"),
            tau=float(os.environ.get("TEMPORAL_RHO", "0")),
            ema_beta=float(os.environ.get("TEMPORAL_EMA_BETA", "1.0")))
    lam = float(os.environ.get("TEMPORAL_COHERENCE_LAMBDA", "0"))
    if lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        coh = ab.coherence_bce_loss(logits, mask)
        logits = MoEAuxLossAutoScaler.apply(logits, lam * coh)   # inject grad onto raw logits
        save_to_aux_losses_tracker("coherence_loss", coh.detach(),
                                   self.layer_number, self.config.num_layers)
    bw_lam = float(os.environ.get("BURSTY_LAMBDA", "0"))
    if bw_lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        bw = ab.bursty_window_loss(logits, int(os.environ.get("BURSTY_WINDOW", "32")))
        logits = MoEAuxLossAutoScaler.apply(logits, bw_lam * bw)
        save_to_aux_losses_tracker("bursty_loss", bw.detach(),
                                   self.layer_number, self.config.num_layers)
    # CoSMoEs (Huber et al., arXiv:2503.00245) -- BASELINE_METHODS_COMPARISON baseline #1.
    # Penalises how many DISTINCT experts a fixed-length block uses. Nothing is forbidden,
    # switching is only made expensive, so unlike rolling residency it never BOUNDS the
    # resident set -- which is the whole comparison: their locality buys a smaller R, ours
    # buys R = k. Appendix E currently rebuts CoSMoEs by quoting their table rather than
    # running it, which that document calls the most attackable move in the paper.
    # Injected exactly like its siblings above so it composes with the same sweep tooling.
    cos_lam = float(os.environ.get("COSMOES_LAMBDA", "0"))
    if cos_lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        cos = ab.cosmoes_block_loss(logits, int(os.environ.get("COSMOES_BLOCK", "32")))
        logits = MoEAuxLossAutoScaler.apply(logits, cos_lam * cos)
        save_to_aux_losses_tracker("cosmoes_loss", cos.detach(),
                                   self.layer_number, self.config.num_layers)
    ant_lam = float(os.environ.get("ANTICIPATORY_LAMBDA", "0"))
    if ant_lam > 0 and self.training:
        from megatron.core.transformer.moe.moe_utils import (
            MoEAuxLossAutoScaler, save_to_aux_losses_tracker)
        gamma = float(os.environ.get("ANTICIPATORY_GAMMA", "0.5"))
        tgt, valid = ab.anticipatory_target(logits, k, gamma)
        ant = ab.anticipatory_bce_loss(logits, tgt, valid)
        logits = MoEAuxLossAutoScaler.apply(logits, ant_lam * ant)
        save_to_aux_losses_tracker("anticipatory_loss", ant.detach(),
                                   self.layer_number, self.config.num_layers)
    head_bce = None
    if head_lam > 0 and self.training and head_logits is not None:
        # BCE(head(h.detach()), discounted-future-demand) — target from the RAW (pre-mask) logits,
        # detached inside anticipatory_target. The loss graph touches ONLY nom_head_weight.
        tgt, valid = ab.anticipatory_target(logits, k, float(os.environ.get("HEAD_GAMMA", "0.5")))
        if os.environ.get("HEAD_TARGET_CENTER", "0") == "1":
            # H3: labels centered on each expert's own baseline — popularity unlearnable.
            tgt = ab.centered_demand_labels(tgt, float(os.environ.get("HEAD_GAMMA_C", "0.015625")))
        head_bce = ab.anticipatory_bce_loss(head_logits, tgt, valid)
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
