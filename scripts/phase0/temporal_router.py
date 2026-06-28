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


def compute_resident_mask(logits: torch.Tensor, k: int, evict: str = "lru") -> torch.Tensor:
    """Rolling-residency expert selection.

    Args:
        logits: [seq, batch, num_experts] router logits (seq-first, as Megatron's router sees them).
        k: resident-set size (= top-k; K = k for this PoC).
        evict: which resident to remove when a swap happens (experiment knob, same swap *trigger*):
            "lru"       — oldest last-refresh time (cache-style; protects just-loaded experts from
                          immediate re-eviction → less thrash; score-neutral w.r.t. the aux loss).
            "min_logit" — lowest current logit, i.e. the same "worst resident" the swap trigger
                          compares against (most consistent; quality-greedy; simpler).

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

    refresh = torch.full((B, E), NEG, device=dev)           # last-refresh time per expert ("lru" only)
    out = torch.zeros(S, B, E, dtype=torch.bool, device=dev)

    # --- t=0 cold fill: R_0 = top-k(logits[0]) ---
    resident = torch.zeros(B, E, dtype=torch.bool, device=dev)
    _, top_i = logits[0].topk(k, dim=-1)                    # [B,k], descending by logit
    resident.scatter_(1, top_i, True)
    # highest logit -> newest (largest refresh); lowest of the k -> oldest (0).
    rank_refresh = torch.arange(k - 1, -1, -1, device=dev).float().expand(B, k)
    refresh.scatter_(1, top_i, rank_refresh)
    out[0] = resident

    for t in range(1, S):
        lt = logits[t]                                      # token t pulls in one expert and uses it
        nom_val, nom_i = lt.masked_fill(resident, NEG).max(dim=-1)      # best non-resident [B]
        worst_val, _ = lt.masked_fill(~resident, POS).min(dim=-1)       # worst resident   [B]
        do_swap = (nom_val > worst_val).unsqueeze(-1)                   # [B,1]: R_{t-1} != global top-k
        evict_key = refresh if use_lru else lt
        evict_i = evict_key.masked_fill(~resident, POS).argmin(dim=-1)  # resident to remove [B]
        evicted = F.one_hot(evict_i, E).bool() & do_swap               # [B,E]
        nominee = F.one_hot(nom_i, E).bool() & do_swap                 # [B,E]
        resident = (resident & ~evicted) | nominee
        refresh = refresh.masked_fill(nominee, float(k + t))           # newest (read only when "lru")
        out[t] = resident

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
# check vs `compute_resident_mask`) and falls back to eager on any capture failure.
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


def compute_resident_mask_accel(logits, k, evict="lru"):
    """Resident mask with the CUDA-graph fast path; identical result to compute_resident_mask.

    Uses the graph path on CUDA (unless TEMPORAL_SCAN=eager); verifies it equals the reference
    once, then trusts it. Any capture/replay failure logs and falls back to eager for the run.
    """
    global _scan_path
    use_graph = (logits.is_cuda and os.environ.get("TEMPORAL_SCAN", "graph") == "graph"
                 and _scan_path != "eager")
    if use_graph:
        try:
            with torch.no_grad():
                out = _graph_scan(logits, k, evict == "lru")
            if _scan_path is None:                          # one-time correctness gate
                if not torch.equal(out, compute_resident_mask(logits, k, evict)):
                    raise RuntimeError("graph scan != reference")
                _scan_path = "cuda-graph"
                print("[temporal] scan path: cuda-graph (verified == reference)")
            return out
        except Exception as e:
            _scan_path = "eager"
            print(f"[temporal] scan path: EAGER fallback ({type(e).__name__}: {str(e)[:100]})")
    elif _scan_path is None:
        _scan_path = "eager"
        print("[temporal] scan path: eager")
    with torch.no_grad():
        return compute_resident_mask(logits, k, evict)


def temporal_forward(self, input: torch.Tensor):
    """Drop-in replacement for TopKRouter.forward: restrict selection to the resident set.

    Masking non-resident experts to -inf and calling the unmodified self.routing() keeps z-loss,
    aux-loss and the top-k/dispatch path byte-for-byte identical (they just see masked logits).
    """
    input = self.apply_input_jitter(input)
    logits = self.gating(input)                             # [seq, batch, num_experts]
    k = self.config.moe_router_topk
    mask = compute_resident_mask_accel(logits, k, evict=os.environ.get("TEMPORAL_EVICT", "lru"))
    logits = logits.masked_fill(~mask, float("-inf"))       # only resident experts are selectable
    return self.routing(logits)


def install():
    """Monkeypatch TopKRouter.forward (call once at startup, before model build)."""
    from megatron.core.transformer.moe.router import TopKRouter
    TopKRouter.forward = temporal_forward
    print(f"[temporal] rolling-residency router installed (evict={os.environ.get('TEMPORAL_EVICT', 'lru')})")
