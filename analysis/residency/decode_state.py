#!/usr/bin/env python3
"""Stateful decode-time residency: the rolling rule carried across generate() forwards.

Teacher-forced scoring sees all router logits in one forward, so the batch scan suffices.
Real generation runs through the KV cache -- each decode forward sees ONE token's logits --
so the resident set must persist across forwards. This module holds that state and applies
the instruct-serving protocol:

    prefill  (S > 1): FREE routing (no mask). The verified batch scan runs observe-only over
                      the prompt logits and its final resident set seeds the decode state.
                      A prefill also RESETS the layer's state, so back-to-back generate()
                      calls need no external bookkeeping.
    decode   (S == 1): one reference `_step` per token (the exact loop body of
                      compute_resident_mask, reused by import -- not a reimplementation),
                      then the token routes inside the returned resident set.

min_logit eviction only (the program's policy; refresh state is an lru concept and is
carried as zeros purely to satisfy _step's signature). Parity with the batch scan on
identical logit streams is asserted by test_decode_state.py.
"""
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal.temporal_router import (  # noqa: E402
    compute_resident_mask, compute_resident_mask_accel, _step, step_accel,
)

DEC = {"on": False, "R": 8, "swaps": 1, "state": {},
       # RESUME support: resume_map keys a submitted prompt (len, hash of head,
       # hash of tail) -> number of leading tokens that are the ORIGINAL prompt
       # and stay free. enforce_from is the per-request boundary the glue resolves
       # from it; tokens at or beyond it are previously-generated and must be
       # walked under the rule, because that is how they were produced.
       "resume_map": {}, "enforce_from": {}}   # state[layer] = (resident, refresh)


def reset():
    DEC["state"].clear()


def prefill(layer, lg):
    """Observe-only over prompt logits lg [S,B,E]; seed state; return None (no mask)."""
    with torch.no_grad():
        scan = compute_resident_mask_accel if lg.is_cuda else compute_resident_mask
        mask = scan(lg.float(), DEC["R"], evict="min_logit", swaps=DEC["swaps"])
    DEC["state"][layer] = (mask[-1].clone(), torch.zeros_like(lg[0], dtype=torch.float))
    return None


def step(layer, lt):
    """One decode token: lt [B,E] raw selection signal. Returns resident bool mask [B,E]."""
    st = DEC["state"].get(layer)
    with torch.no_grad():
        if st is None:                          # cold start (no prefill): scan's t=0 cold fill
            resident = torch.zeros_like(lt, dtype=torch.bool)
            _, top_i = lt.float().topk(DEC["R"], dim=-1)
            resident.scatter_(1, top_i, True)
            refresh = torch.zeros_like(lt, dtype=torch.float)
        else:
            resident, refresh = st
            ltf = lt.float()
            for _ in range(DEC["swaps"]):
                resident, refresh = step_accel(ltf, resident, refresh, use_lru=False)
    DEC["state"][layer] = (resident, refresh)
    return resident


def observe_chunk(layer, lg):
    """Observe a prefill CHUNK lg [T,B,E] without masking, carrying state across chunks
    (vLLM chunked prefill). First chunk token cold-fills iff no state exists (the scan's
    t=0 rule); every other token is one reference _step. Equivalent to running the batch
    scan over the concatenated chunks -- asserted by test_decode_state.py."""
    st = DEC["state"].get(layer)
    t0 = 0
    with torch.no_grad():
        if st is None:
            resident = torch.zeros_like(lg[0], dtype=torch.bool)
            _, top_i = lg[0].float().topk(DEC["R"], dim=-1)
            resident.scatter_(1, top_i, True)
            refresh = torch.zeros_like(lg[0], dtype=torch.float)
            t0 = 1
        else:
            resident, refresh = st
        for t in range(t0, lg.shape[0]):
            for _ in range(DEC["swaps"]):
                resident, refresh = _step(lg[t].float(), resident, refresh,
                                          torch.zeros((), device=lg.device), use_lru=False)
    DEC["state"][layer] = (resident, refresh)


def route(layer, lg):
    """Dispatch on shape: lg [S,B,E]. Returns mask [S,B,E] or None (prefill = free)."""
    if not DEC["on"]:
        return None
    if lg.shape[0] > 1:
        return prefill(layer, lg)
    return step(layer, lg[0]).unsqueeze(0)
