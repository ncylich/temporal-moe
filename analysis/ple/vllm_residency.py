#!/usr/bin/env python3
"""Decode-time residency under continuous batching (vLLM glue, engine-agnostic core).

vLLM flattens every scheduled request into one token stream per model step: prefill
chunks (n>1 or a 1-token tail) and single-token decode steps, interleaved, membership
changing step to step. The core here is deliberately tiny: a per-step span list
(req_id, n_tokens, is_prefill) published by a ~10-line runner patch, and `apply()`,
which walks spans over the flattened router logits keying decode_state by
(req_id, layer): prefill spans are observed free (protocol), decode spans get one
reference _step and a mask.

Correctness under scheduling churn is closed-form: state is keyed by request, never by
batch row, so compaction/reordering cannot mix streams; preemption in vLLM v1 discards
KV and replays the prefill, which rebuilds state from scratch, so pruning state for
requests absent from the current step is always safe (done in the runner patch).
test_vllm_walker.py simulates an adversarial schedule and asserts mask equality with
the per-request reference scan.
"""
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_state as DS                                            # noqa: E402
from decode_state import DEC                                         # noqa: E402

STEP = {"spans": None}     # [(req_id, n_tokens, is_prefill)] for the current model step


def set_step(spans):
    STEP["spans"] = spans
    if spans is not None:                       # prune state of departed requests (safe:
        live = {r for r, _, _ in spans}         # preemption replays the prefill)
        for key in [k for k in DEC["state"] if k[0] not in live]:
            del DEC["state"][key]


def apply(layer, router_logits):
    """router_logits [N,E] flattened across spans -> same tensor with non-resident
    experts masked to -inf on DECODE rows only. Prefill rows pass through free."""
    if not DEC["on"] or STEP["spans"] is None:
        return router_logits
    N = router_logits.shape[0]
    total = sum(n for _, n, _ in STEP["spans"])
    assert total == N, f"span/token mismatch: spans cover {total}, logits have {N}"
    out = router_logits.clone()
    o = 0
    for req_id, n, is_prefill in STEP["spans"]:
        key = (req_id, layer)
        lg = router_logits[o:o + n].unsqueeze(1)                     # [n, 1, E]
        if is_prefill:
            DS.observe_chunk(key, lg)
        else:
            assert n == 1, f"decode span with {n} tokens"
            resident = DS.step(key, lg[0])                           # [1, E] bool
            out[o] = out[o].masked_fill(~resident[0], float("-inf"))
        o += n
    return out
