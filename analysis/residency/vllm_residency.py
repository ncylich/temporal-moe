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
    experts masked to -inf on DECODE rows only. Prefill rows pass through free.

    All decode rows advance in ONE batched reference _step (the scan's step is batched
    over B); with hundreds of concurrent requests a per-request loop would be
    kernel-launch-bound -- the exact pathology this stack exists to escape."""
    if not DEC["on"] or STEP["spans"] is None:
        return router_logits
    N = router_logits.shape[0]
    total = sum(n for _, n, _ in STEP["spans"])
    assert total == N, f"span/token mismatch: spans cover {total}, logits have {N}"
    out = router_logits.clone()
    dec_rows, dec_keys, cold, replay = [], [], [], []
    o = 0
    for req_id, n, is_prefill in STEP["spans"]:
        key = (req_id, layer)
        if is_prefill:
            DS.observe_chunk(key, router_logits[o:o + n].unsqueeze(1))
        elif n > 1:
            # preemption replay: vLLM discarded KV and recomputes the generated
            # tokens as one chunk. Logits are deterministic, so stepping the scan
            # through the chunk rebuilds the original per-token schedule.
            replay.append((key, o, n))
        else:
            if key not in DEC["state"]:
                cold.append((key, o))                # rare: no prefill seen (cold fill)
            else:
                dec_rows.append(o)
                dec_keys.append(key)
        o += n
    for key, row in cold:
        resident = DS.step(key, router_logits[row:row + 1])
        out[row] = out[row].masked_fill(~resident[0], float("-inf"))
    for key, row, n in replay:
        start = 0
        if key not in DEC["state"]:                  # state pruned while preempted:
            r0 = DS.step(key, router_logits[row:row + 1])  # re-seed from first token
            out[row] = out[row].masked_fill(~r0[0], float("-inf"))
            start = 1
        resident, refresh = DEC["state"][key]
        with torch.no_grad():
            for j in range(start, n):
                lt = router_logits[row + j:row + j + 1].float()
                for _ in range(DEC["swaps"]):
                    resident, refresh = DS._step(lt, resident, refresh,
                                                 torch.zeros((), device=lt.device),
                                                 use_lru=False)
                out[row + j] = out[row + j].masked_fill(~resident[0], float("-inf"))
        DEC["state"][key] = (resident, refresh)
    if dec_rows:
        with torch.no_grad():
            lt = router_logits[dec_rows].float()                     # [D, E]
            resident = torch.cat([DEC["state"][k][0] for k in dec_keys])
            refresh = torch.cat([DEC["state"][k][1] for k in dec_keys])
            for _ in range(DEC["swaps"]):
                resident, refresh = DS.step_accel(lt, resident, refresh,
                                                  use_lru=False)
        for j, k in enumerate(dec_keys):
            DEC["state"][k] = (resident[j:j + 1], refresh[j:j + 1])
        out[dec_rows] = out[dec_rows].masked_fill(~resident, float("-inf"))
    return out
