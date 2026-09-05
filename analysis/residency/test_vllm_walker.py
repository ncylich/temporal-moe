#!/usr/bin/env python3
"""Continuous-batching parity: an adversarial synthetic schedule through the span walker
must give every request exactly the masks the reference scan gives it in isolation.

Schedule stresses: chunked prefill split across steps, requests joining mid-flight,
interleaved decode, one request finishing early, one preempted (vanishes, then replays
its prefill). Pure CPU, no vLLM, no GPU -- runs in milliseconds.
"""
import torch

import vllm_residency as VR
from decode_state import DEC, compute_resident_mask

R, E = 8, 64
g = torch.Generator().manual_seed(11)
# per-request logit streams: prompt_len, total_len
REQS = {"a": (17, 33), "b": (9, 29), "c": (23, 31)}
STREAM = {r: torch.randn(t, 1, E, generator=g) for r, (_, t) in REQS.items()}


def reference(r):
    """Free prompt observed, rule enforced on decode positions: masks for t >= prompt."""
    p, t = REQS[r]
    full = compute_resident_mask(STREAM[r], R, evict="min_logit", swaps=1)
    return full[p:]


# step -> list of (req, [span-start, span-end), is_prefill); token positions per request
SCHEDULE = [
    [("a", 0, 10, True)],                                    # a: prefill chunk 1
    [("a", 10, 17, True), ("b", 0, 9, True)],                # a finishes prefill; b joins
    [("a", 17, 18, False), ("b", 9, 10, False)],             # both decode
    [("a", 18, 19, False), ("b", 10, 11, False), ("c", 0, 15, True)],   # c joins mid-flight
    [("a", 19, 20, False), ("b", 11, 12, False), ("c", 15, 23, True)],
    [("a", 20, 21, False), ("c", 23, 24, False)],            # b preempted (vanishes)
    # b resumes: vLLM re-prefills prompt PLUS the 3 tokens it had generated (0..11),
    # then decode continues at 12. The re-prefilled generated positions are observed
    # free on replay -- state updates are identical to the enforced steps they replace
    # (observe and step share _step), so later masks still match the reference.
    [("b", 0, 12, True), ("a", 21, 22, False), ("c", 24, 25, False)],
]
# then run everyone to completion, one decode token per step
mx = max(t for _, t in REQS.values())
pos = {"a": 22, "b": 12, "c": 25}
for _ in range(mx):
    step = []
    for r in ("a", "b", "c"):
        if pos[r] < REQS[r][1]:
            step.append((r, pos[r], pos[r] + 1, False))
            pos[r] += 1
    if step:
        SCHEDULE.append(step)

# the fast walker is GPU-only (Triton); its buffers live on the current CUDA device
import os
DEV = "cuda" if os.environ.get("TEMPORAL_WALKER", "fast") == "fast" and torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    os.environ["TEMPORAL_WALKER"] = os.environ.get("TEMPORAL_WALKER", "slots")
    if VR._WALKER == "fast":
        VR._WALKER = "slots"
DEC.update(on=True, R=R, swaps=1)
DEC["state"].clear()
got = {r: {} for r in REQS}                     # req -> {position: mask row}
for step in SCHEDULE:
    spans = [(r, e - s, pf) for r, s, e, pf in step]
    VR.set_step(spans)
    flat = torch.cat([STREAM[r][s:e, 0] for r, s, e, _ in step]).to(DEV)
    out = VR.apply(0, flat.clone()).cpu()
    o = 0
    for r, s, e, pf in step:
        if not pf:
            row = out[o]
            got[r][s] = torch.isfinite(row)     # resident = not masked to -inf
        o += e - s

for r, (p, t) in REQS.items():
    ref = reference(r)
    for pos_i, m in sorted(got[r].items()):
        want = ref[pos_i - p, 0]
        assert torch.equal(m, want), f"req {r} pos {pos_i} mask mismatch"
n = sum(len(v) for v in got.values())
print(f"CONTINUOUS-BATCHING WALKER PARITY PASS [{VR._WALKER} walker on {DEV}] ({n} decode positions, "
      f"chunked/join/finish/preempt-replay all exercised)")
