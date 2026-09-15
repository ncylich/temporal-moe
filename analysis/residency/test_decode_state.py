#!/usr/bin/env python3
"""Parity: the stateful decode path must reproduce the reference batch scan exactly.

Given the same logit stream, (a) prefill-observe on a prefix + token-by-token steps must
equal the full-sequence scan position-for-position after the prefix, and (b) cold-start
steps from t=0 must equal the full scan everywhere. Run on cpu and cuda.
"""
import torch

import decode_state as DS
from decode_state import DEC, compute_resident_mask


def check(device):
    g = torch.Generator().manual_seed(7)
    S, B, E, R = 96, 1, 64, 8
    logits = torch.randn(S, B, E, generator=g).to(device)
    ref = compute_resident_mask(logits, R, evict="min_logit", swaps=1)

    for p in (32, 1):                                   # prefill length (1 == just the cold fill)
        DEC.update(on=True, R=R, swaps=1)
        DS.reset()
        DS.prefill(0, logits[:p])
        got = [DS.step(0, logits[t]) for t in range(p, S)]
        got = torch.stack(got)
        assert torch.equal(got, ref[p:]), f"prefix p={p} mismatch on {device}"

    DS.reset()                                          # cold start: no prefill at all
    got = torch.stack([DS.step(0, logits[t]) for t in range(S)])
    assert torch.equal(got, ref), f"cold-start mismatch on {device}"

    # chunked prefill observe (vLLM path): observing [0:13],[13:40],[40:64] then stepping
    # must equal a single full-prefix prefill at 64.
    DS.reset()
    for a, b in ((0, 13), (13, 40), (40, 64)):
        DS.observe_chunk(0, logits[a:b])
    got = torch.stack([DS.step(0, logits[t]) for t in range(64, S)])
    assert torch.equal(got, ref[64:]), f"chunked-observe mismatch on {device}"
    print(f"  parity OK on {device} (S={S}, E={E}, R={R}, prefixes 32/1/cold/chunked)")


if __name__ == "__main__":
    check("cpu")
    if torch.cuda.is_available():
        check("cuda")
    print("DECODE-STATE PARITY PASS")
