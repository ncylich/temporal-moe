#!/usr/bin/env python3
"""Test T2 (TRUNCATION_RERUN_PLAN §3): a resumed residency walk is BIT-identical
to the continuous one.

The resident set is path-dependent: S_t is the product of every swap decision
before t. So a continuation that re-derives its starting set instead of carrying
the true one is not a continuation, it is a different run wearing the same
prefix -- and it would silently contaminate every number downstream. This holds
the splice to bit equality, never approximate equality.

Covered, because these are where eviction order is ambiguous and a resume is most
likely to diverge: exact logit ties, all-equal logits, R == E, R == 1, -inf
entries, and both evict policies (min_logit and lru -- the latter also checks the
refresh clock keeps absolute time across the splice via t0).

Run: python temporal/tests/test_resume_residency.py
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from temporal.temporal_router import compute_resident_mask  # noqa: E402


def splice_equals_continuous(logits, k, evict, cuts, swaps=1):
    """Run continuously, then in chunks carrying state; require bit equality."""
    ref = compute_resident_mask(logits, k, evict=evict, swaps=swaps)
    S = logits.shape[0]
    parts, state, prev = [], {}, 0
    for cut in list(cuts) + [S]:
        if cut <= prev:
            continue
        chunk = logits[prev:cut]
        if prev == 0:
            m = compute_resident_mask(chunk, k, evict=evict, swaps=swaps,
                                      return_state=state)
        else:
            m = compute_resident_mask(chunk, k, evict=evict, swaps=swaps,
                                      init_resident=state["resident"],
                                      init_refresh=state["refresh"], t0=prev,
                                      return_state=state)
        parts.append(m)
        prev = cut
    spliced = torch.cat(parts, dim=0)
    return torch.equal(spliced, ref), ref, spliced


def check(name, logits, k, evict, cuts, swaps=1):
    ok, ref, got = splice_equals_continuous(logits, k, evict, cuts, swaps)
    if not ok:
        bad = (ref != got).any(dim=-1).nonzero()[:5].flatten().tolist()
        print(f"  FAIL {name}: diverges first at positions {bad}")
    else:
        print(f"  ok   {name}")
    return ok


def main():
    torch.manual_seed(20260823)
    ok = True
    S, B, E, k = 96, 4, 32, 8
    cuts = [1, 7, 33, 64, 95]           # incl. cut at 1 and at S-1 (edge cases)

    for evict in ("min_logit", "lru"):
        base = torch.randn(S, B, E)
        ok &= check(f"random, evict={evict}", base, k, evict, cuts)
        ok &= check(f"random, evict={evict}, swaps=3", base, k, evict, cuts, swaps=3)

        # exact ties everywhere: quantization guarantees duplicate max/min
        q = (torch.randn(S, B, E) * 2).round() / 2
        ok &= check(f"quantized ties, evict={evict}", q, k, evict, cuts)

        # degenerate: every logit identical -> every comparison is a tie
        z = torch.zeros(S, B, E)
        ok &= check(f"all-equal logits, evict={evict}", z, k, evict, cuts)

        # R == E: nothing is non-resident, no swap can fire
        ok &= check(f"R == E, evict={evict}", torch.randn(S, B, 8), 8, evict, cuts)

        # R == 1
        ok &= check(f"R == 1, evict={evict}", torch.randn(S, B, E), 1, evict, cuts)

        # -inf entries (masked/absent experts arrive this way upstream)
        inf = torch.randn(S, B, E)
        inf[:, :, ::7] = float("-inf")
        ok &= check(f"-inf entries, evict={evict}", inf, k, evict, cuts)

        # every single cut point, one at a time (the exhaustive check)
        sm = torch.randn(24, 2, 16)
        bad = [t for t in range(1, 24)
               if not splice_equals_continuous(sm, 4, evict, [t])[0]]
        print(f"  {'ok  ' if not bad else 'FAIL'} every cut point 1..23, "
              f"evict={evict}" + (f" -- diverged at {bad}" if bad else ""))
        ok &= not bad

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
