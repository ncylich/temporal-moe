#!/usr/bin/env python3
"""V1: SLC (system-level cache) characterization for the floor's copy traffic.

Question: when the benchmark 'instructs a miss' (cycled cold-pool reads, staged
writes), does Apple silicon secretly serve part of the traffic from cache?

Probes (all expert-sized gathers, fine geometry: 16 x 663,552 B per op):
  A. READ footprint sweep, recycled writes: gather from a cold pool of size S
     (S = 32 MB ... 5.7 GB, cycled windows), letting MLX recycle the one staging
     buffer (the benchmark's current write pattern). If bandwidth falls as S
     exceeds SLC, reads are DRAM-honest at large S; the plateau value is the
     read-side ceiling under cache-absorbed writes.
  B. WRITE rotation sweep at max read footprint: same as A (S = 5.7 GB) but
     retain the last D staged outputs alive so the allocator must rotate write
     targets over D x 10.1 MB of distinct memory. D: 1 (recycle) -> 64 (656 MB
     >> SLC). The drop from D=1 to large D = the write-cache subsidy the
     current benchmark enjoys.
  C. Same-window control: S = 5.7 GB pool but read the SAME 16-expert window
     every op (reuse distance ~10 MB). Should be much faster than cycled reads
     if and only if the cache serves short-reuse reads (sanity that the probe
     can detect caching at all).

Output: JSON. Run standalone; ~1 GB..6 GB transient allocations.
"""
import json
import time
from collections import deque

import mlx.core as mx

EXPERT_B = 663552
N = 16                      # experts per op (floor n16)
CHUNK = N * EXPERT_B        # ~10.1 MB per gather
E_PER_POOL = 192
POOL_B = E_PER_POOL * EXPERT_B   # ~121.5 MB, one fine layer's flat pool


def make_pools(total_bytes):
    n_pools = max(1, total_bytes // POOL_B)
    pools = [mx.random.randint(0, 255, (E_PER_POOL, EXPERT_B), dtype=mx.uint8)
             for _ in range(n_pools)]
    mx.eval(*pools)
    return pools


def run(pools, n_ops, retain=1, same_window=False, group=45):
    """n_ops gathers of N experts, cycling pools and source windows.

    Ops are issued lazily in groups of `group` (one decode token's worth of
    floor copies = 45 layers) with ONE eval per group -- matching the bench's
    dispatch pattern. A per-op eval would swamp the measurement with dispatch
    overhead (~0.3 ms/eval) and read as ~25-40 GB/s regardless of caching."""
    keep = deque(maxlen=retain)
    cyc = 0
    mx.synchronize()
    t0 = time.perf_counter()
    pending = []
    for i in range(n_ops):
        pool = pools[i % len(pools)]
        st = 0 if same_window else (cyc % (E_PER_POOL - N))
        cyc += N
        srcs = mx.arange(st, st + N, dtype=mx.uint32)
        out = mx.take(pool, srcs, axis=0)
        keep.append(out)
        pending.append(out)
        if len(pending) == group:
            mx.eval(*pending)
            pending = []
    if pending:
        mx.eval(*pending)
    mx.synchronize()
    dt = time.perf_counter() - t0
    nb = n_ops * CHUNK
    return {"ms": dt * 1e3, "bytes": nb, "GBps": nb / dt / 1e9}


def main():
    out = {"chunk_MB": CHUNK / 1e6}
    n_ops = 450  # ~10 tokens' worth of floor n16 copy ops (45 layers each)

    # A: read footprint sweep, recycled writes (retain=1)
    for total in (POOL_B, 4 * POOL_B, 12 * POOL_B, 45 * POOL_B):
        pools = make_pools(total)
        run(pools, 90, retain=1)  # warmup
        r = run(pools, n_ops, retain=1)
        out[f"A_read_{len(pools)*POOL_B//2**20}MB_recycledW"] = r
        del pools

    # B: write rotation sweep at full 45-pool footprint
    pools = make_pools(45 * POOL_B)
    for d in (1, 4, 16, 64):
        run(pools, 90, retain=d)
        r = run(pools, n_ops, retain=d)
        out[f"B_write_rotate_{d}buf_{d*CHUNK//2**20}MB"] = r

    # C: same-window reads (short reuse distance) at full footprint
    run(pools, 90, retain=1, same_window=True)
    out["C_same_window_reads"] = run(pools, n_ops, retain=1, same_window=True)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
