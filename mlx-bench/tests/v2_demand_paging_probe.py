#!/usr/bin/env python3
"""V2: demand-paging / mmap-advisory verification (RESULTS.md cache appendix).

Claim under test (raised in review): macOS keeps frequently reused disk-backed
(mmap'd) pages resident in RAM regardless of advisory flags (F_NOCACHE,
madvise) — so a "disk-offloaded" cold expert pool on a Mac silently becomes a
RAM pool whenever it fits, and a naive mmap/SSD floor measurement would be
page-cache-flattered.

Phases (expert-sized reads: 663,552 B chunks, cycled offsets, like the floor):
  P1  RAM-fitting pool file (5.7 GB = the fine model's flat cold pools):
      a) warm mmap read bandwidth (page-cache regime),
      b) after madvise(MADV_DONTNEED) on the mapping — advisory eviction,
      c) pread via an F_NOCACHE fd — advisory cache bypass,
      d) after real memory pressure (touch ~17 GB of random anonymous pages,
         the only non-sudo eviction lever) — first re-read.
      If (b) and (c) still run at RAM speeds, the flags are demonstrably
      suggestions: a RAM-fitting "SSD offload" measures as a RAM pool -> our
      RAM-pool floor is the floor macOS actually gives this regime.
  P2  Bigger-than-RAM pool file (30 GB): cycled expert reads over the whole
      file; the page cache cannot hold it, so steady-state = true SSD misses.
      This is the regime where a Mac floor re-deepens.
  P3  Projection: implied floor_n16 tok/s under each measured bandwidth,
      using the bench's measured no-copy baseline.

Files are created under the session scratchpad (pass as argv[1]); deleted at
the end. Total disk use ~36 GB transient. Do NOT run concurrently with GPU
benchmarks (page-cache churn + memory pressure would contaminate them).
"""
import json
import mmap
import os
import sys
import time

CHUNK = 663552
E = 192
POOL_B = E * CHUNK                    # one layer's flat pool, ~121.5 MB
F_NOCACHE = 48                        # fcntl.h on macOS


def write_file(path, nbytes):
    """Random (incompressible) file, written in 128 MB strides."""
    blk = os.urandom(1 << 27)
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        left = nbytes
        while left > 0:
            n = min(left, len(blk))
            f.write(blk[:n] if n != len(blk) else blk)
            left -= n
        f.flush()
        os.fsync(f.fileno())
    return time.perf_counter() - t0


def read_mmap(path, nbytes, read_bytes, evict_first=False):
    """Cycled expert-sized reads through an mmap; returns GB/s (+checksum)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, nbytes, prot=mmap.PROT_READ)
        if evict_first:
            try:
                mm.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError) as e:
                print(f"  madvise(DONTNEED) failed: {e}", file=sys.stderr)
        n_chunks = nbytes // CHUNK
        acc = 0
        done = 0
        i = 0
        t0 = time.perf_counter()
        while done < read_bytes:
            off = (i * 7919 % n_chunks) * CHUNK   # cycled, stride-7919 chunks
            acc += mm[off]
            acc += mm[off + CHUNK - 1]
            _ = mm[off:off + CHUNK]               # touch the full chunk
            done += CHUNK
            i += 1
        dt = time.perf_counter() - t0
        mm.close()
        return read_bytes / dt / 1e9, acc
    finally:
        os.close(fd)


def read_nocache(path, nbytes, read_bytes):
    """pread through an F_NOCACHE fd (advisory bypass); returns GB/s."""
    import fcntl
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
        n_chunks = nbytes // CHUNK
        done = 0
        i = 0
        t0 = time.perf_counter()
        while done < read_bytes:
            off = (i * 7919 % n_chunks) * CHUNK
            b = os.pread(fd, CHUNK, off)
            done += len(b)
            i += 1
        dt = time.perf_counter() - t0
        return read_bytes / dt / 1e9
    finally:
        os.close(fd)


def memory_pressure(gb):
    """Touch `gb` of RANDOM anonymous pages (incompressible, so the memory
    compressor can't absorb them) to force file-page eviction; then free."""
    blocks = []
    blk = os.urandom(1 << 27)  # 128 MB random template
    try:
        for _ in range(int(gb * 8)):
            blocks.append(bytearray(blk))   # copy -> distinct dirty pages
    except MemoryError:
        pass
    n = len(blocks) / 8
    del blocks
    return n


def main():
    scratch = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    small = os.path.join(scratch, "v2_pool_ramfit.bin")
    big = os.path.join(scratch, "v2_pool_bigger_than_ram.bin")
    out = {}

    small_b = 45 * POOL_B                      # 5.72 GB
    print(f"P1: writing {small_b/2**30:.1f} GB pool file ...", file=sys.stderr)
    write_file(small, small_b)
    read = 4 * 2**30
    g, _ = read_mmap(small, small_b, read)
    out["P1a_warm_mmap_GBps"] = g
    g, _ = read_mmap(small, small_b, read, evict_first=True)
    out["P1b_after_madvise_DONTNEED_GBps"] = g
    out["P1c_F_NOCACHE_pread_GBps"] = read_nocache(small, small_b, read)
    print("P1d: applying ~17 GB memory pressure ...", file=sys.stderr)
    out["P1d_pressure_applied_GB"] = memory_pressure(17)
    g, _ = read_mmap(small, small_b, 2 * 2**30)
    out["P1d_first_reread_after_pressure_GBps"] = g
    g, _ = read_mmap(small, small_b, read)
    out["P1d_second_reread_GBps"] = g
    os.unlink(small)

    big_b = 30 * 2**30
    print(f"P2: writing {big_b/2**30:.0f} GB file ...", file=sys.stderr)
    write_file(big, big_b)
    g, _ = read_mmap(big, big_b, 2 * 2**30)    # pass 1 (partially cache-warm
    out["P2_pass1_GBps"] = g                   #   from the write)
    g, _ = read_mmap(big, big_b, 4 * 2**30)
    out["P2_pass2_steady_GBps"] = g
    os.unlink(big)

    # P3: implied fine floor_n16 from the bench's no-copy baseline (68.6 tok/s
    # = 14.6 ms/token) + 477.8 MB/token of tier reads at each bandwidth.
    base_ms = 14.6
    for name, g in [("ram_pool_measured_bench", None),
                    ("page_cache", out["P1a_warm_mmap_GBps"]),
                    ("ssd_steady", out["P2_pass2_steady_GBps"])]:
        if g is None:
            continue
        ms = base_ms + 477.8e6 / (g * 1e9) * 1e3
        out[f"P3_implied_floor_n16_tok_s_{name}"] = round(1000 / ms, 1)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
