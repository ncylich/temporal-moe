#!/usr/bin/env python3
"""Turn results/runs.jsonl into the morning summary.

Every figure is a ratio to the ceiling measured on THIS device in THIS session, and every
row carries the cache state and clock state it was measured under, because a decode number
without those is not interpretable.
"""
import json, sys
from collections import defaultdict

MIB = 1048576
FILE_MIB = 6707          # qwen3moe-rand-fine-Q4_K_M.gguf
NONEXPERT_MIB = 1239     # file minus 45 layers x 121.5 MiB of expert tensors
PER_TOKEN_ACTIVE_MIB = 513
UFS_MBPS = 1920          # measured cold sequential read


def load(path="results/runs.jsonl"):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def storage_mib(r):
    """Bytes actually read from the block device.

    Prefer the per-process counter; fall back to the device-wide diskstats delta when
    the pid capture missed (the two were cross-validated to 0.1% on the same run, and
    the device is otherwise quiet during a measurement). Returns (mib, source).
    """
    rb = r.get("read_bytes", -1)
    if rb and rb > 0:
        return rb / MIB, "proc"
    sec = r.get("disk_sectors_read", 0)
    if sec:
        return sec * 512 / MIB, "disk"
    return None, None


def fmt(r, ceiling):
    d = r.get("decode_tps")
    ratio = f"{100*d/ceiling:5.1f}%" if (d and ceiling) else "    -"
    mib, src = storage_mib(r)
    read = f"{mib:7.0f}{src[0]}" if mib else "      -"
    clock = ""
    if r.get("clock_min"):
        cm = r["clock_min"]
        k = "7" if "7" in cm else list(cm)[0]
        clock = f"{int(cm[k])/1000:.0f}MHz"
    return (f"  {r['label']:<26} {r['backend']:<15} {r['cache']:<5} "
            f"{(d or 0):7.2f} {ratio}  sd={r.get('decode_sd',0):5.2f}  "
            f"read={read} MiB  res={r.get('resident_pct_before',-1):5.1f}->"
            f"{r.get('resident_pct_after',-1):5.1f}%  {clock:>8}  {r.get('status','')}")


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "results/runs.jsonl")
    if not rows:
        print("no runs recorded yet")
        return

    ceil_rows = [r for r in rows if r["label"].startswith("ceiling") and r.get("decode_tps")]
    ceiling = sum(r["decode_tps"] for r in ceil_rows) / len(ceil_rows) if ceil_rows else None

    # The coarse variant has its OWN ceiling (--mmap 0 on the coarse model). Rating a
    # coarse run against the fine ceiling produced a nonsense "115% of ceiling".
    coarse_ceils = [r["decode_tps"] for r in rows
                    if "coarse" in r["label"] and "mmap0" in r["label"] and r.get("decode_tps")]
    coarse_ceiling = sum(coarse_ceils)/len(coarse_ceils) if coarse_ceils else None
    ceil_p = ([r["prefill_tps"] for r in ceil_rows if r.get("prefill_tps")] or [None])[0]

    print("=" * 118)
    print("SAMSUNG SM-S942U1 (Snapdragon 8 Elite Gen 5) -- temporal-MoE, Qwen3-MoE fine "
          "18-of-192, 10.5B, Q4_K_M, 6.5 GiB")
    print("=" * 118)
    if ceiling:
        spread = max(r["decode_tps"] for r in ceil_rows) - min(r["decode_tps"] for r in ceil_rows)
        print(f"\nCEILING (--mmap 0, weights in anonymous RAM, verified not swapped):")
        print(f"  decode  {ceiling:.2f} tok/s   inter-run spread {spread:.2f} "
              f"({100*spread/ceiling:.1f}%)   n={len(ceil_rows)}")
        if ceil_p:
            print(f"  prefill {ceil_p:.2f} tok/s")
        print(f"  targets: decode >= {0.75*ceiling:.2f} tok/s (75%), "
              f"prefill >= {0.50*ceil_p:.2f} tok/s (50%)" if ceil_p else "")

    print(f"\n{'RUN':<28} {'BACKEND':<15} {'CACHE':<5} {'DECODE':>7} {'RATIO':>6}"
          f"        {'STORAGE READ':>12}     {'RESIDENCY':>12}   {'CLOCK':>8}  STATUS")
    print("-" * 118)
    for r in rows:
        c = coarse_ceiling if ("coarse" in r["label"] and coarse_ceiling) else ceiling
        print(fmt(r, c))
    if coarse_ceiling:
        print(f"\n  (coarse rows rated against the coarse ceiling: {coarse_ceiling:.2f} tok/s)")

    # ---- routing diversity verdict ----------------------------------------
    div = sorted([r for r in rows if r["label"].startswith("diversity")],
                 key=lambda r: r.get("decode_tokens", 0))
    if len(div) >= 2:
        print("\n" + "=" * 118)
        print("ROUTING DIVERSITY (Phase G) -- does the working set grow with decode length?")
        print("-" * 118)
        for r in div:
            n = r.get("decode_tokens", 0)
            rb = (storage_mib(r)[0] or 0)
            expected_fixed = NONEXPERT_MIB + PER_TOKEN_ACTIVE_MIB
            print(f"  n={n:<4} read {rb:7.0f} MiB from storage   "
                  f"(fixed-expert-set prediction ~{expected_fixed} MiB, "
                  f"fully-diverse prediction ~{min(FILE_MIB, NONEXPERT_MIB + n*PER_TOKEN_ACTIVE_MIB)} MiB)")
        first, last = div[0], div[-1]
        f0, l0 = storage_mib(first)[0], storage_mib(last)[0]
        if f0:
            growth = l0 / f0
            tok_growth = last.get("decode_tokens", 1) / max(1, first.get("decode_tokens", 1))
            print(f"\n  reads grew {growth:.2f}x while decode length grew {tok_growth:.0f}x")
            if growth < 1.3:
                print("  => VERDICT: reads PLATEAU. The router reuses a near-fixed expert set,")
                print("     so the working set is ~513 MiB TOTAL, not per token. Any paging win")
                print("     here is 'a small fixed slice stays cached', NOT 'temporal residency")
                print("     works under diverse routing'. Random weights are the likely cause.")
            else:
                print("  => VERDICT: reads GROW with decode length, so routing is genuinely")
                print("     diverse and the paging result reflects real expert turnover.")

    # ---- best configuration ------------------------------------------------
    # This phone cannot hold peak clock for a whole 5-rep run -- the clock sags to ~47%
    # of rated within ~90 s of sustained load, on every run including the ceiling. So a
    # degraded_clock status is the NORMAL sustained-load state here, not a spoiled run,
    # and excluding it would leave nothing to compare. What matters is that the arms are
    # in the SAME state (M35), which is why the clock is printed on every row and the
    # ceiling is measured under identical conditions.
    OK = ("ok", "degraded_clock", "throttled")
    # diversity_* runs used 16/64/256 decode tokens, so their throughput is not
    # comparable to the 128-token protocol; excluded from "best config".
    cand = [r for r in rows if r.get("decode_tps") and not r["label"].startswith("ceiling")
            and not r["label"].startswith("diversity")
            and "mmap0" not in r["label"]
            and str(r.get("status", "")).startswith(OK)]
    if cand and ceiling:
        best = max(cand, key=lambda r: r["decode_tps"])
        # rate against the ceiling of the SAME model variant
        is_coarse = "coarse" in best["label"]
        ceiling = coarse_ceiling if (is_coarse and coarse_ceiling) else ceiling
        cp = [r["prefill_tps"] for r in rows if "coarse" in r["label"]
              and "mmap0" in r["label"] and r.get("prefill_tps")]
        ceil_p = (sum(cp)/len(cp)) if (is_coarse and cp) else ceil_p
        print("\n" + "=" * 118)
        print(f"BEST NON-CEILING CONFIG: {best['label']} ({best['backend']}, "
              f"{best['cache']} cache)")
        print(f"  decode {best['decode_tps']:.2f} tok/s = "
              f"{100*best['decode_tps']/ceiling:.1f}% of ceiling "
              f"[target 75%: {'MET' if best['decode_tps'] >= 0.75*ceiling else 'NOT MET'}]")
        if best.get("prefill_tps") and ceil_p:
            print(f"  prefill {best['prefill_tps']:.2f} tok/s = "
                  f"{100*best['prefill_tps']/ceil_p:.1f}% of ceiling "
                  f"[target 50%: {'MET' if best['prefill_tps'] >= 0.50*ceil_p else 'NOT MET'}]")
        print(f"  measured with page cache {best.get('resident_pct_before',-1):.1f}% resident "
              f"at start, {(storage_mib(best)[0] or 0):.0f} MiB actually read from storage")

    # ---- roofline sanity (M31) --------------------------------------------
    print("\n" + "=" * 118)
    print("ROOFLINE SANITY (M31)")
    print(f"  active expert weights per token: {PER_TOKEN_ACTIVE_MIB} MiB")
    print(f"  measured UFS cold sequential:    {UFS_MBPS/1000:.2f} GB/s")
    print(f"  => decode is capped at {UFS_MBPS/PER_TOKEN_ACTIVE_MIB:.2f} tok/s IF every active "
          f"expert is fetched from storage each token.")
    print("  Any decode above that is only possible with experts already resident in RAM.")
    for r in rows:
        d = r.get("decode_tps") or 0
        rb = storage_mib(r)[0]
        if d > UFS_MBPS / PER_TOKEN_ACTIVE_MIB and rb:
            implied = d * PER_TOKEN_ACTIVE_MIB / 1000
            print(f"    {r['label']}: {d:.1f} tok/s implies {implied:.1f} GB/s expert traffic "
                  f"-> served from RAM, not storage")
            break


if __name__ == "__main__":
    main()
