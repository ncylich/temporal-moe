#!/usr/bin/env python3
"""Summary of gpubench/overlap/speed.sh or pinned.sh rows: per label, tok/s (mean over every kept run), ms per
token, the mean of each pass, the sd over passes, loads per decoded token-layer, peak VRAM.

    summary.py speed.jsonl [--md]
"""
import json, re, sys
from collections import defaultdict
from statistics import mean, pstdev, stdev

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(list); order = []
for r in rows:
    if "tok_s" not in r: print("FAILED", r["tag"], r.get("note"), r.get("error"), file=sys.stderr); continue
    if r["tag"] not in by: order.append(r["tag"])
    by[r["tag"]].append(r)
print("| label | runs | tok/s | ms per token | per pass (tok/s) | sd over passes | min to max over runs | loads per decoded token-layer | load histogram 0/1/2/3+ (%) | peak VRAM MiB |")
print("|---|---|---|---|---|---|---|---|---|---|")
out = {}
for lab in order:
    R = by[lab]; ts = [r["tok_s"] for r in R]
    pp = defaultdict(list)
    for r in R: pp[re.search(r"pass (\d+)", r["note"]).group(1)].append(r["tok_s"])
    pm = [mean(v) for _, v in sorted(pp.items())]
    lt = ld = 0; hist = None
    for r in R:
        for l in r["route_lines"]:
            m = re.search(r"temporal-route2: .*decode_layer_tokens=(\d+) loads=(\d+).*hist=\[([0-9,]+)\]", l)
            if m:
                lt += int(m.group(1)); ld += int(m.group(2)); h = [int(x) for x in m.group(3).split(",")]
                hist = h if hist is None else [a + b for a, b in zip(hist, h)]
            m = re.search(r"temporal-route-cuda: .*layer_tokens=(\d+) cold_fills=(\d+) swaps=(\d+)", l)
            if m: lt += int(m.group(1)) - int(m.group(2)); ld += int(m.group(3))
    hs = ""
    if hist: t = sum(hist); hs = " / ".join(f"{100*x/t:.1f}" for x in (hist[0], hist[1], hist[2], sum(hist[3:])))
    out[lab] = {"tok_s": mean(ts), "ms": mean(1000 / t for t in ts), "pass": pm, "n": len(ts), "loads": ld / lt if lt else None}
    print(f"| {lab} | {len(ts)} | {mean(ts):.1f} | {mean(1000/t for t in ts):.4f} | {' / '.join(f'{x:.1f}' for x in pm)} | "
          f"{(stdev(pm) if len(pm) > 1 else 0):.2f} | {min(ts):.1f} to {max(ts):.1f} | {(f'{ld/lt:.4f}' if lt else '')} | {hs} | {max(r['vram_peak_mib'] for r in R)} |")
json.dump(out, open(sys.argv[1].replace(".jsonl", "_summary.json"), "w"), indent=1)
