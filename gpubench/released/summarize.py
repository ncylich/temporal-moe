#!/usr/bin/env python3
"""Tables from run_released.py rows: per (ctx, arm) the pooled decode rate (mean of every kept trial, i.e.
trials 2 to 4 of every pass), its per-pass values, ms per token, peak VRAM, ratio to the stock baseline,
and the loads per decode layer-token with their distribution (from the engine's exit line, which counts the
decode steps of all four trials and the warm-up token).

    python3 summarize.py gpubench/results/released/qwen35/qwen_decode_final.jsonl [--md out.md]
"""
import argparse, collections, json, re

ORDER = ["base", "ours_vram", "skliar_vram", "ours_stream", "skliar_stream",
         "ours_nobytes", "skliar_nobytes", "ours_nocopy", "skliar_nocopy"]


def loads(row):
    for l in row.get("route_lines", []):
        if l.startswith("temporal-route2:"):
            m = re.search(r"decode_layer_tokens=(\d+) loads=(\d+).*hist=\[([0-9,]+)\]", l)
            rows, ld, h = int(m.group(1)), int(m.group(2)), [int(x) for x in m.group(3).split(",")]
            return ld / rows, [x / rows for x in h]
    return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rows"); ap.add_argument("--md")
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.rows)]
    by = collections.defaultdict(list)
    for r in R:
        if "samples_ts" in r:
            by[(r["ctx"], r["arm"])].append(r)
    out = []
    for ctx in sorted({k[0] for k in by}):
        base = None
        lines = [f"\n### {ctx} context\n",
                 "| arm | tok/s pooled | per pass | ms per token | peak VRAM MiB | vs stock | loads per layer-token | P(0) / P(1) / P(2) / P(3+) |",
                 "|---|---|---|---|---|---|---|---|"]
        for arm in ORDER:
            rs = by.get((ctx, arm))
            if not rs:
                continue
            kept = [t for r in rs for t in r["samples_ts"][1:]]
            ts = sum(kept) / len(kept)
            if arm == "base":
                base = ts
            mean_l, h = loads(rs[-1])
            dist = "" if h is None else f"{h[0]:.3f} / {h[1]:.3f} / {h[2]:.3f} / {sum(h[3:]):.3f}"
            lines.append(f"| {arm} | {ts:.2f} | {' / '.join('%.2f' % r['tok_s'] for r in rs)} | {1000 / ts:.3f} | "
                         f"{max(r['vram_peak_mib'] for r in rs)} | {ts / base:.3f} | "
                         f"{'' if mean_l is None else '%.4f' % mean_l} | {dist} |" if base else "")
        out += lines
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
