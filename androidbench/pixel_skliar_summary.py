#!/usr/bin/env python3
"""Tables for PIXEL_SKLIAR_LEDGER.md from results/pixel_skliar/rows.jsonl and toklog/*.csv.

    python3 pixel_skliar_summary.py --tag t1 [--arms ...] > results/pixel_skliar/summary_t1.md

Per arm: mean over the texts of each text's tok/s (llama-bench's mean of its repetitions, none discarded), the
range over texts, ms per token, loads per decoded token-layer with the pooled histogram, peak RSS. Void rows
are left out and counted. Per-token tables come from the engine's token log: time per decoded token grouped
by the largest number of experts one layer had to load on that token, and a least-squares line of token
time against the token's total loads.
"""
import argparse, collections, csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "pixel_skliar")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="t1"); ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(OUT, "rows.jsonl"))]
    rows = [r for r in rows if r["tag"] == a.tag]
    arms = a.arms or list(dict.fromkeys(r["arm"] for r in rows))
    print(f"## rows tagged `{a.tag}`\n")
    print("| arm | texts | tok/s, mean over texts [min, max] | sd over reps, mean | ms per token | loads per decoded token-layer | "
          "0 / 1 / 2 / 3 / 4+ loads, % of token-layers | peak RSS MB | void rows |")
    print("|---|---|---|---|---|---|---|---|---|")
    base = None
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        ok = {}
        for r in ar:                                   # the last clean row per text
            if not r["void"]:
                ok[r["text"]] = r
        if not ok:
            print(f"| {arm} | 0 | -- | | | | | | {len(ar)} |"); continue
        tps = [r["decode_tps"] for r in ok.values()]
        m = sum(tps) / len(tps)
        hist = collections.Counter(); loads = lt = 0
        for r in ok.values():
            for i, c in enumerate(r.get("hist", [])):
                hist[min(i, 4)] += c
            loads += r.get("loads", 0); lt += r.get("decode_layer_tokens", 0)
        n = sum(hist.values())
        hs = " / ".join(f"{100 * hist[i] / n:.1f}" for i in range(5)) if n else "--"
        rss = [r["peak_rss_mb"] for r in ok.values()]
        print(f"| {arm} | {len(ok)} | {m:.2f} [{min(tps):.2f}, {max(tps):.2f}] | {sum(r['stddev_tps'] for r in ok.values()) / len(ok):.2f} | "
              f"{1000 / m:.2f} | {loads / lt:.4f} | {hs} | {sum(rss) / len(rss):.0f} | {sum(r['void'] for r in ar)} |" if lt else
              f"| {arm} | {len(ok)} | {m:.2f} [{min(tps):.2f}, {max(tps):.2f}] | {sum(r['stddev_tps'] for r in ok.values()) / len(ok):.2f} | "
              f"{1000 / m:.2f} | -- | -- | {sum(rss) / len(rss):.0f} | {sum(r['void'] for r in ar)} |")
        if base is None:
            base = (arm, m)
    print("\nPer text (tok/s, the three repetitions):\n")
    print("| arm | " + " | ".join(f"text {v}" for v in sorted({r['text'] for r in rows})) + " |")
    print("|---|" + "---|" * len({r['text'] for r in rows}))
    for arm in arms:
        cells = []
        for v in sorted({r["text"] for r in rows}):
            rr = [r for r in rows if r["arm"] == arm and r["text"] == v and not r["void"]]
            cells.append(f"{rr[-1]['decode_tps']:.2f} ({', '.join('%.1f' % x for x in rr[-1]['samples_ts'])})" if rr else "--")
        print(f"| {arm} | " + " | ".join(cells) + " |")

    print("\nPer decoded token (engine token log; ms):\n")
    for arm in arms:
        toks = []
        for r in rows:
            if r["arm"] != arm or r["void"]:
                continue
            p = os.path.join(OUT, "toklog", f"{arm}_v{r['text']}_{r['tag']}.csv")
            if os.path.exists(p):
                toks += [x for x in csv.DictReader(open(p)) if int(x["rep"]) >= 0]
        if not toks:
            continue
        ms = lambda xs: sum(int(x["ns"]) for x in xs) / 1e6 / len(xs)
        first = [x for x in toks if int(x["i"]) == 0]; rest = [x for x in toks if int(x["i"]) > 0]
        print(f"**{arm}**: {len(toks)} tokens, mean {ms(toks):.2f}; first token after the prompt {ms(first):.1f} (n {len(first)}); "
              f"later tokens {ms(rest):.2f}, median {sorted(int(x['ns']) for x in rest)[len(rest) // 2] / 1e6:.2f}, "
              f"p95 {sorted(int(x['ns']) for x in rest)[int(len(rest) * 0.95)] / 1e6:.2f}, max {max(int(x['ns']) for x in rest) / 1e6:.1f}")
        if any(int(x["loads"]) for x in rest):
            b = collections.defaultdict(list)
            for x in rest:
                b[min(int(x["max_layer_loads"]), 5)].append(x)
            print("\n| largest per-layer load count on the token | tokens | share | ms per token | loads on the token (13 layers) | ms blocked on fetches |")
            print("|---|---|---|---|---|---|")
            for k in sorted(b):
                v = b[k]
                print(f"| {k}{'+' if k == 5 else ''} | {len(v)} | {100 * len(v) / len(rest):.1f}% | {ms(v):.2f} | "
                      f"{sum(int(x['loads']) for x in v) / len(v):.1f} | {sum(int(x['wait_ns']) for x in v) / 1e6 / len(v):.2f} |")
            xs = [float(x["loads"]) for x in rest]; ys = [int(x["ns"]) / 1e6 for x in rest]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                r2 = (sl * sxx) ** 2 / (sxx * sum((y - my) ** 2 for y in ys)) if sum((y - my) ** 2 for y in ys) else 0
                print(f"\nLeast squares, tokens after the first of each repetition: ms = {my - sl * mx:.2f} + {sl:.3f} x loads on the token (r^2 {r2:.2f}).")
        print()


if __name__ == "__main__":
    main()
