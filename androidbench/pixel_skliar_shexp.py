#!/usr/bin/env python3
"""Tables for ledger S1-7: LLAMA_TEMPORAL_SHEXP_FIRST off (tag sx0) against on (tag sx1).

    python3 pixel_skliar_shexp.py > results/pixel_skliar/summary_shexp.md

Per arm and switch: tok/s as the mean over the five texts of llama-bench's mean of three repetitions, ms per
token, and from the engine's token log (tokens after the first of each repetition) the mean, 95th percentile,
slowest token, time blocked on fetches, and for Skliar the least-squares line of token time against the
token's loads. Off against on is paired by text. Void rows are left out and counted.
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "pixel_skliar")
ARMS = ["trainedtext_stream", "skliar_stream", "stock_resident"]
TAGS = {"sx0": "off", "sx1": "on"}

rows = [json.loads(l) for l in open(os.path.join(OUT, "rows.jsonl"))]
ok, void = {}, {}
for r in rows:
    if r["tag"] in TAGS and r["arm"] in ARMS:
        k = (r["arm"], r["tag"])
        if r["void"]:
            void[k] = void.get(k, 0) + 1
        else:
            ok.setdefault(k, {})[r["text"]] = r           # the last clean row per text


def toks(arm, tag):
    out = []
    for v in sorted(ok.get((arm, tag), {})):
        p = os.path.join(OUT, "toklog", f"{arm}_v{v}_{tag}.csv")
        out += [x for x in csv.DictReader(open(p)) if int(x["rep"]) >= 0 and int(x["i"]) > 0]
    return out


def fit(ts):
    xs = [float(x["loads"]) for x in ts]; ys = [int(x["ns"]) / 1e6 for x in ts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - sl * mx, sl


print("| arm | switch | texts | tok/s, mean over texts [min, max] | ms per token | loads per decoded token-layer | "
      "later tokens: mean ms | p95 ms | slowest ms | ms blocked on fetches per token | fit: ms = a + b x loads | peak RSS MB | void rows |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
mean = {}
for arm in ARMS:
    for tag, name in TAGS.items():
        d = ok.get((arm, tag), {})
        if not d:
            print(f"| {arm} | {name} | 0 | | | | | | | | | | {void.get((arm, tag), 0)} |"); continue
        tps = [r["decode_tps"] for r in d.values()]; m = sum(tps) / len(tps); mean[(arm, tag)] = m
        lt = sum(r.get("decode_layer_tokens", 0) for r in d.values())
        loads = f"{sum(r.get('loads', 0) for r in d.values()) / lt:.4f}" if lt else ("1.0000" if arm.startswith("trained") else "--")
        ts = toks(arm, tag); ns = sorted(int(x["ns"]) for x in ts)
        f = fit(ts)
        print(f"| {arm} | {name} | {len(d)} | {m:.2f} [{min(tps):.2f}, {max(tps):.2f}] | {1000 / m:.2f} | {loads} | "
              f"{sum(ns) / len(ns) / 1e6:.2f} | {ns[int(len(ns) * 0.95)] / 1e6:.2f} | {ns[-1] / 1e6:.1f} | "
              f"{sum(int(x['wait_ns']) for x in ts) / 1e6 / len(ts):.2f} | "
              + (f"{f[0]:.2f} + {f[1]:.3f}" if f else "--") +
              f" | {sum(r['peak_rss_mb'] for r in d.values()) / len(d):.0f} | {void.get((arm, tag), 0)} |")

print("\nPer text, tok/s (three repetitions), off then on, and on against off in percent:\n")
print("| arm | " + " | ".join(f"text {v}" for v in range(1, 6)) + " | mean of the paired differences [min, max] |")
print("|---|" + "---|" * 6)
for arm in ARMS:
    cells, ds = [], []
    for v in range(1, 6):
        a, b = ok.get((arm, "sx0"), {}).get(v), ok.get((arm, "sx1"), {}).get(v)
        if a and b:
            d = 100 * (b["decode_tps"] / a["decode_tps"] - 1); ds.append(d)
            cells.append(f"{a['decode_tps']:.2f} / {b['decode_tps']:.2f} ({d:+.1f}%)")
        else:
            cells.append("--")
    print(f"| {arm} | " + " | ".join(cells) + " | " + (f"{sum(ds) / len(ds):+.2f}% [{min(ds):+.1f}, {max(ds):+.1f}]" if ds else "--") + " |")

print("\nTrained model against Skliar on the same text, tok/s, positive = trained faster:\n")
print("| switch | " + " | ".join(f"text {v}" for v in range(1, 6)) + " | mean of the paired differences [min, max] | ratio of the means |")
print("|---|" + "---|" * 7)
for tag, name in TAGS.items():
    cells, ds = [], []
    for v in range(1, 6):
        a, b = ok.get(("skliar_stream", tag), {}).get(v), ok.get(("trainedtext_stream", tag), {}).get(v)
        if a and b:
            d = 100 * (b["decode_tps"] / a["decode_tps"] - 1); ds.append(d); cells.append(f"{d:+.1f}%")
        else:
            cells.append("--")
    rm = (f"{100 * (mean[('trainedtext_stream', tag)] / mean[('skliar_stream', tag)] - 1):+.2f}%"
          if ("trainedtext_stream", tag) in mean and ("skliar_stream", tag) in mean else "--")
    print(f"| {name} | " + " | ".join(cells) + " | " + (f"{sum(ds) / len(ds):+.2f}% [{min(ds):+.1f}, {max(ds):+.1f}]" if ds else "--") + f" | {rm} |")

print("\nSkliar, later tokens grouped by the largest number of experts one layer loaded on the token (ms per token, off / on):\n")
print("| largest per-layer load count | tokens off / on | ms per token off / on | ms blocked on fetches off / on |")
print("|---|---|---|---|")
g = {}
for tag in TAGS:
    for x in toks("skliar_stream", tag):
        g.setdefault(min(int(x["max_layer_loads"]), 5), {}).setdefault(tag, []).append(x)
for k in sorted(g):
    c = []
    for f in (lambda v: str(len(v)), lambda v: f"{sum(int(x['ns']) for x in v) / 1e6 / len(v):.2f}",
              lambda v: f"{sum(int(x['wait_ns']) for x in v) / 1e6 / len(v):.2f}"):
        c.append(" / ".join(f(g[k][t]) if g[k].get(t) else "--" for t in TAGS))
    print(f"| {k}{'+' if k == 5 else ''} | " + " | ".join(c) + " |")
