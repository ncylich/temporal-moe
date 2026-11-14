#!/usr/bin/env python3
"""Tables from speed.sh rows (speed.jsonl): for every arm and kernel setting, the mean over the text rows of
the kept-trial mean (trials 2 to 4), its range over rows, loads per decoded token-layer and peak VRAM; then
each method at its fastest setting next to the all-in-VRAM rows.

    python3 gpubench/pretrained/speed_summary.py gpubench/results/pretrained/speed2/speed.jsonl [--md out.md]
"""
import argparse, collections, json, re


def loads(r):
    for l in r.get("route_lines", []):
        m = re.search(r"decode_layer_tokens=(\d+) loads=(\d+)", l)
        if m:
            return int(m.group(2)) / int(m.group(1))
        m = re.search(r"temporal-route-cuda:.*layer_tokens=(\d+).*?swaps=(\d+)", l)   # trained rule's kernel: all tokens
        if m and int(m.group(1)):
            return int(m.group(2)) / int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rows"); ap.add_argument("--md")
    a = ap.parse_args()
    T = collections.defaultdict(list)
    for line in open(a.rows):
        r = json.loads(line)
        if "samples_ts" not in r:
            print("FAILED row:", r.get("tag"), r.get("arm")); continue
        cfg = r["tag"].rsplit("-v", 1)[0]
        T[(r["arm"], cfg)].append((sum(r["samples_ts"][1:]) / len(r["samples_ts"][1:]), loads(r), r["vram_peak_mib"]))
    S = {}
    for k, v in T.items():
        ts = [x[0] for x in v]; ld = [x[1] for x in v if x[1] is not None]
        S[k] = dict(n=len(v), ts=sum(ts) / len(ts), lo=min(ts), hi=max(ts), loads=(sum(ld) / len(ld) if ld else None), peak=max(x[2] for x in v))
    out = ["| arm | setting | texts | tok/s | range over texts | ms per token | loads per decoded token-layer | peak VRAM MiB |", "|---|---|---|---|---|---|---|---|"]
    for (arm, cfg), s in sorted(S.items()):
        out.append(f"| {arm} | {cfg} | {s['n']} | {s['ts']:.1f} | {s['lo']:.1f} to {s['hi']:.1f} | {1000 / s['ts']:.3f} | "
                   f"{'--' if s['loads'] is None else '%.3f' % s['loads']} | {s['peak']} |")
    out += ["", "Each streamed method at its fastest setting:", "", "| arm | fastest setting | tok/s | ms per token |", "|---|---|---|---|"]
    for arm in ("ours_stream", "skliar_stream", "trained_stream"):
        c = {cfg: s for (a_, cfg), s in S.items() if a_ == arm}
        if c:
            cfg = max(c, key=lambda k: c[k]["ts"])
            out.append(f"| {arm} | {cfg} | {c[cfg]['ts']:.1f} | {1000 / c[cfg]['ts']:.3f} |")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
