#!/usr/bin/env python3
"""Table of multictx.sh rows: per conversation and configuration, tok/s of base / ours_stream / skliar_stream
(mean of kept trials) and loads per decode layer-token, then the mean over conversations.

    python3 multictx_summary.py gemma_multictx.jsonl [--md out.md]
"""
import argparse, collections, json, re


def loads(r):
    for l in r.get("route_lines", []):
        m = re.search(r"temporal-route2:.*decode_layer_tokens=(\d+) loads=(\d+)", l)
        if m:
            return int(m.group(2)) / int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rows"); ap.add_argument("--md")
    a = ap.parse_args()
    T = collections.defaultdict(dict)
    for line in open(a.rows):
        r = json.loads(line)
        if "samples_ts" not in r:
            continue
        cfg, var = r["tag"].rsplit("-", 2)[-2:]
        k = r["samples_ts"][1:]
        T[(cfg, var)][r["arm"]] = (sum(k) / len(k), loads(r))
    out = []
    for cfg in ("shexp", "default"):
        out += [f"\n**{cfg}**\n", "| conversation | stock tok/s | ours streamed | Skliar streamed | Skliar / ours | loads ours | loads Skliar |",
                "|---|---|---|---|---|---|---|"]
        acc = collections.defaultdict(list)
        for (c, v), d in sorted(T.items()):
            if c != cfg:
                continue
            o, s = d.get("ours_stream"), d.get("skliar_stream")
            b = d.get("base", (None, None))[0]
            out.append(f"| {v} | {'' if b is None else '%.2f' % b} | {o[0]:.2f} | {s[0]:.2f} | {s[0] / o[0]:.3f} | {o[1]:.4f} | {s[1]:.4f} |")
            for key, val in (("b", b), ("o", o[0]), ("s", s[0]), ("r", s[0] / o[0]), ("lo", o[1]), ("ls", s[1])):
                if val is not None:
                    acc[key].append(val)
        m = {k: sum(v) / len(v) for k, v in acc.items()}
        out.append(f"| **mean of {len(acc['o'])}** | {m.get('b', float('nan')):.2f} | {m['o']:.2f} | {m['s']:.2f} | {m['r']:.3f} | {m['lo']:.4f} | {m['ls']:.4f} |")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
