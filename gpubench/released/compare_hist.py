#!/usr/bin/env python3
"""Side-by-side loads-per-layer-token distribution: engine (TEMPORAL_ROUTE_HIST json) against a saved Python
histogram (results/ablations/skliar_c8_*_hist.json), plus a GSM8K score of the engine's generations
(flexible extract: the last number in the response, as lm-eval's flexible-extract filter).

    python3 compare_hist.py --engine hist_skliar.json --ref results/ablations/skliar_c8_qwen_lam04_hist.json \
        [--gen gen_gsm_skliar.jsonl] [--out row.json]
"""
import argparse, json, re


def dist(h):
    n = sum(h)
    return [x / n for x in h], n, sum(i * x for i, x in enumerate(h)) / n


def score(path):
    ok = tot = 0
    for line in open(path):
        r = json.loads(line)
        nums = re.findall(r"-?[$0-9.,]{2,}|-?[0-9]+", r["content"])
        pred = nums[-1].replace("$", "").replace(",", "").rstrip(".") if nums else ""
        gold = r["answer"].replace(",", "")
        try:
            ok += float(pred) == float(gold)
        except ValueError:
            pass
        tot += 1
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True); ap.add_argument("--ref"); ap.add_argument("--gen"); ap.add_argument("--out")
    a = ap.parse_args()
    e = json.load(open(a.engine))
    fe, ne, me = dist(e["hist"])
    row = {"engine": a.engine, "engine_rows": ne, "engine_mean": me, "engine_frac": fe[:9]}
    print(f"engine: {ne} decode layer-tokens, mean loads {me:.4f}")
    if a.ref:
        r = json.load(open(a.ref))
        fr, nr, mr = dist(r["hist"])
        row.update(ref=a.ref, ref_rows=nr, ref_mean=mr, ref_frac=fr[:9])
        print(f"saved:  {nr} decode layer-tokens, mean loads {mr:.4f}")
        print("loads  engine   saved")
        for i in range(9):
            print(f"{i if i < 8 else '8+':>5}  {fe[i] if i < 8 else sum(fe[8:]):.4f}   {fr[i] if i < 8 else sum(fr[8:]):.4f}")
        row["total_variation"] = 0.5 * sum(abs(x - y) for x, y in zip(fe, fr))
        print(f"total variation distance {row['total_variation']:.4f}")
    if a.gen:
        ok, tot = score(a.gen)
        row.update(gsm8k_correct=ok, gsm8k_n=tot)
        print(f"GSM8K flexible-extract on these generations: {ok}/{tot} = {ok / tot:.3f}")
    if a.out:
        open(a.out, "w").write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
