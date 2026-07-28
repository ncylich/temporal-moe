#!/usr/bin/env python3
"""lmeval-widen: parse the 4 extra zero-shot tasks (sciq, boolq, lambada_openai, copa) for the four
1e19 cells into results/ablations/t19_lmeval_extra.csv (schema: model,task,metric,value,stderr).
acc and acc_norm where defined. Prints the headline: dense<<moe~=temporal ordering + which cells
sit >2 stderr above chance (chance = per-task random baseline).
"""
import os, sys, csv, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT   # canonical resolver: $TMOE_ROOT, then git, then file location
RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/t19_lmeval_extra.csv")
MODELS = ["dense_1e19", "moe_coarse_1e19", "g1_tmoe_coarse_1e19", "temporal_fine_g3_1e19"]
LABEL = {"dense_1e19": "dense_1e19", "moe_coarse_1e19": "moe_coarse_1e19",
         "g1_tmoe_coarse_1e19": "temporal_coarse_1e19", "temporal_fine_g3_1e19": "temporal_fine_1e19"}
CHANCE = {"sciq": 0.25, "boolq": 0.5, "copa": 0.5, "lambada_openai": 0.0}   # random baselines


def latest_json(d):
    js = glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
    return max(js, key=os.path.getmtime) if js else None


def main():
    rows = []
    for run in MODELS:
        d = os.path.join(RUNS, run, "lmeval_extra_0shot")
        j = latest_json(d) if os.path.isdir(d) else None
        if not j:
            print(f"[skip] {run}: no json", file=sys.stderr); continue
        res = json.load(open(j)).get("results", {})
        for task, m in sorted(res.items()):
            for base in ("acc", "acc_norm"):
                vk = next((k for k in m if k == base or k.startswith(base + ",")), None)
                sk = next((k for k in m if k == base + "_stderr" or k.startswith(base + "_stderr,")), None)
                if vk is None or m.get(vk) is None:
                    continue
                sv = m.get(sk)
                rows.append([LABEL[run], task, base, round(float(m[vk]), 6),
                             (round(float(sv), 6) if isinstance(sv, (int, float)) else "")])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "task", "metric", "value", "stderr"]); w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")
    # headline
    print("\nHEADLINE  (acc; task | dense | moe_coarse | temporal_coarse | temporal_fine)")
    by = {}
    for m, t, me, v, s in rows:
        if me == "acc":
            by.setdefault(t, {})[m] = (v, s)
    order = ["dense_1e19", "moe_coarse_1e19", "temporal_coarse_1e19", "temporal_fine_1e19"]
    for t in sorted(by):
        cells = by[t]
        line = f"  {t:16}"
        for m in order:
            v, s = cells.get(m, (float('nan'), 0))
            above = ""
            if CHANCE.get(t) is not None and s and v - CHANCE[t] > 2 * s:
                above = "*"   # >2 stderr above chance
            line += f" {v:.3f}{above:1}"
        print(line + "   (* = >2se above chance)")


if __name__ == "__main__":
    main()
