#!/usr/bin/env python3
"""Append one FLAME-1e18 run's lm-eval results into flame1e18_downstream.csv.
Wide schema matching results/ablations/t19_lmeval.csv + a stderr column:
    model,task,acc,acc_norm,stderr   (acc_norm blank for tasks with no acc_norm; stderr is acc's stderr)
Usage: flame1e18_downstream_csv.py <model> <lmeval_output_dir> <csv_path>
"""
import os, sys, csv, json, glob

def latest_json(d):
    js = glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
    js = [j for j in js if os.path.basename(j).startswith("results_")]
    return max(js, key=os.path.getmtime) if js else None

def pick(m, base):
    vk = next((k for k in m if k == base or k.startswith(base + ",")), None)
    return m.get(vk) if vk is not None else None

def main():
    model, out_dir, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
    j = latest_json(out_dir)
    if not j:
        print(f"[csv] {model}: NO json in {out_dir} -- nothing written", file=sys.stderr)
        sys.exit(1)
    res = json.load(open(j)).get("results", {})
    new_rows = []
    for task, m in sorted(res.items()):
        acc = pick(m, "acc")
        acc_norm = pick(m, "acc_norm")
        stderr = pick(m, "acc_stderr")
        if acc is None:
            continue
        new_rows.append([
            model, task,
            round(float(acc), 6),
            (round(float(acc_norm), 6) if isinstance(acc_norm, (int, float)) else ""),
            (round(float(stderr), 6) if isinstance(stderr, (int, float)) else ""),
        ])
    header = ["model", "task", "acc", "acc_norm", "stderr"]
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            r = list(csv.reader(f))
        if r and r[0] == header:
            rows = [x for x in r[1:] if x and x[0] != model]  # drop any stale rows for this model
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows + new_rows)
    print(f"[csv] {model}: wrote {len(new_rows)} rows from {os.path.basename(j)} -> {csv_path}")

if __name__ == "__main__":
    main()
