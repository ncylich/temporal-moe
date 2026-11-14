#!/usr/bin/env python3
"""Four-column table from an evals.sh output directory: Skliar | temporal base (our rule, no training) |
temporal trained | normal MoE. Rows: test CE (tokens 1024..2047 of 512 test rows), loads per decoded
token-layer, and every lm-eval task metric with its standard error.

    python3 gpubench/pretrained/table.py gpubench/results/pretrained/evals [--md out.md]
"""
import argparse, glob, json, os, re

COLS = [("moe_skliar", "Skliar (plain MoE)"), ("moe_ours", "temporal base (plain MoE, our rule)"),
        ("tmoe_trained", "temporal trained"), ("moe_topk", "normal MoE (stock top-k)")]
TASKS = ["arc_challenge", "arc_easy", "boolq", "copa", "hellaswag", "lambada_openai", "openbookqa", "piqa", "sciq", "winogrande"]


def ce(d, name):
    t = open(os.path.join(d, f"ce_{name}.log")).read()
    m = re.search(r"^loss ([0-9.]+) nats/token \(SE over samples ([0-9.]+)\)", t, re.M)
    ld = re.search(r"loads per token-layer ([0-9.]+)", t)
    return float(m.group(1)), float(m.group(2)), (float(ld.group(1)) if ld else None)


def lme(d, name):
    r = json.load(open(sorted(glob.glob(os.path.join(d, f"lme_{name}", "results_*.json")))[-1]))["results"]
    ld = [json.loads(l) for l in open(os.path.join(d, f"lme_{name}", "loads.jsonl"))][-1] if os.path.exists(os.path.join(d, f"lme_{name}", "loads.jsonl")) else None
    return r, (ld["loads"] / ld["token_layers"] if ld and ld["token_layers"] else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--md")
    a = ap.parse_args()
    C = {n: ce(a.dir, n) for n, _ in COLS}
    L = {n: lme(a.dir, n) for n, _ in COLS}
    out = ["| | " + " | ".join(t for _, t in COLS) + " |", "|---|" + "---|" * len(COLS)]
    out.append("| test CE, nats per token (lower is better) | " + " | ".join(f"{C[n][0]:.4f} ± {C[n][1]:.4f}" for n, _ in COLS) + " |")
    out.append("| loads per decoded token-layer, test CE rows | " + " | ".join("--" if C[n][2] is None else f"{C[n][2]:.3f}" for n, _ in COLS) + " |")
    out.append("| loads per decoded token-layer, lm-eval | " + " | ".join("--" if L[n][1] is None else f"{L[n][1]:.3f}" for n, _ in COLS) + " |")
    for t in TASKS:
        for metric in ("acc", "acc_norm", "perplexity"):
            if f"{metric},none" not in L[COLS[0][0]][0][t]:
                continue
            cells = []
            for n, _ in COLS:
                v = L[n][0][t]; se = v.get(f"{metric}_stderr,none")
                cells.append(f"{v[f'{metric},none']:.4f}" + (f" ± {se:.4f}" if isinstance(se, float) else "") if metric != "perplexity" else f"{v[f'{metric},none']:.1f}")
            out.append(f"| {t} {metric} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
