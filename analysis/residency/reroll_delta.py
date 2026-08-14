#!/usr/bin/env python3
"""Ladder-era vs single-pass deltas -- producer of reroll_delta_record.md numbers.

Pairing rule: for every (record, arm, task) cell, the LAST post-cutover row in the
history file (the measurement retired when the truncate-and-retry ladder was
abandoned) against the authoritative row in the live CSV, primary accuracy metric
per task only. Both files are read AT A PINNED COMMIT (--ref), because the live CSV
keeps moving (e.g. the 2026-08-14 gpt-oss budget wave) while this comparison is a
snapshot of what the ladder retirement changed. Default ref = dba0c2e, the state
right after the full single-pass re-measurement and before the 8192-budget wave.

Era caveat (stated in the record too): post-cutover history rows cannot be
attributed to ladder-vs-other causes from the CSV alone; rows superseded for other
reasons (bad runs, budget probes) are included. Pairs are stratified by whether
max_gen_toks matches to make the budget-change subset visible.
"""
import argparse
import csv
import subprocess

CUTOVER = "Rows BELOW use the FINAL corrected protocol"
PRIMARY = {"gsm8k_cot_zeroshot": "exact_match,flexible-extract",
           "ifeval": "prompt_level_strict_acc,none",
           "humaneval_instruct": "pass@1,create_test",
           "humaneval_think": "pass@1,create_test",
           "humaneval_gemma_fixed": "pass@1,create_test",
           "humaneval_gptoss": "pass@1,create_test",
           "mmlu_flan_cot_fewshot": "exact_match,get-answer",
           "mmlu_gptoss_relaxed": "exact_match,get-answer"}


def show(ref, path):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True,
                         text=True, check=True).stdout
    return out.splitlines()


def rows(lines):
    for line in lines:
        r = next(csv.reader([line]), None)
        if r and len(r) > 9 and not r[0].startswith("#") and r[0] != "model":
            yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="dba0c2e")
    a = ap.parse_args()
    hist = show(a.ref, "results/ablations/superseded/instruct_genbench_vllm_history.csv")
    live = show(a.ref, "results/ablations/instruct_genbench_vllm.csv")
    cut = next(i for i, ln in enumerate(hist) if CUTOVER in ln)

    old = {}
    for r in rows(hist[cut + 1:]):
        if PRIMARY.get(r[5]) == r[6]:
            old[(r[0], r[3], r[5])] = (float(r[7]), r[9])
    pairs = []
    for r in rows(live):
        key = (r[0], r[3], r[5])
        if PRIMARY.get(r[5]) == r[6] and key in old:
            ov, ob = old[key]
            pairs.append((key, ov, float(r[7]), ob == r[9]))

    for label, sel in [("ALL", pairs),
                       ("same-budget", [p for p in pairs if p[3]]),
                       ("budget-changed", [p for p in pairs if not p[3]])]:
        if not sel:
            print(f"{label}: n=0")
            continue
        d = [100 * (new - ov) for _, ov, new, _ in sel]
        mx = max(sel, key=lambda p: p[2] - p[1])
        print(f"{label}: n={len(d)}  mean {sum(d)/len(d):+.2f}  "
              f"mean|d| {sum(abs(x) for x in d)/len(d):.2f}  "
              f"max {100*(mx[2]-mx[1]):+.1f} ({' '.join(mx[0])})  "
              f"positive {sum(x > 0 for x in d)}/{len(d)}")


if __name__ == "__main__":
    main()
