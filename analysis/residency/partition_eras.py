#!/usr/bin/env python3
"""Partition instruct_genbench_vllm.csv: live file = authoritative rows only.

Every superseded, invalid, or probe row moves to
results/ablations/superseded/instruct_genbench_vllm_history.csv (full original line
order preserved there). The live file keeps, per (record, arm, task, metric), ONLY the
last row -- minus records/tasks that are invalid in every era (smoke_*, lfm25_vllm,
LFM/qwen humaneval_instruct, lfm25_fullset_audit stays live as an explicitly-named
audit record). After this, cross-era mispairing is structurally impossible: one row
per cell, no history in the analysis path.

Run ONLY after a rerun wave completes (partition keeps the newest row per cell).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

LIVE = os.path.join(ABLATIONS, "instruct_genbench_vllm.csv")
HISTDIR = os.path.join(ABLATIONS, "superseded")
HIST = os.path.join(HISTDIR, "instruct_genbench_vllm_history.csv")

NEVER_LIVE_RECORDS = ("smoke_", "lfm25_vllm", "gemma4_adapted",
                      "gemma4_ctrl_sft", "lfm25_fullset_audit")
# the greedy-era adaptation trio and the ladder-era audit probe live ONLY in
# history: the trio is valid solely as its internally-paired comparison there
NEVER_LIVE_METRICS = ("exact_match,strict-match", )   # inert under chat protocol
NEVER_LIVE_METRIC_SUFFIX = ",answer-only"  # rescores of since-overwritten generations
NEVER_LIVE_CELLS = {("lfm25_instruct", "humaneval_instruct"),
                    ("qwen35_instruct", "humaneval_instruct"),
                    ("gemma4_instruct", "humaneval_instruct"),
                    ("gptoss_20b", "mmlu_flan_cot_fewshot"),
                    ("gptoss_120b", "mmlu_flan_cot_fewshot"),
                    ("gptoss_120b", "humaneval_instruct"),
                    ("gptoss_20b", "humaneval_instruct")}


def main():
    lines = open(LIVE).readlines()
    parsed = []
    for idx, line in enumerate(lines):
        row = next(csv.reader([line]), None)
        key = None
        if row and len(row) > 7 and not row[0].startswith("#") and row[0] != "model":
            key = (row[0], row[3], row[5], row[6])
        parsed.append((idx, line, row, key))

    last = {}
    for idx, _, row, key in parsed:
        if key:
            last[key] = idx

    keep, hist = [], []
    for idx, line, row, key in parsed:
        if key is None:
            if "AUTHORITATIVE ROWS ONLY" in line:
                continue                    # fresh banner is prepended each run
            if "PROTOCOL CUTOVER" in line or "Rows BELOW" in line:
                hist.append(line)               # cutover markers belong to history now
            else:
                keep.append(line)               # header + producer comments
            continue
        rec, arm, task, met = key
        invalid = any(rec.startswith(p) for p in NEVER_LIVE_RECORDS) \
            or (rec, task) in NEVER_LIVE_CELLS \
            or met in NEVER_LIVE_METRICS \
            or met.endswith(NEVER_LIVE_METRIC_SUFFIX) \
            or (rec, task) == ("lfm25_instruct", "mmlu_flan_cot_fewshot")
        if invalid or last[key] != idx:
            hist.append(line)
        else:
            keep.append(line)

    os.makedirs(HISTDIR, exist_ok=True)
    with open(HIST, "a") as fh:
        if (os.path.getsize(HIST) if os.path.exists(HIST) else 0) == 0:
            fh.write('"# Superseded/probe rows partitioned out of the live CSV '
                     '(2026-08-14). Original order preserved. Validity rules: '
                     'analysis/residency/partition_eras.py; protocol: '
                     '../DATA_CONTRACT.md"\n')
        fh.writelines(hist)
    banner = ('"# AUTHORITATIVE ROWS ONLY: one row per (record, arm, task, metric), all '
              'from the single-pass sampled protocol (never greedy; eos-only stops; '
              'thinking stripped before scoring; budgets in max_gen_toks) or the bespoke '
              'producers listed in DATA_CONTRACT.md. Full history: '
              'superseded/instruct_genbench_vllm_history.csv"\n')
    open(LIVE, "w").writelines([banner] + keep)
    print(f"live: {len(keep)} lines ({len(last)} cells); history: +{len(hist)} lines")


if __name__ == "__main__":
    main()
