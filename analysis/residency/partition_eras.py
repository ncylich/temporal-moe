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

NEVER_LIVE_RECORDS = ("smoke_", "lfm25_vllm")
NEVER_LIVE_CELLS = {("lfm25_instruct", "humaneval_instruct"),
                    ("qwen35_instruct", "humaneval_instruct")}


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
            if "PROTOCOL CUTOVER" in line or "Rows BELOW" in line:
                hist.append(line)               # cutover markers belong to history now
            else:
                keep.append(line)               # header + producer comments
            continue
        rec, arm, task, met = key
        invalid = any(rec.startswith(p) for p in NEVER_LIVE_RECORDS) \
            or (rec, task) in NEVER_LIVE_CELLS
        if invalid or last[key] != idx:
            hist.append(line)
        else:
            keep.append(line)

    os.makedirs(HISTDIR, exist_ok=True)
    with open(HIST, "a") as fh:
        if os.path.getsize(HIST) if os.path.exists(HIST) else 0 == 0:
            fh.write('"# Superseded/probe rows partitioned out of the live CSV '
                     '(2026-08-14). Original order preserved. Era ledger: '
                     '../PROTOCOL_ERAS.md"\n')
        fh.writelines(hist)
    banner = ('"# AUTHORITATIVE ROWS ONLY (partitioned 2026-08-14): one row per '
              '(record, arm, task, metric), produced by the single-pass protocol or '
              'listed valid in PROTOCOL_ERAS.md. Full history: '
              'superseded/instruct_genbench_vllm_history.csv"\n')
    open(LIVE, "w").writelines([banner] + keep)
    print(f"live: {len(keep)} lines ({len(last)} cells); history: +{len(hist)} lines")


if __name__ == "__main__":
    main()
