#!/usr/bin/env python3
"""Truncation decomposition and three-way rescore, both surfaces (TRUNCATION_RERUN_PLAN §1, §6.4, §6.5).

Splits every blown-up constrained generation by HOW IT ENDED, because the pooled
"blown-up generations are wrong" effect mixes two unrelated things:

  A. hit the cap, thinking never closed      - emitted no answer at all
  B. hit the cap, thinking closed, cut off   - answer truncated mid-emission
  C. over 2x the free counterpart, finished  - the actual length-quality signal
  D. blown only because the FREE arm capped  - says nothing about the constrained run

A and B are mechanical: a generation that never emitted a program or a letter is
wrong because it emitted nothing. Only C speaks to derailment.

Also emits the three-way rescore per cell:
  reported      - scored as-is
  honest        - unfinished (A) counts as unanswered
  finished-only - over the subset that actually finished, n stated

Writes results/ablations/truncation_decomp.csv (per cell) and prints the pooled
table. Re-analysis of committed dumps only; no GPU, no regeneration.
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
MARK = {"qwen35_instruct": "</think>", "lfm25_instruct": "</think>",
        "gemma4_think_on": "<channel|>", "gemma4_instruct": "<channel|>",
        "gptoss": "<|channel|>final<|message|>"}


def marker(rec):
    for pre, mk in MARK.items():
        if rec.startswith(pre):
            return mk
    return MARK["gptoss"] if rec.startswith("gptoss") else None


def correct(item, surface):
    if surface == "HumanEval":
        return item.get("pass")
    g = item.get("gold")
    return None if g is None else item.get("pred_relaxed") == g


def load(rec, arm, task):
    p = f"{SAMP}/{rec}_{arm}_{task}.json"
    if not os.path.exists(p):
        return None
    items = json.load(open(p))["items"]
    if len(items) < 100 or not all("doc" in i and "raw" in i for i in items):
        return None                     # pre-repair dumps lack doc keys / raw text
    return {i["doc"]: i for i in items}


def cells():
    """(surface, record, arm, task) for every constrained cell with a free partner."""
    out = []
    for p in sorted(glob.glob(f"{SAMP}/*.json")):
        b = os.path.basename(p)[:-5]
        for task, surf in (("humaneval_gemma_fixed", "HumanEval"),
                           ("humaneval_gptoss", "HumanEval"),
                           ("humaneval_think", "HumanEval"),
                           ("mmlu_dual", "MMLU")):
            if not b.endswith("_" + task):
                continue
            stem = b[: -len("_" + task)]
            m = re.match(r"(.+)_(R\d+)$", stem)
            if m:
                out.append((surf, m.group(1), m.group(2), task))
    return out


def decompose():
    """(pooled groups, per-cell rows). Groups are {surface: {group: [n, n_wrong]}}
    pooled over every constrained cell. Split out of main so a figure producer can
    read exactly the numbers this file prints, with no chance of the two drifting."""
    groups = {s: {g: [0, 0] for g in "ABCD"} | {"N": [0, 0]}
              for s in ("HumanEval", "MMLU")}          # [n, n_wrong]
    rows = []
    for surf, rec, arm, task in cells():
        fr, cn = load(rec, "free", task), load(rec, arm, task)
        if not fr or not cn:
            continue
        mk = marker(rec)
        cap_f = max(i["gen_toks"] for i in fr.values())
        cap_c = max(i["gen_toks"] for i in cn.values())
        common = sorted(set(fr) & set(cn))
        n = unfin = 0
        ok_all = ok_fin = n_fin = 0
        for d in common:
            f, c = fr[d], cn[d]
            ac = correct(c, surf)
            if ac is None:
                continue
            n += 1
            capped_c = c["gen_toks"] >= cap_c - 8
            capped_f = f["gen_toks"] >= cap_f - 8
            open_think = bool(mk) and mk not in c["raw"]
            ok_all += bool(ac)
            if capped_c and open_think:
                g = "A"
                unfin += 1
            elif capped_c:
                g = "B"
            elif c["gen_toks"] > 2 * max(1, f["gen_toks"]):
                g = "C"
            elif capped_f or f["gen_toks"] > 2 * max(1, c["gen_toks"]):
                g = "D"
            else:
                g = "N"
            groups[surf][g][0] += 1
            groups[surf][g][1] += not ac
            if g != "A":
                n_fin += 1
                ok_fin += bool(ac)
        if n:
            rows.append((surf, rec, arm, n, unfin, ok_all / n,
                         ok_fin / n, ok_fin / max(1, n_fin), n_fin))
    return groups, rows


def main():
    groups, rows = decompose()
    out = os.path.join(ABLATIONS, "truncation_decomp.csv")
    with open(out, "w", newline="") as fh:
        fh.write('"# Truncation decomposition per constrained cell (TRUNCATION_RERUN_PLAN '
                 'section 1). acc_reported = scored as-is; acc_honest = generations that '
                 'hit the cap with thinking still open count as unanswered; acc_finished '
                 '= over the n_finished that emitted something. n_unfinished is group A. '
                 'Re-analysis of committed dumps; no regeneration. Producer: '
                 'analysis/residency/truncation_decomp.py"\n')
        w = csv.writer(fh)
        w.writerow(["surface", "record", "arm", "n", "n_unfinished", "acc_reported",
                    "acc_honest", "acc_finished", "n_finished"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.6f}", f"{r[6]:.6f}",
                        f"{r[7]:.6f}", r[8]])
    lab = {"A": "A. cap, thinking never closed", "B": "B. cap, answer cut off",
           "C": "C. >2x free, finished cleanly", "D": "D. blown by the FREE arm",
           "N": "Normal length"}
    print(f"{'group':32s} {'HumanEval n / wrongness':>26} {'MMLU n / wrongness':>22}")
    for g in ("A", "B", "C", "D", "N"):
        h, m = groups["HumanEval"][g], groups["MMLU"][g]
        print(f"{lab[g]:32s} {h[0]:8d} / {h[1]/max(1,h[0]):8.3f} "
              f"{m[0]:12d} / {m[1]/max(1,m[0]):8.3f}")
    for s in ("HumanEval", "MMLU"):
        c, nrm = groups[s]["C"], groups[s]["N"]
        r = (c[1] / max(1, c[0])) / max(1e-9, nrm[1] / max(1, nrm[0]))
        print(f"  {s}: group C vs normal = {r:.1f}x  (n_C={c[0]})")
    print("wrote", out, f"({len(rows)} cells)")


if __name__ == "__main__":
    main()
