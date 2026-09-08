#!/usr/bin/env python3
"""Read a retrained adapter's grid and set it beside the PUBLISHED Section 8 row.

This is a REPLICATION, not a reproduction, and the table says so. The pool it trained on
was rebuilt from scratch (RECOVER_DATA_PLAN 1.1): the original was substantially
self-generated -- math_selfgen, Magicoder-style code, format drills, mcq-writer -- and only
its chat and math_user lanes came from a real corpus, whereas the rebuild is real-corpus
throughout. Roughly half the pool is therefore different in kind, and mcq-writer (691) is
absent entirely. gemma_adapt_RESULTS also records +-2pt single-run screening noise.

So a cell landing near but not on its published value is the expected outcome and not
evidence of a defect. Cells landing far off are worth investigating; the pre-registered
first suspect for MMLU specifically is the missing mcq-writer lane, the only
MMLU-format-facing lane in the original pool.

    compare_to_published.py --record gemma4_ce_rebuild --dual gemma4_ce_rebuild_dual
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.paths import ABLATIONS                                 # noqa: E402

# published D12 / r2 damage vs the UNCONSTRAINED base, in points (gemma_adapt_RESULTS.md)
PUBLISHED = {
    "gemma4": {"label": "D12 (published)",
               "R8": {"GSM8K": 0.0, "IFEval": -1.0, "HumanEval": -1.2, "MMLU": -1.8},
               "base_R8": {"GSM8K": -6.0, "IFEval": 0.0, "HumanEval": -6.1, "MMLU": -0.2}},
    "qwen35": {"label": "d12r2 (published)",
               "R8": {"GSM8K": -3.5, "IFEval": -6.0, "HumanEval": -1.2, "MMLU": -0.4},
               "R16": {"GSM8K": -3.5, "IFEval": -6.0, "HumanEval": -1.8, "MMLU": -1.3}},
}
# task -> (csv task name, metric) ; MMLU comes from mmlu_gptoss.py under the _dual record
TASKS = [("GSM8K", "gsm8k_cot_zeroshot", "exact_match,flexible-extract", False),
         ("IFEval", "ifeval", "prompt_level_strict_acc,none", False),
         ("HumanEval", "humaneval", "pass@1,create_test", False),
         ("MMLU", "mmlu_gptoss_relaxed", "acc,relaxed-extract", True)]


def load(csv_name="instruct_genbench_vllm.csv"):
    p = os.path.join(ABLATIONS, csv_name)
    rows = [l for l in open(p) if not l.startswith('"#')]
    out = {}
    for x in csv.DictReader(rows):
        out[(x["model"], x["arm"], x["task"], x["metric"])] = float(x["value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--dual", required=True, help="record carrying the relaxed MMLU rows")
    ap.add_argument("--family", default="gemma4", choices=sorted(PUBLISHED))
    ap.add_argument("--arms", default="R8,R16")
    A = ap.parse_args()
    G = load()
    pub = PUBLISHED[A.family]

    print(f"\n{A.record}: retrained adapter vs {pub['label']}")
    print("Damage in points vs this run's OWN free arm; negative = worse than free.\n")
    for arm in A.arms.split(","):
        if arm not in pub:
            continue
        print(f"  arm {arm}")
        print(f"    {'task':10s} {'this run':>10} {'published':>10} {'delta':>8}")
        for label, task, metric, is_dual in TASKS:
            rec = A.dual if is_dual else A.record
            free = G.get((rec, "free", task, metric))
            got = G.get((rec, arm, task, metric))
            if free is None or got is None:
                print(f"    {label:10s} {'(missing)':>10}")
                continue
            dmg = 100.0 * (got - free)
            p = pub[arm][label]
            print(f"    {label:10s} {dmg:+10.1f} {p:+10.1f} {dmg - p:+8.1f}")
        print()
    print("Read as a replication: the pool was rebuilt, roughly half of it differs in kind")
    print("from the original (real-corpus where the original self-generated), mcq-writer is")
    print("absent, and single-run screening noise is +-2pt. Near is expected; far is a lead.")


if __name__ == "__main__":
    main()
