#!/usr/bin/env python3
"""Re-score mmlu_dual dumps with lm_eval's own flan-CoT `get-answer` filter.

CAVEAT, READ BEFORE CITING: this filter takes the FIRST "answer is" in the text.
On think-in-text models whose traces argue with themselves ("...the answer is not
simply X..."), that first hit is a mid-reasoning aside, so the extracted letter is
stray or absent. On qwen thinking-on it disagrees with the harness's own strict
metric on 26 of 228 items IN BOTH DIRECTIONS. The numbers here are therefore a
diagnostic of EXTRACTION, not an era-comparable accuracy for such models -- use
them to show that stock extraction floors harmony/thinking formats (which they do,
decisively, on gpt-oss), and use the harness's own acc,strict-flan when checking a
regenerated cell against a mmlu_flan_cot_fewshot grid row.

The relaxed harness (mmlu_gptoss.py) writes two metrics: `acc,relaxed-extract`
(the reported one) and `acc,strict-flan` (its own "The answer is (X)" regex).
Neither is the metric behind the older `mmlu_flan_cot_fewshot` grid rows, whose
filter is lm_eval's

    regex_pattern: "(?<=answer is )(.*)(?=.)"   then take_first, exact_match

which also accepts "The correct answer is (C)" -- phrasing that the harness's
stricter regex rejects. Comparing the two across eras understates the newer runs
by 17-26 points on gemma. This producer applies the genuine lm_eval filter to
the dumped scored text, so a regenerated cell can be checked against its grid
row like with like. Re-analysis only: never regenerates.

Writes results/ablations/mmlu_flan_rescore.csv.
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
FLAN = re.compile(r"(?<=answer is )(.*)(?=.)")      # lm_eval get-answer, verbatim
LETTER = re.compile(r"([A-D])")


def flan_extract(text):
    """lm_eval get-answer + take_first, then the letter of the extracted span
    (the grid's exact_match compares against the '(X)' target string; taking the
    letter makes the comparison robust to trailing markup like '**(D)**')."""
    m = FLAN.search(text)
    if not m:
        return None
    lm = LETTER.search(m.group(1))
    return lm.group(1) if lm else None


def main():
    out = os.path.join(ABLATIONS, "mmlu_flan_rescore.csv")
    rows = []
    for p in sorted(glob.glob(f"{SAMP}/*_mmlu_dual.json")):
        b = json.load(open(p))
        items = b["items"]
        if len(items) < 100:                 # early 4-item overwrite-bug files
            continue
        name = os.path.basename(p)[: -len("_mmlu_dual.json")]
        m = re.match(r"(.+)_(free|R\d+)$", name)
        if not m:
            continue
        rec, arm = m.group(1), m.group(2)
        n = hit = miss = 0
        for i in items:
            g = i.get("gold")
            if g is None:
                continue
            n += 1
            pred = flan_extract(i["text"])
            miss += pred is None
            hit += pred == g
        if n:
            rows.append((rec, arm, n, hit / n, miss))
    with open(out, "w", newline="") as fh:
        fh.write('"# mmlu_dual dumps re-scored with lm_eval\'s own flan-CoT '
                 'get-answer filter ("(?<=answer is )(.*)(?=.)" + take_first), the '
                 'metric behind the mmlu_flan_cot_fewshot grid rows. Use THIS column '
                 'to check a regenerated cell against its grid row; the harness\'s '
                 'own acc,strict-flan is a stricter regex (rejects "the correct '
                 'answer is") and is not era-comparable. Re-analysis of committed '
                 'dumps. Producer: analysis/residency/mmlu_flan_rescore.py"\n')
        w = csv.writer(fh)
        w.writerow(["record", "arm", "n", "acc_flan_getanswer", "unextracted"])
        for rec, arm, n, acc, miss in rows:
            w.writerow([rec, arm, n, f"{acc:.6f}", miss])
            print(f"{rec:28s} {arm:5s} n={n} flan-get-answer={acc:.4f} "
                  f"({miss} unextracted)")
    print("wrote", out)


if __name__ == "__main__":
    main()
