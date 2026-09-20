#!/usr/bin/env python3
"""Re-score MMLU dumps with the corrected policy: an unfinished thought is not an answer.

A response that exhausts its budget INSIDE the thinking block never emits an
answer. The old scoring split the text on an absent end-of-think marker, which
returns the whole trace, so the answer extractor read raw deliberation -- and
deliberation is full of "if X then the answer is (D), if not then (B)" asides.
The extractor then scores the model's scratch work, well above chance and well
below its real accuracy, blending two populations into one number.

This recomputes each cell three ways from the committed dumps (no regeneration):
  reported  - what the cell currently reports (extractor ran on everything)
  honest    - unfinished responses count as unanswered (the corrected policy)
  finished  - accuracy on the finished subset only, with n stated
`finished` is the model's real accuracy at that budget; `honest` is what the
benchmark should report, since running out of budget is a failure to answer.
The gap between them measures how much of the cell is budget rather than ability.

Writes results/ablations/mmlu_unfinished_rescore.csv.
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
        "gemma4_think_on": "<channel|>", "gptoss": "<|channel|>final<|message|>"}


def marker(rec):
    for pre, mk in MARK.items():
        if rec.startswith(pre):
            return mk
    return MARK["gptoss"] if rec.startswith("gptoss") else None


def main():
    rows = []
    for p in sorted(glob.glob(f"{SAMP}/*_mmlu_dual.json")):
        base = os.path.basename(p)[: -len("_mmlu_dual.json")]
        m = re.match(r"(.+)_(free|R\d+)$", base)
        if not m:
            continue
        rec, arm = m.group(1), m.group(2)
        mk = marker(rec)
        items = json.load(open(p))["items"]
        if mk is None or len(items) < 100:
            continue
        cap = max(i["gen_toks"] for i in items)      # budget inferred from the cell
        n = fin = ok_all = ok_fin = 0
        for i in items:
            g = i.get("gold")
            if g is None:
                continue
            n += 1
            unfinished = mk not in i["raw"] and i["gen_toks"] >= cap - 8
            hit = i.get("pred_relaxed") == g
            ok_all += hit
            if not unfinished:
                fin += 1
                ok_fin += hit
        rows.append((rec, arm, n, n - fin, ok_all / n, ok_fin / n,
                     ok_fin / max(1, fin)))
    out = os.path.join(ABLATIONS, "mmlu_unfinished_rescore.csv")
    with open(out, "w", newline="") as fh:
        fh.write('"# MMLU re-scored from committed dumps under the corrected policy: '
                 'a response that hit the generation cap while still inside its '
                 'thinking block emitted no answer, so the extractor must not read '
                 'its deliberation. reported = extractor ran on everything (the '
                 'contaminated number); honest = unfinished count as unanswered; '
                 'finished = accuracy over the finished subset only (n_finished = '
                 'n - n_unfinished). Producer: '
                 'analysis/residency/mmlu_unfinished_rescore.py"\n')
        w = csv.writer(fh)
        w.writerow(["record", "arm", "n", "n_unfinished", "acc_reported",
                    "acc_honest", "acc_finished"])
        print(f"{'cell':30s} {'n':>4} {'unfin':>6} {'reported':>9} {'honest':>7} "
              f"{'finished':>9}")
        for rec, arm, n, unf, a, h, f in rows:
            w.writerow([rec, arm, n, unf, f"{a:.6f}", f"{h:.6f}", f"{f:.6f}"])
            flag = "   <-- budget-contaminated" if unf > n * 0.05 else ""
            print(f"{rec + ' ' + arm:30s} {n:4d} {unf:6d} {a:9.4f} {h:7.4f} "
                  f"{f:9.4f}{flag}")
    print("wrote", out)


if __name__ == "__main__":
    main()
