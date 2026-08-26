#!/usr/bin/env python3
"""Paired significance for residency-gap comparisons, from the committed sample dumps.

Every arm number in this program is a DIFFERENCE of differences: (constrained - free)
for an adapted model, compared against (constrained - free) for the base. Both levels
are paired on the same questions, so the right error bar is McNemar's, not a binomial
on the accuracies -- using the latter overstates precision on the within-record gap and
understates it on the cross-record one.

At the sampled n=200 the cross-record SE is ~3.0 points, which is wider than every
adapted-arm effect measured on the D7 rebuild (2026-08-26). Run this before concluding
that an arm helped.

    arm_power.py --task gsm8k_cot_zeroshot --base gemma4_instruct --arm R8 REC [REC...]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")


def load(rec, arm, task):
    """doc_id -> correct. Each doc appears once per metric (strict + flexible);
    flexible-extract dominates strict, and strict is identically 0 for models that
    emit channel markers, so the max over a doc's rows is the flexible score."""
    p = f"{SAMP}/{rec}_{arm}_{task}.json"
    if not os.path.exists(p):
        return None
    out = {}
    for i in json.load(open(p))["items"]:
        # gsm8k/mmlu dumps key on doc_id and carry exact_match once per metric;
        # the HumanEval producer keys on doc and carries a boolean pass instead.
        k = i["doc_id"] if "doc_id" in i else i["doc"]
        v = float(i["exact_match"]) if "exact_match" in i else float(bool(i["pass"]))
        out[k] = max(out.get(k, 0.0), v)
    return {k: bool(v) for k, v in out.items()}


def gap(rec, arm, task):
    """(delta, se, n) for constrained-minus-free, paired by question (McNemar)."""
    f_, c = load(rec, "free", task), load(rec, arm, task)
    if not f_ or not c:
        return None
    k = sorted(set(f_) & set(c))
    n = len(k)
    b = sum(1 for d in k if f_[d] and not c[d])
    cc = sum(1 for d in k if c[d] and not f_[d])
    return (cc - b) / n, math.sqrt(b + cc) / n, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--task", default="gsm8k_cot_zeroshot")
    ap.add_argument("--arm", default="R8")
    ap.add_argument("--base", default="gemma4_instruct")
    A = ap.parse_args()

    ref = gap(A.base, A.arm, A.task)
    print(f"{A.arm} minus free, paired by question. delta<0 = residency hurts.\n")
    print(f"{'record':<30}{'n':>6}{'delta':>8}{'SE':>6}{'z':>7}")
    rows = [A.base] + [r for r in A.records if r != A.base]
    got = {}
    for r in rows:
        g = gap(r, A.arm, A.task)
        if g is None:
            print(f"{r:<30}{'-- no dump --':>27}")
            continue
        d, s, n = got.setdefault(r, g)
        print(f"{r:<30}{n:>6}{100*d:>+8.1f}{100*s:>6.1f}{d/s if s else 0:>+7.2f}")

    if ref is None:
        return
    bd, bs, _ = ref
    print(f"\nvs base ({A.base}); a real improvement needs |z| > 1.96")
    for r, (d, s, _) in got.items():
        if r == A.base:
            continue
        diff, se = d - bd, math.sqrt(s * s + bs * bs)
        print(f"  {r:<28}{100*diff:>+6.1f} +/- {100*se:4.1f}  z={diff/se:>+5.2f}  "
              f"{'REAL' if abs(diff) > 1.96 * se else 'not resolved'}")


if __name__ == "__main__":
    main()
