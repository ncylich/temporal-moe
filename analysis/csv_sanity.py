#!/usr/bin/env python3
"""Flag committed CSVs whose contents stopped meaning anything, even though they parse and reproduce.

Idempotence is not sanity. A bootstrap that emits `lo == hi` on every row regenerates byte-identically
forever and passes every reproducibility check, while reporting confidence intervals of zero width —
that shipped here, in 12 of 24 rows, and no existing check saw it. Faithfulness is not sanity either:
a file can match its committed copy exactly and still be a table of nans.

Checks, all over `results/ablations/*.csv`:

  degenerate interval   a `*_lo95` whose paired `*_hi95` is equal on some rows — an interval of zero
                        width, which is the specific failure above
  constant column       a numeric column with one distinct value across every row; usually a knob that
                        stopped varying, or a metric that silently became a constant
  empty column          entirely blank or nan
  shrunk file           fewer rows, or fewer distinct runs, than the committed version — the class-B
                        trap, where a partial regeneration overwrites a complete file

Exit status is 1 if anything is flagged, so it can gate a reproduction pass.

    $PY analysis/csv_sanity.py            # check the working tree against HEAD
    $PY analysis/csv_sanity.py --quiet    # findings only

A flag is not automatically a defect. Some columns are legitimately constant (a fixed batch size) and
some files legitimately shrink. Report what it flags either way — a legitimate flag that goes unstated
looks the same as one nobody looked at.
"""
import csv
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ABLATIONS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def committed(rel):
    """The HEAD version of a file, as rows; empty if untracked."""
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout
        return list(csv.DictReader(out.splitlines()))
    except subprocess.CalledProcessError:
        return None


def check(path):
    rel = os.path.relpath(path, REPO)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    if not rows:
        return [f"{rel}: EMPTY (no data rows)"]
    # A ragged row yields a None key from DictReader's restkey; drop it rather than crash.
    cols = [c for c in rows[0] if isinstance(c, str)]

    # degenerate intervals: lo == hi on any row
    for lo in [c for c in cols if c.endswith("_lo95")]:
        hi = lo[:-5] + "_hi95"
        if hi not in cols:
            continue
        bad = [r for r in rows if _num(r[lo]) is not None and _num(r[lo]) == _num(r[hi])]
        if bad:
            out.append(f"{rel}: DEGENERATE INTERVAL {lo}=={hi} on {len(bad)}/{len(rows)} rows "
                       f"— zero-width confidence interval")

    for c in cols:
        vals = [r[c] for r in rows]
        if all(v in ("", "nan", "NaN", None) for v in vals):
            out.append(f"{rel}: EMPTY COLUMN {c}")
            continue
        nums = [_num(v) for v in vals]
        nums = [x for x in nums if x is not None]
        if len(nums) == len(rows) and len(rows) > 2 and len(set(nums)) == 1:
            out.append(f"{rel}: CONSTANT COLUMN {c} = {nums[0]} on all {len(rows)} rows")

    prior = committed(rel)
    if prior:
        if len(rows) < len(prior):
            out.append(f"{rel}: SHRANK {len(prior)} -> {len(rows)} rows vs committed")
        key = "run" if "run" in cols else ("label" if "label" in cols else None)
        if key and key in (prior[0] if prior else {}):
            a, b = {r[key] for r in prior}, {r[key] for r in rows}
            if a - b:
                out.append(f"{rel}: LOST RUNS vs committed: {sorted(a - b)}")
    return out


def main():
    quiet = "--quiet" in sys.argv
    findings, n = [], 0
    for p in sorted(glob.glob(os.path.join(ABLATIONS, "*.csv"))):
        n += 1
        findings += check(p)
    if not quiet:
        print(f"checked {n} CSVs in {ABLATIONS}")
    for f in findings:
        print(f"  [flag] {f}")
    if findings:
        print(f"\n{len(findings)} flag(s). A flag is not automatically a defect — a constant column "
              f"may be a fixed setting, and a file may legitimately shrink. State the verdict either "
              f"way rather than leaving it unexamined.")
        return 1
    if not quiet:
        print("no flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
