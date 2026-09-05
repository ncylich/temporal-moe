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

# Standing verdicts. A flag with a recorded reason is answered, not silenced -- the point of the
# linter is that every flag gets a verdict, and one repeated 30 times does not need re-deciding.
# Anything not matched here is reported and needs a fresh judgement.
#   (flag substring, reason)
VERDICTS = (
    ("CONSTANT COLUMN seed", "training seed, fixed at 1234 by design"),
    ("CONSTANT COLUMN n_tokens", "every row scores the SAME audited held-out slice; that it is\n     constant is the point -- a varying value here would mean the regimes were not matched"),
    ("CONSTANT COLUMN budget", "one file per compute budget; constant is the file's definition"),
    ("CONSTANT COLUMN k ", "top-k is an architectural constant within a grain"),
    ("CONSTANT COLUMN topk_k", "as above"),
    ("CONSTANT COLUMN E ", "expert count is an architectural constant within a grain"),
    ("CONSTANT COLUMN num_experts", "as above"),
    ("CONSTANT COLUMN hidden", "hidden size is fixed within a shape"),
    ("CONSTANT COLUMN iters", "iteration count fixed by the isoFLOP budget"),
    ("CONSTANT COLUMN first_layer", "MoE layers start at 2 in every config; layer 1 is dense"),
    ("CONSTANT COLUMN layer ", "single-layer probe by design"),
    ("CONSTANT COLUMN window_w", "fixed probe window"),
    ("CONSTANT COLUMN stride", "fixed probe stride"),
    ("CONSTANT COLUMN positions_per_seq", "fixed by the probe geometry"),
    ("CONSTANT COLUMN n_sequences", "fixed evaluation batch"),
    ("CONSTANT COLUMN n_fit_rows", "fit sample capped at a constant"),
    ("CONSTANT COLUMN base_rate", "k/E, an architectural constant"),
    ("CONSTANT COLUMN divisor", "a fixed normalisation constant"),
    ("CONSTANT COLUMN baseline_CE", "one baseline per file by construction"),
    ("CONSTANT COLUMN random_pct", "k/E expressed as a percentage"),
    ("CONSTANT COLUMN swaprate", "a fixed-rate arm; the rate is the arm's definition"),
    ("CONSTANT COLUMN expert_swap_bytes", "bytes per expert are fixed by the architecture"),
    ("CONSTANT COLUMN packs", "fixed pack count for the eval"),
    ("CONSTANT COLUMN seqlen", "sequence length fixed at 2048 throughout"),
    ("CONSTANT COLUMN ref_", "reference constants, identical across rows by definition"),
    ("CONSTANT COLUMN train_tokens", "fixed token budget for the cell"),
    ("CONSTANT COLUMN ubatch", "fixed micro-batch for the serving benchmark"),
    ("CONSTANT COLUMN context", "fixed context length for the serving benchmark"),
    ("EMPTY COLUMN geometry_note", "records why weight geometry was skipped; empty means it was not"),
    ("EMPTY COLUMN val_ce", "this probe recorded test CE only"),
)


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
        # Same comment filter as the working-file read. Filtering one side and not the other makes
        # every commented file look like it shrank -- which it did, on this linter's own output.
        # .lstrip('"') as well: a comment row containing commas is quoted by csv.writer, so a bare
        # startswith("#") misses it here while the working-file read strips the quote and skips it.
        # That one-character asymmetry is itself the failure this docstring warns about, and it was
        # fixed on one side only -- it is the real cause of the standing phantom shrink on
        # layer_freeing_downstream.csv, which had been attributed to CRLF line endings.
        return list(csv.DictReader([ln for ln in out.splitlines()
                                    if not ln.lstrip().lstrip('"').startswith("#")]))
    except subprocess.CalledProcessError:
        return None


def check(path):
    rel = os.path.relpath(path, REPO)
    with open(path) as f:
        # Some files carry '#' provenance lines above the header. DictReader would take the first of
        # them as the header, turning prose fragments into column names -- which is exactly how this
        # linter reported '8 packs' and 'D=3.1089.' as empty columns.
        # Provenance comments appear both bare and quoted -- '# note' and '"# note'. Skipping only
        # the bare form left the quoted one to be read as the header, which then made every real row
        # look like it had extra fields. That produced three spurious SHIFTED reports.
        lines = [ln for ln in f if not ln.lstrip().lstrip('"').startswith("#")]
    rows = list(csv.DictReader(lines))
    out = []
    if not rows:
        return [f"{rel}: EMPTY (no data rows)"]
    # A ragged row yields a None key from DictReader's restkey; drop it rather than crash.
    cols = [c for c in rows[0] if isinstance(c, str)]

    # Field count must match the header. An unquoted separator inside a value silently shifts every
    # column after it -- DictReader then reads one field as another, e.g. a schedule fragment '3:18'
    # parsed as a loss. This linter passed a file with two such rows, so the check was missing.
    # Distinguish two very different things that both make a row's field count differ:
    #
    #   too MANY fields  an unquoted separator inside a value. Every column after it shifts, so a
    #                    reader silently gets the wrong data -- this is the real defect, and it is how
    #                    A5's test_CE came to be read as the string '3:18'.
    #   too FEW fields   a trailing value simply absent. Nothing shifts; the reader gets None for the
    #                    missing tail. Some files hold heterogeneous row types by design --
    #                    stability_trunk.csv writes resid_absmean only on residual rows, per its
    #                    writer's docstring -- so this is reported separately and only as a note.
    over = sum(1 for r in rows if None in r)                       # DictReader restkey => extra fields
    under = sum(1 for r in rows if any(v is None for v in r.values()))
    if over:
        out.append(f"{rel}: SHIFTED ROWS {over}/{len(rows)} have MORE fields than the header — an "
                   f"unquoted separator inside a value shifts every later column")
    if under:
        out.append(f"{rel}: SHORT ROWS {under}/{len(rows)} omit trailing field(s) — no column shift; "
                   f"check whether the file holds more than one row type by design")

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
    answered, open_ = [], []
    for f in findings:
        why = next((r for pat, r in VERDICTS if pat in f), None)
        (answered if why else open_).append((f, why))
    for f, why in open_:
        print(f"  [flag] {f}")
    if answered and not quiet:
        print(f"  ({len(answered)} flag(s) with standing verdicts, suppressed — see VERDICTS)")
    findings = [f for f, _ in open_]
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
