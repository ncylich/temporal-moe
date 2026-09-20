#!/usr/bin/env python3
"""Refuse to replace a complete results CSV with a partial one.

Nine scripts under `analysis/probes/` share one pattern: they take a run list from argv

    only = [a for a in sys.argv[1:] if not a.startswith("--")]

filter the registry by it, and then write the result with `open(OUT, "w")` — no read of what is on
disk, no merge, no comparison. Run any of them on a subset for a quick check and a complete committed
file is silently replaced by a partial one, exit status 0. It has happened twice: a two-run diagnostic
cut `mechinterp_locus_1e19.csv` from 87552 rows to 9216, and a `--split position` run cut
`mechinterp_locus_slopes.csv` from 24 rows to 17.

`guard(path, rows, key_index)` aborts when the new row set is smaller than the file already there, or
when it drops runs the file already covers, and names what would be lost. Losing rows is sometimes
right — a run genuinely retired, a metric genuinely narrowed — so `--replace` on the command line
allows it. What is not right is losing them without anyone deciding to.

    from safe_csv import guard
    guard(OUT, rows, key_index=HEADER.index("run"))
    with open(OUT, "w", newline="") as f:
        ...
"""
import csv
import os
import sys


def guard(path, rows, key_index=None, allow_flag="--replace"):
    """Abort if writing `rows` to `path` would shrink it or drop runs it already covers.

    `key_index` is the position of the run/label column within each row of `rows`; when given, the
    check also compares the *set* of runs, which catches a swap of one run for another at equal row
    count. Passing `allow_flag` on the command line permits the shrink.
    """
    if not os.path.exists(path) or allow_flag in sys.argv:
        return
    with open(path) as f:
        prior = list(csv.DictReader(f))
    if not prior:
        return

    problems = []
    if len(rows) < len(prior):
        problems.append(f"{len(prior)} rows on disk, {len(rows)} computed now")

    if key_index is not None:
        cols = list(prior[0])
        kcol = next((c for c in ("run", "label", "model") if c in cols), None)
        if kcol:
            had = {r[kcol] for r in prior if r.get(kcol)}
            now = {r[key_index] for r in rows if len(r) > key_index}
            lost = sorted(had - now)
            if lost:
                problems.append(f"would drop {len(lost)} run(s): {lost}")

    if problems:
        sys.exit(
            f"[abort] refusing to shrink {os.path.basename(path)}:\n"
            + "".join(f"         - {p}\n" for p in problems)
            + f"         This usually means the script was run on a subset. It rewrites the whole\n"
              f"         file from whatever it was given, so a partial run replaces a complete file.\n"
              f"         Re-run over everything, or pass {allow_flag} if the loss is intended.")
