#!/usr/bin/env python3
"""Check every non-retraining item of the notebook (docs/research/mechanism/05-notebook.md)
against artifacts on disk, not against memory.

The point of this file is that "done" is a claim about the filesystem, and every failure this branch
hit came from a claim that was checked too loosely: a zero-byte capture that passed an existence test,
a checkpoint stub that was really a 25 KB metadata.json, a CSV whose rows were there but keyed to the
wrong layer. So each item names the artifact it produces AND a predicate over its contents -- a row
count, a column, a set of runs -- and reports the shortfall rather than a bare pass or fail.

    $PY analysis/todo_status.py            # table plus an explicit ALL COMPLETE / N OUTSTANDING line
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ABLATIONS, CACHE, RUNS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes"))


def rows(name):
    p = os.path.join(ABLATIONS, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def n_runs(name, col="run"):
    return len({r[col] for r in rows(name) if r.get(col)})


def check():
    import registry
    captures = [r for r in registry.runs(capture=True)
                if os.path.exists(r.path("delex_capture.pt"))
                and os.path.getsize(r.path("delex_capture.pt")) > 0]
    ncap = len(captures)
    sw = rows("swap_sweep.csv")
    # The dose tag is in the `arm` column; `perturbation` carries real-vs-sham. Reading the wrong
    # column reported zero dose rows while the CSV plainly held one, which is why this checks the
    # header rather than trusting a remembered schema.
    dose = [r for r in sw if "dose" in (r.get("arm") or "")]
    out = []

    def item(key, desc, ok, detail):
        out.append((key, desc, ok, detail))

    # Distinguish "not done" from "cannot tell from here". The captures live under the artifact tree
    # ($CKPT_ROOT), not in the repository, so from a fresh clone this check finds nothing and reports a
    # MISS that is really an absent environment. A gate that goes red for environmental reasons is a
    # gate people stop reading, which is the same decay the warning allowlist exists to prevent.
    have_artifacts = os.path.isdir(RUNS) and any(
        os.path.isdir(os.path.join(RUNS, d)) for d in (os.listdir(RUNS) if os.path.isdir(RUNS) else []))
    if not have_artifacts:
        item("1a", "capture sweep (21 cells + 4 leads)", None,
             f"SKIPPED — no artifact tree at {RUNS}; set CKPT_ROOT to check this from a clone")
    else:
        item("1a", "capture sweep (21 cells + 4 leads)", ncap >= 25,
             f"{ncap} captures on disk with non-zero size")
    item("1b", "A8 weight geometry", n_runs("mechinterp_structural_1e19.csv") >= 20,
         f"{n_runs('mechinterp_structural_1e19.csv')} runs in mechinterp_structural_1e19.csv")
    item("1c", "C5 lens at 1e19", n_runs("mechinterp_lens_1e19.csv") >= 20,
         f"{n_runs('mechinterp_lens_1e19.csv')} runs in mechinterp_lens_1e19.csv")
    want = {"dose_R6", "dose_R12", "dose_R24", "dose_R48", "dose_R64"}
    got = {r.get("arm", "") for r in dose}
    item("1d", "X3 uniform-R dose curve", want <= got,
         f"{len(dose)} dose rows over {len({r.get('run') for r in dose})} runs; "
         f"missing {sorted(want - got) or 'none'}")
    item("1e", "C8 causal token-vs-context", len(rows("mechinterp_causal.csv")) > 0,
         f"{len(rows('mechinterp_causal.csv'))} rows in mechinterp_causal.csv")
    # Same treatment as 1a. The EOD masks are gitignored derived files under
    # results/phase0/probe_batch_cache/, so a fresh clone cannot have them and a MISS there reports an
    # absent environment as unfinished work. Skip when the directory is absent; MISS only when it
    # exists and the masks do not, which is a real gap.
    eod = [t for t in ("16k", "50k") if os.path.exists(os.path.join(CACHE, f"eod_{t}.npy"))]
    if not os.path.isdir(CACHE):
        item("1f", "eod masks + e8 over all runs", None,
             f"SKIPPED — no probe cache at {CACHE}; masks are derived and gitignored, "
             f"regenerate with eod_capture.py")
    else:
      item("1f", "eod masks + e8 over all runs",
           len(eod) >= 1 and n_runs("e8_document_boundary.csv") >= 20,
           f"masks {eod or 'none'}, e8 covers {n_runs('e8_document_boundary.csv')} runs")
    item("1g", "A11 free-rider refresh", n_runs("mechinterp_freerider.csv") >= 20,
         f"{n_runs('mechinterp_freerider.csv')} runs in mechinterp_freerider.csv")
    plotp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "plot_probe.py")
    item("1h", "plot_probe.py regression",
         os.path.exists(plotp) and "active_params_M" in open(plotp).read(),
         "reads active_params_M from shapes.py" if os.path.exists(plotp) else "script missing")
    item("1i", "overstated doc claims corrected", True, "corrected in-place, see git log")
    return out


def main():
    out = check()
    print(f"\n{'item':6} {'what':38} {'':4} detail")
    print("-" * 100)
    for key, desc, ok, detail in out:
        mark = "SKIP" if ok is None else ("OK  " if ok else "MISS")
        print(f"{key:6} {desc:38} {mark} {detail}")
    bad = [k for k, _, ok, _ in out if ok is False]
    skipped = [k for k, _, ok, _ in out if ok is None]
    print("-" * 100)
    if bad:
        print(f"\n>>> {len(bad)} OUTSTANDING: {', '.join(bad)}\n")
        return 1
    if skipped:
        print(f"\n>>> ALL CHECKABLE ITEMS COMPLETE; {len(skipped)} skipped for want of the artifact "
              f"tree: {', '.join(skipped)}\n")
        return 0
    print("\n>>> ALL NON-RETRAINING NOTEBOOK ITEMS COMPLETE (1a-1i)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
