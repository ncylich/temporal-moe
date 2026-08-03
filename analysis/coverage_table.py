#!/usr/bin/env python3
"""Generate the coverage table in docs/research/mechanism/04-coverage.md from the CSVs it describes.

That table is a hand-maintained copy of numbers that live in `results/ablations/*.csv`, and it has
drifted twice: rows claimed a metric covered 2, 3 or 12 models while its CSV held 26, and a warning
paragraph said e8 had "regressed to zero rows" directly below a row reading 22. Both times the repair
was applied only to the rows someone happened to name, so the rest stayed wrong.

A copy of a number will drift. The durable fix is to stop maintaining it:

    $PY analysis/coverage_table.py            # print the table
    $PY analysis/coverage_table.py --write    # splice it into the doc between the markers

The doc carries `<!-- COVERAGE:BEGIN -->` / `<!-- COVERAGE:END -->` and everything between them is
regenerated. Editing inside the markers by hand is pointless — the next run overwrites it.

Model counts come from the CSV's own key column, so a metric that gains runs shows up without anyone
remembering to update prose. Layer ranges likewise. A file that is absent or empty says so rather than
silently keeping a stale number, which is the failure this replaces.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ABLATIONS

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "research", "mechanism", "04-coverage.md")
BEGIN, END = "<!-- COVERAGE:BEGIN -->", "<!-- COVERAGE:END -->"

# (label, csv glob relative to ABLATIONS, per-layer?)
METRICS = [
    ("locus probes (A_tok, A_ctx)",              "mechinterp_locus*.csv",        "yes"),
    ("null floors",                              "mechinterp_floors*.csv",       "yes"),
    ("token-id oracle ceiling (C7)",             "mechinterp_oracle.csv",        "yes"),
    ("frequency-stratified token AUC (C9)",      "mechinterp_freqstrat.csv",     "yes"),
    ("cross-layer probe transfer (C10)",         "mechinterp_transfer.csv",      "layer pairs"),
    ("per-layer constraint swap (C3)",           "swap_sweep.csv",               "yes"),
    ("constraint cost shape (C3 decomposition)", "swap_shape.csv",               "summary"),
    ("magnitude-matched sham (N1)",              "sham_magnitude_matched.csv",   "yes"),
    ("causal token/context substitution (C8)",   "mechinterp_causal.csv",        "yes"),
    ("null battery (which null is valid)",       "mechinterp_null_battery.csv",  "one layer"),
    ("output logit lens (effective vocab)",      "mechinterp_lens*.csv",         "yes"),
    ("cache hit rate",                           "e6_per_layer_ranking.csv",     "yes"),
    ("swap rate / burst length",                 "e1_swap_rate_by_layer.csv",    "yes"),
    ("document-boundary churn (e8)",             "e8_document_boundary.csv",     "n/a"),
    ("selectivity, generalists, router entropy", "mechinterp_structural*.csv",   "yes"),
    ("demand forecastability (A10/C6)",          "mechinterp_demand*.csv",       "yes"),
    ("free-rider / tokens-per-expert (A11)",     "mechinterp_freerider.csv",     "n/a"),
    ("per-layer cost vs churn (N7)",             "n7_cost_vs_churn.csv",         "yes"),
]

KEYS = ("run", "label", "model")          # whichever the file actually uses, in preference order

# Files that exist but must not be cited for current claims. The coverage table above lists what is
# measured; without this second table a reader cannot tell a file that was never measured from one
# that was measured and then superseded, and silence about a file is indistinguishable from the file
# not existing. Membership of `superseded/` is read from disk; the rest are judgments and are listed.
SUPERSEDED_DIR = os.path.join(ABLATIONS, "superseded")

NOT_CURRENT = [
    ("mechinterp_structural.csv",
     "superseded method (pooled across layers), but the sole record of 6 runs the per-layer "
     "replacement never held, 4 of them unrecoverable; also pooled effective rank"),
    ("mechinterp_lens.csv",
     "superseded method, but the sole record of 2 runs whose checkpoints are gone"),
    ("specialization_summary.csv",
     "sole structural record of 5 router variants absent from mechinterp_structural.csv"),
    ("specialization_m3.csv",
     "sole geometry record of the same 5 router variants"),
    ("specialization_probe.csv",
     "per-expert detail behind the two above, same 5 variants"),
    ("oracle_a3.csv", "no producer was ever committed; 0 of 20 runs have a preserved router log"),
    ("oracle_horizon_map.csv", "no producer; 0 of 4 runs preserved"),
    ("block_replay.csv", "no producer; 0 of 11 runs preserved"),
    ("anomaly_pred.csv", "no producer; 0 of 5 runs preserved"),
    ("karen_center_replay.csv", "no producer; neither run preserved"),
    ("karen_promotion_s2_1e17.csv", "no producer; eval-only and the checkpoint is gone"),
    ("unmask_eval.csv", "no producer; eval-only, most 1e16/1e17 checkpoints gone"),
    ("unmask_eval_1e19.csv", "no producer, but eval-only and its 1e19 checkpoints survive"),
    ("momr_replay.csv", "no producer, but both runs have preserved logs — reconstructible"),
]


def not_current_table():
    """Archived files (read from disk) plus in-place files that must not carry current claims."""
    out = ["| file | where | why it is not current |", "|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(SUPERSEDED_DIR, "*.csv"))):
        out.append(f"| `{os.path.basename(p)}` | `superseded/` | "
                   f"every run it holds is covered by its replacement |")
    for name, why in NOT_CURRENT:
        present = os.path.exists(os.path.join(ABLATIONS, name))
        where = "`results/ablations/`" if present else "**absent**"
        out.append(f"| `{name}` | {where} | {why} |")
    return "\n".join(out)


def survey(pattern):
    """-> (n_models, layer_range, n_rows, files). Absent or empty files report as such."""
    paths = sorted(glob.glob(os.path.join(ABLATIONS, pattern)))
    if not paths:
        return None, None, 0, []
    models, layers, rows = set(), set(), 0
    for p in paths:
        with open(p) as f:
            rd = csv.DictReader(f)
            key = next((k for k in KEYS if rd.fieldnames and k in rd.fieldnames), None)
            for r in rd:
                rows += 1
                if key and r.get(key):
                    models.add(r[key])
                v = r.get("layer", "")
                if v and v.replace("-", "").isdigit() and v != "-":
                    layers.add(int(v))
    rng = f"{min(layers)}–{max(layers)}" if layers else "—"
    return len(models), rng, rows, [os.path.basename(p) for p in paths]


def table():
    out = ["| metric | file | per-layer? | layers | models | rows |",
           "|---|---|---|---|---|---|"]
    missing = []
    for label, pat, per in METRICS:
        n, rng, rows, files = survey(pat)
        if n is None:
            out.append(f"| {label} | `{pat}` | {per} | — | **absent** | 0 |")
            missing.append(pat)
            continue
        fl = ", ".join(f"`{x}`" for x in files)
        out.append(f"| {label} | {fl} | {per} | {rng} | **{n}** | {rows} |")
    return "\n".join(out), missing


def main():
    md, missing = table()
    stamp = ("*Generated by `analysis/coverage_table.py` from `results/ablations/*.csv`. "
             "Do not edit by hand — run the script. Model counts are distinct values of the CSV's own "
             "key column; layer ranges are the min and max of its `layer` column.*")
    nc = ("### Files that are not current\n\n"
          "*Also generated. Archived files are read from `results/ablations/superseded/`; the rest sit "
          "in `results/ablations/` because each is the sole surviving record of at least one run. "
          "Verdicts and run lists are in that directory's `README.md`.*\n\n"
          + not_current_table())
    block = f"{BEGIN}\n\n{stamp}\n\n{md}\n\n{nc}\n\n{END}"
    if "--write" in sys.argv:
        s = open(DOC).read()
        if BEGIN in s and END in s:
            pre, rest = s.split(BEGIN, 1)
            _, post = rest.split(END, 1)
            open(DOC, "w").write(pre + block + post)
            print(f"[write] spliced into {DOC}")
        else:
            sys.exit(f"markers not found in {DOC}; add {BEGIN} / {END} around the table first")
    else:
        print(block)
    if missing:
        print(f"\n[warn] {len(missing)} metric(s) have no CSV on disk: {', '.join(missing)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
