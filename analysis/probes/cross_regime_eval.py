#!/usr/bin/env python3
"""Cross-regime evaluation: what it costs to change residency regime at evaluation time.

Two directions, both measured on a model that never saw the other regime in training:

    removal      a temporal-trained model evaluated with residency OFF
    imposition   a full-MoE model evaluated with residency ON at the paired R

`unmask_eval.csv` and `unmask_eval_1e19.csv` hold eight such cells and **neither has ever had a
producer, in any commit on any branch**. That is the defect this file exists to close. It is also
why three of those eight cells can no longer be re-derived: the CSVs record a short cell label
(`L0`, `minlogit`, `v16k`) and not a run directory, and `L0` alone matches seventeen runs under
`$CKPT_ROOT`. A label is not provenance. Every row this script writes carries the run name, the
checkpoint iteration and the divisor.

**Both arms are scored on identical batches.** `sweep_eval.py` caches the evaluation micro-batches on
the first arm and replays those same tensors for the second, so the delta is attributable to the
regime and nothing else. Evaluating the two regimes in separate processes would advance Megatron's
data iterator and score them on different documents -- a difference that looks exactly like a result.

**The checkpoint iteration is chosen here, not inherited.** Megatron loads whatever
`latest_checkpointed_iteration.txt` names, and in this repository that file is wrong for at least
three runs: `moe_coarse_1e19`, `g1_tmoe_coarse_1e19` and `g3_tmoe_s2_1e17` all name iteration 10
while their trained checkpoints are `iter_0004318`, `iter_0004318` and `iter_0003861`. A producer
that trusts it evaluates a ten-step model and reports numbers that look plausible and mean nothing.
This resolves the highest real `iter_*` directory, says so, and restores the file afterwards.

    cross_regime_eval.py --list                      # show the cell table, run nothing
    cross_regime_eval.py --cell temporal_coarse_1e19 # one cell
    cross_regime_eval.py --all --out results/ablations/cross_regime_eval.csv

BPB = CE_nats / divisor, divisor = ln2 * bytes_per_token, a property of the tokenizer and corpus:
2.7568 for this repository's 16k BPE, 2.9780 for pythia-50k (docs/EVALUATION_METHODOLOGY.md §1).
It is recorded per row and never inherited across tokenizers.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/workspace/FLAME-MoE/results/phase0/runs")
OUT_DEFAULT = os.path.join(ROOT, "results", "ablations", "cross_regime_eval.csv")

D_16K, D_50K = 2.7568, 2.9780
# The divisor and the corpus are the same fact stated twice: 16k BPE runs trained on tok16k_full,
# pythia-50k runs on dclm_tokenized. Evaluating a run on the other corpus would produce a number
# with no relation to anything published, so DATA_DIR is set per cell rather than inherited.
CORPUS = {D_16K: "/workspace/FLAME-MoE/data/tok16k_full",
          D_50K: "/workspace/FLAME-MoE/data/dclm_tokenized"}

# Micro-batch, per run, from the geometry each was TRAINED at. run.sh defaults to 32, and at 1e19
# -- 14 layers, hidden 800, a 50k vocab -- the vocab-parallel cross-entropy allocates 12.28 GiB for
# the logits and the evaluation dies of CUDA OOM with 1.36 GiB free. Those runs trained at mb=8.
# Evaluation is not training, but the activation peak scales the same way, so inheriting a default
# rather than the run's own micro-batch is what turns a 33 GB checkpoint into an OOM.
MICRO_BATCH = {"g1_tmoe_coarse_1e19": 8, "moe_coarse_1e19": 8, "temporal_fine_g3_1e19": 8}

# label, run_name, shape, flops, grain, paradigm, R, divisor, note
#
# GRAIN and R are read from each checkpoint's real geometry, not from run.meta. run.meta is rewritten
# by experiments/run.sh on every invocation and at least two of these files no longer describe the
# run they sit in. At grain 3 a layer holds 192 experts with top-18, so R must be 18; passing grain 1
# against a grain-3 checkpoint fails with a router shape mismatch of (192,H) against (64,H), which is
# how this was found.
#
# `R` is the residency the paired regime uses: for a temporal model it is the R it TRAINED under and
# the cross arm turns residency off; for a full-MoE model it is the R imposed at eval only.
#
# The g3 pairs are the point of the table. Every existing imposition cell is a different model
# family, so the claim that the imposition cost does not rise with budget rests on a comparison
# confounded with family. g3_moe at s0/1e16 and s2/1e17 is one family at two budgets in the
# imposition direction, and g3_tmoe at the same two points gives the removal direction alongside it.
CELLS = [
    # --- re-derivations of existing committed cells, where the run is unambiguous ---
    ("moe_coarse_1e19",      "moe_coarse_1e19",       "s19opt", "1e19", 1, "full_moe", 6,  D_50K,
     "re-derive unmask_eval_1e19.csv imposition cell"),
    ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19",   "s19opt", "1e19", 1, "temporal", 6,  D_50K,
     "re-derive unmask_eval_1e19.csv removal cell"),
    ("temporal_fine_1e19",   "temporal_fine_g3_1e19", "s19opt", "1e19", 3, "temporal", 18, D_50K,
     "re-derive unmask_eval_1e19.csv removal cell"),
    ("flame38m_1e18",        "flame38m_g5_temporal",  "s38m",   "1e18", 5, "temporal", 30, D_50K,
     "re-derive the unmask_eval.csv 1e18 cell, which is recorded in val_CE only"),
    # --- the unconfounded budget comparison: one family, two budgets, both directions ---
    ("g3_moe_s0_1e16",       "g3_moe_s0_1e16",        "s0",     "1e16", 3, "full_moe", 18, D_16K,
     "imposition, g3 family at 1e16"),
    ("g3_moe_s2_1e17",       "g3_moe_s2_1e17",        "s2",     "1e17", 3, "full_moe", 18, D_16K,
     "imposition, g3 family at 1e17 -- the cell the confound needs"),
    ("g3_tmoe_s0_1e16",      "g3_tmoe_s0_1e16_mom",   "s0",     "1e16", 3, "temporal", 18, D_16K,
     "removal, g3 family at 1e16"),
    ("g3_tmoe_s2_1e17",      "g3_tmoe_s2_1e17",       "s2",     "1e17", 3, "temporal", 18, D_16K,
     "removal, g3 family at 1e17"),
]

HEADER = ["cell", "run_name", "ckpt_iter", "scale", "trained_paradigm", "direction",
          "native_regime", "native_CE", "native_BPB", "cross_regime", "cross_CE", "cross_BPB",
          "delta_CE", "delta_BPB", "divisor", "note"]


def resolve_iter(run):
    """Highest real iter_* directory. Never trusts latest_checkpointed_iteration.txt."""
    ck = os.path.join(CKPT_ROOT, run, "ckpt")
    iters = sorted(int(m.group(1)) for d in os.listdir(ck)
                   if (m := re.fullmatch(r"iter_0*(\d+)", d)))
    if not iters:
        raise SystemExit(f"[abort] {run}: no iter_* directory under {ck}")
    latest_file = os.path.join(ck, "latest_checkpointed_iteration.txt")
    claimed = None
    if os.path.exists(latest_file):
        claimed = open(latest_file).read().strip()
    if claimed != str(iters[-1]):
        print(f"[warn] {run}: latest_checkpointed_iteration.txt says {claimed!r}, highest real "
              f"checkpoint is {iters[-1]}. Using {iters[-1]}; the file would have loaded a "
              f"{claimed}-step model.", flush=True)
    return iters[-1], latest_file, claimed


def run_cell(spec, keep_going=False):
    label, run, shape, flops, grain, paradigm, R, divisor, note = spec
    it, latest_file, claimed = resolve_iter(run)

    # "Unconstrained" is R=E, NOT R=0. temporal_router.temporal_forward reads
    #     resid_R = int(os.environ.get("TEMPORAL_RESIDENCY_R", "0")) or k
    # so 0 falls through to k, the MAXIMAL constraint. Passing 0 for the free arm makes both arms
    # R=k and the delta comes out at 2e-6, which looks like "changing regime is free" and is really
    # "no regime was changed". The committed CSVs encode E in their own labels -- masked_R18 pairs
    # with unmasked_R192 and masked_R6 with unmasked_R64 -- which is how this was confirmed.
    E = 64 * grain
    if paradigm == "temporal":
        direction, native_tag, cross_tag = "removal", f"masked_R{R}", f"unmasked_R{E}"
        sweep = f"native:{R} cross:{E}"
    else:
        direction, native_tag, cross_tag = "imposition", f"unconstrained_R{E}", f"imposed_R{R}"
        sweep = f"native:{E} cross:{R}"

    # `experiments/run.sh` REWRITES run.meta on every invocation, including a read-only evaluation
    # like this one, with whatever geometry the caller passed. That silently destroys the record of
    # what the run was actually trained as. It has already happened at least once in this repository
    # -- g3_tmoe_s2_1e17/run.meta no longer matches its MANIFEST.csv hash and now reads temporal=0
    # for a run whose name and checkpoint say otherwise -- and it happened again while this script
    # was being written, to g3_moe_s0_1e16. Both files are restored from `latest` and `run.meta`
    # backups here so that a failed evaluation cannot cost provenance.
    # sweep_eval.py writes results/ablations/sweep_eval.csv on every invocation, so running this
    # producer would otherwise leave an unrelated committed file modified and the gate red.
    guarded = [latest_file, os.path.join(CKPT_ROOT, run, "run.meta"),
               os.path.join(ROOT, "results", "ablations", "sweep_eval.csv")]
    backups = {}
    for path in guarded:
        if os.path.exists(path):
            backups[path] = path + ".crossregime_bak"
            shutil.copy2(path, backups[path])
    open(latest_file, "w").write(f"{it}\n")
    try:
        env = dict(os.environ, SWEEPEVAL="1", SWEEP=sweep, TEMPORAL="1",
                   SHAPE=shape, TARGET_FLOPS=flops, GRAIN=str(grain), RUN_NAME=run,
                   DATA_DIR=CORPUS[divisor],
                   MICRO_BATCH=str(MICRO_BATCH.get(run, 32)),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   CKPT_ROOT=CKPT_ROOT)
        p = subprocess.run(["bash", os.path.join(ROOT, "experiments", "run.sh")],
                           env=env, capture_output=True, text=True, cwd=ROOT)
    finally:
        for path, bak in backups.items():
            shutil.move(bak, path)
        if latest_file not in backups and os.path.exists(latest_file):
            os.remove(latest_file)

    got = dict(re.findall(r"\[sweep\] (\w+) lm_loss=([0-9.]+)", p.stdout))
    if "native" not in got or "cross" not in got:
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-15:])
        msg = f"[FAIL] {label}: sweep produced {sorted(got)}, expected native and cross\n{tail}"
        if keep_going:
            print(msg, file=sys.stderr, flush=True)
            return None
        raise SystemExit(msg)

    nce, cce = float(got["native"]), float(got["cross"])
    row = dict(zip(HEADER, [label, run, it, f"{shape}_{flops}", paradigm, direction,
                            native_tag, f"{nce:.4f}", f"{nce / divisor:.4f}",
                            cross_tag, f"{cce:.4f}", f"{cce / divisor:.4f}",
                            f"{cce - nce:+.4f}", f"{(cce - nce) / divisor:+.4f}",
                            f"{divisor:.4f}", note]))
    print(f"[cell] {label:22} iter={it:<6} {direction:10} "
          f"CE {nce:.4f} -> {cce:.4f} ({cce - nce:+.4f})  "
          f"BPB {nce / divisor:.4f} -> {cce / divisor:.4f} ({(cce - nce) / divisor:+.4f})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--keep-going", action="store_true",
                    help="record the cells that succeed instead of aborting on the first failure")
    a = ap.parse_args()

    if a.list:
        print(f"{'label':24}{'run':26}{'scale':10}{'g':3}{'paradigm':10}{'R':>3}  direction")
        for label, run, shape, flops, grain, par, R, _d, _n in CELLS:
            d = "removal" if par == "temporal" else "imposition"
            print(f"{label:24}{run:26}{shape + '_' + flops:10}g{grain} {par:10}{R:>3}  {d}")
        return

    want = [c for c in CELLS if a.all or c[0] in a.cell]
    if not want:
        raise SystemExit("nothing selected; use --all, --cell LABEL, or --list")

    rows = [r for r in (run_cell(c, a.keep_going) for c in want) if r]
    if not rows:
        raise SystemExit("[abort] no cell produced a result; nothing written")

    prior = []
    if os.path.exists(a.out):
        with open(a.out) as f:
            pr = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
        if pr and pr[0] == HEADER:
            done = {r["cell"] for r in rows}
            prior = [r for r in pr[1:] if r[0] not in done]
    with open(a.out, "w", newline="") as f:
        # LF, not csv.writer's default CRLF: csv_sanity counts CRLF differently across its two
        # read paths and reports a phantom shrink, the false positive already on record for
        # layer_freeing_downstream.csv. Matching the rest of results/ablations/ avoids it.
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# Cross-regime evaluation cost. Both arms scored on identical cached batches "
                    f"(analysis/probes/sweep_eval.py). BPB = CE / divisor; divisor is per tokenizer "
                    f"and recorded per row. ckpt_iter is the highest real iter_* directory, not "
                    f"whatever latest_checkpointed_iteration.txt claimed."])
        w.writerow(HEADER)
        w.writerows(prior)
        w.writerows([[r[h] for h in HEADER] for r in rows])
    print(f"[write] {a.out}: {len(rows)} new cell(s), {len(prior)} kept", flush=True)


if __name__ == "__main__":
    main()
