#!/bin/bash
# Capture sweep for the locus/lens/structural family (re-run plan Step 3).
#
# For each named run, reconstruct its training configuration from its own run.meta and re-invoke
# experiments/run.sh with DELEXPROBE=1, producing <run>/delex_capture.pt. One forward pass per run,
# single GPU. This is the driver the three preserved captures never had: they were produced ad hoc,
# so the family had no reproducible entry point and no record of what shape each capture was taken at.
#
#   scripts/phase0/delex_capture_sweep.sh                     # the default selection set
#   scripts/phase0/delex_capture_sweep.sh --list               # print the set and exit
#   scripts/phase0/delex_capture_sweep.sh run_a run_b          # named runs
#   FORCE=1 scripts/phase0/delex_capture_sweep.sh run_a        # recapture even if one exists
#
# Selection rule (Step 3 item 10): one run per (budget, regime, granularity) cell, preferring the
# plainest recipe name, plus the dense control at each budget as a floor. 1e18 comes first: no
# mechanistic measurement of any kind exists there, and it is the budget at which the temporal model
# wins. `analysis/probes/registry.py --selection` computes the set; nothing is hardcoded here.
#
# Two hazards this handles, both of which would corrupt the record silently:
#
# 1. run.sh rewrites $OUT/run.meta on every invocation, before the mode branch. run.meta is the
#    provenance of a published run and the source registry.py reads, so an env var reconstructed
#    wrongly here would overwrite it with the wrong architecture. Each run.meta is snapshotted and
#    restored around the call, and a diff is reported if it changed.
# 2. A capture whose micro-batch differs from its published sibling is not comparable to it. run.sh
#    derives N_MB from MICRO_BATCH so every capture is the same 64 sequences x 2048 tokens; this
#    driver therefore passes each run's own recorded mb rather than run.sh's default.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$ROOT"

RUNS_DIR=${CKPT_ROOT:-$ROOT/results/phase0/runs}

if [ "${1:-}" = "--list" ]; then
  "$PY" analysis/probes/registry.py --selection
  exit 0
fi

if [ $# -gt 0 ]; then
  SELECTION=("$@")
else
  mapfile -t SELECTION < <("$PY" analysis/probes/registry.py --selection --names-only)
fi

echo "=== delex capture sweep: ${#SELECTION[@]} runs ==="
printf '  %s\n' "${SELECTION[@]}"
echo

ok=0; skipped=0; failed=0
for run in "${SELECTION[@]}"; do
  meta="$RUNS_DIR/$run/run.meta"
  if [ ! -f "$meta" ]; then
    echo "[skip] $run: no run.meta (scripts/artifacts.py pull --run $run)"; skipped=$((skipped+1)); continue
  fi
  if [ ! -d "$RUNS_DIR/$run/ckpt" ]; then
    echo "[skip] $run: no checkpoint on disk (scripts/artifacts.py pull --run $run)"
    skipped=$((skipped+1)); continue
  fi
  if [ -f "$RUNS_DIR/$run/delex_capture.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $run: capture already present (FORCE=1 to recapture)"; skipped=$((skipped+1)); continue
  fi

  # Reconstruct the configuration from run.meta. `shape=` is present only in the two-line form; the
  # flame38m/flame192/flame512 runs record ffn/moe_ffn/num_experts but no shape, so those must be
  # given a SHAPE explicitly via the environment.
  get() { sed -n "s/.*[[:space:]]$1=\([^[:space:]]*\).*/\1/p" "$meta" | head -1; }
  shape=$(get shape); grain=$(get grain); topk=$(get topk); mb=$(get mb); gb=$(get gb)
  flops=$(get flops); temporal=$(get temporal); dense=$(get dense); mode=$(get mode)
  smult=$(get shared_mult); lr=$(get lr)
  [ -z "$temporal" ] && { [ "$mode" = "temporal" ] && temporal=1 || temporal=0; }
  [ -z "$dense" ] && { [ "$mode" = "dense" ] && dense=1 || dense=0; }
  shape=${SHAPE:-$shape}
  flops=${TARGET_FLOPS:-$flops}
  if [ -z "$shape" ] || [ -z "$flops" ]; then
    echo "[skip] $run: run.meta records no shape=/flops= (the 1e18 single-line form)."
    echo "        Re-run with an explicit shape, e.g.  SHAPE=s4 TARGET_FLOPS=1e18 $0 $run"
    skipped=$((skipped+1)); continue
  fi

  cp "$meta" "$meta.presweep"
  echo "--- $run: shape=$shape grain=$grain topk=$topk mb=$mb gb=$gb temporal=$temporal dense=$dense"
  set +e
  env DELEXPROBE=1 RUN_NAME="$run" SHAPE="$shape" TARGET_FLOPS="$flops" \
      GRAIN="${grain:-1}" TOPK="$topk" MICRO_BATCH="$mb" GLOBAL_BATCH="$gb" \
      SHARED_MULT="${smult:-2}" PEAK_LR="${lr:-3e-3}" \
      TEMPORAL="$temporal" DENSE="$dense" \
      TEMPORAL_EVICT="${TEMPORAL_EVICT:-min_logit}" \
      ./experiments/run.sh
  rc=$?
  set -e
  # Restore provenance, and say so if run.sh wrote something different from what was there.
  if ! cmp -s "$meta" "$meta.presweep"; then
    echo "[warn] $run: run.sh rewrote run.meta; restoring the original. Difference was:"
    diff "$meta.presweep" "$meta" | sed 's/^/        /' || true
  fi
  mv -f "$meta.presweep" "$meta"

  if [ $rc -ne 0 ]; then
    echo "[fail] $run: run.sh exited $rc (see $RUNS_DIR/$run/delexprobe.log)"; failed=$((failed+1)); continue
  fi
  if [ ! -f "$RUNS_DIR/$run/delex_capture.pt" ]; then
    echo "[fail] $run: run.sh succeeded but no capture was written"; failed=$((failed+1)); continue
  fi
  echo "[ok]   $run: $(du -h "$RUNS_DIR/$run/delex_capture.pt" | cut -f1) capture"
  ok=$((ok+1))
done

echo
echo "=== captured $ok, skipped $skipped, failed $failed ==="
echo "next: \$PY analysis/probes/delex_locus_driver.py   # null-control gate, then the A-family"
[ $failed -eq 0 ]
