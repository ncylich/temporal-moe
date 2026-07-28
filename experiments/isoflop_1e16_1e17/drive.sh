#!/bin/bash
# Sequential run driver for Phase-0 sweeps (single GPU). Reads config lines from a file:
#   NAME SHAPE FLOPS PEAK_LR WARMUP_FRAC GLOBAL_BATCH SEED [AUX_COEFF]
# Skips runs already complete (final-iter checkpoint present). Parses each + appends to log.md.
# Usage: drive.sh <configs_file>
set -uo pipefail
# One environment contract: ROOT, PY, DATA_DIR, TOKENIZER_MODEL, CKPT_ROOT, NV.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
CFG=${1:?need configs file}
export HF_TOKEN=${HF_TOKEN:-}
PORT=29510

is_done() {  # $1=run_dir $2=total_iters -> 0 if final checkpoint present
  local d=$1 it=$2
  local tag=$(printf "iter_%07d" "$it")
  [ -d "$d/ckpt/$tag" ]
}

while read -r NAME SHAPE FLOPS LR WU GB SEED AUX; do
  [ -z "${NAME:-}" ] && continue
  case "$NAME" in \#*) continue;; esac
  AUX=${AUX:-0.01}
  # total iters for skip-check
  read _N ITERS < <("$PY" analysis/shapes.py iters "$SHAPE" "$FLOPS" "$GB")
  RUNDIR=$ROOT/results/phase0/runs/$NAME
  if is_done "$RUNDIR" "$ITERS"; then
    echo "[drive] SKIP $NAME (complete, iters=$ITERS)"
  else
    echo "[drive] RUN $NAME  shape=$SHAPE flops=$FLOPS lr=$LR wu=$WU gb=$GB seed=$SEED aux=$AUX iters=$ITERS  $(date)"
    PORT=$((PORT+1))
    SHAPE=$SHAPE TARGET_FLOPS=$FLOPS PEAK_LR=$LR WARMUP_FRAC=$WU GLOBAL_BATCH=$GB SEED=$SEED \
      AUX_COEFF=$AUX RUN_NAME=$NAME RDZV_PORT=$PORT bash experiments/run.sh
  fi
  # parse + log
  SUMMARY=$("$PY" analysis/parse_run.py "$RUNDIR" 2>/dev/null | grep '^SUMMARY')
  echo "[drive] $SUMMARY"
  {
    echo ""
    echo "### $NAME  ($(date '+%Y-%m-%d %H:%M'))"
    echo "Config: shape=$SHAPE flops=$FLOPS peak_lr=$LR warmup=$WU gb=$GB seed=$SEED aux=$AUX iters=$ITERS"
    echo "$SUMMARY"
    "$PY" analysis/parse_run.py "$RUNDIR" 2>/dev/null | grep '^{'
  } >> "$ROOT/results/phase0/log.md"
done < "$CFG"
echo "[drive] ALL CONFIGS DONE $(date)"
