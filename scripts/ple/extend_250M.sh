#!/usr/bin/env bash
# Take the free-set cells to 250M and score them downstream.
#
# 250M rather than 200M because that is where the comparators live. Arm C's published curve runs to
# 250M, the bake-off arms trained 250M, and CE_adapt_R8 -- the full-residency column every downstream
# table is read against -- is the 250M model. A 200M cell beating it is a real result reported at a
# handicap, and the handicap is invisible in the table unless someone reads the note.
#
# Two cells, in priority order:
#
#   1. {0,1,14,15}  50M -> 250M   the best 50M cell (0.786275). This is the run the overnight chain
#                                 should have made; its winner rule called a 0.0115 gap a tie under
#                                 the published 0.012 bar, which the replicate in that same chain then
#                                 measured at 0.000004.
#   2. {0,1,15}    200M -> 250M   already at 200M, so 50M more finishes it. Makes the two free-set
#                                 cells matched at 250M against each other AND against CE-250M.
#
# Both resume from their own checkpoints. LR is constant with no warmup or schedule and --resume-c
# restores optimizer state and the data cursor, so continuing is arithmetically the same run.
#
# Waits for the GPU rather than assuming it is free: two cells at mb16 do not fit in 80 GB.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
TARGET=250000000

# tag_out : free_set : resume_checkpoint
CELLS=(
  "ce_free_0_1_14_15_250M:0,1,14,15:csurf_ce_free_0_1_14_15_at50M.pt"
  "ce_free_0_1_15_250M:0,1,15:csurf_ce_free_0_1_15_200M_at200M.pt"
)

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

_wait_gpu() {
  while pgrep -f "analysis/ple/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done
}

echo "=== $(date +%H:%M) waiting for the GPU to clear"
_wait_gpu
echo "=== $(date +%H:%M) GPU clear"

for spec in "${CELLS[@]}"; do
  IFS=':' read -r tag fs ck <<< "$spec"

  # ---- train to 250M ----
  if [ -f "$DATA/ple_${tag}.json" ]; then
    echo "[skip] $tag (already has a result)"
  elif [ ! -f "$DATA/$ck" ]; then
    echo "[FAIL] $tag: no checkpoint $ck to resume from" >&2; continue
  else
    echo "=== $(date +%H:%M) $tag: resume $ck, train to $((TARGET / 1000000))M"
    "$PY" analysis/ple/train_ple.py --tag "$tag" --rank off --lora 32 --free-set "$fs" \
          --data-seed 0 --tokens "$TARGET" --eval-every 10000000 --mb 16 \
          --resume-c "$DATA/$ck" > "$LOGS/${tag}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${tag}.json" ]; then
      echo "[FAIL] $tag rc=$rc; tail:" >&2; tail -20 "$LOGS/${tag}.log" >&2; continue
    fi
    grep -E "^\[resume\]|^\[DONE\]" "$LOGS/${tag}.log"
    "$PY" analysis/ple/consolidate.py > /dev/null || { echo "[FAIL] consolidate aborted" >&2; continue; }
    _gitsafe "$tag: {$fs} to 250M"
  fi

  # ---- downstream on it ----
  dck="csurf_${tag}_at250M.pt"
  if [ ! -f "$DATA/$dck" ]; then
    echo "[skip] downstream $tag (no $dck)"
  elif grep -q ",$tag," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
    echo "[skip] downstream $tag (already scored)"
  else
    echo "=== $(date +%H:%M) downstream $tag"
    "$PY" analysis/ple/downstream.py --csurf "$dck" --free-set "$fs" --tag "$tag" \
          > "$LOGS/ds_${tag}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[FAIL] downstream $tag rc=$rc; tail:" >&2; tail -20 "$LOGS/ds_${tag}.log" >&2
    else
      grep -E "^\[ds\] (identity|warn|mean|wrote)" "$LOGS/ds_${tag}.log"
      _gitsafe "downstream 10-task: $tag"
    fi
  fi
done

# ---- the 50M control for the four-layer cell, last: it separates the token effect from the
# ---- freeing effect, and is the cheapest thing to lose if the night runs out.
if [ -f "$DATA/csurf_ce_free_0_1_14_15_at50M.pt" ] && \
   ! grep -q ",ce_free_0_1_14_15," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
  echo "=== $(date +%H:%M) downstream ce_free_0_1_14_15 (50M control)"
  "$PY" analysis/ple/downstream.py --csurf csurf_ce_free_0_1_14_15_at50M.pt \
        --free-set 0,1,14,15 --tag ce_free_0_1_14_15 > "$LOGS/ds_ce_free_0_1_14_15.log" 2>&1 \
    && { grep -E "^\[ds\] (identity|warn|mean|wrote)" "$LOGS/ds_ce_free_0_1_14_15.log"
         _gitsafe "downstream 10-task: ce_free_0_1_14_15 (50M control)"; } \
    || { echo "[FAIL] downstream 50M control" >&2; tail -20 "$LOGS/ds_ce_free_0_1_14_15.log" >&2; }
fi

echo "=== 250M EXTENSION COMPLETE $(date +%H:%M) ==="
