#!/usr/bin/env bash
# Extend {0,1,14,15} to 200M and score it downstream.
#
# This is the run the overnight chain should have made. Its winner rule called the 0.0115 BPB gap
# between {0,1,14,15} (0.786275) and {0,1,15} (0.797810) a tie, because the gap is under the
# program's pre-registered 2 sigma = 0.012 bar, and handed the extension to {0,1,15} on the memory
# tie-break. The replicate run in the same chain then measured what that bar is actually worth:
# ce_free_0_1_15 and ce_free_0_1_15_ds1 differ by 0.000004 BPB at 50M on different corpus
# permutations. The bar is about 3000x the spread it is meant to cover, so the gap was never a tie.
# ple_RESULTS.md §6 had already said as much -- sigma was measured by scoring the base model on
# DISJOINT data subsamples, while every arm is scored on the same fixed 256-pack subset with a
# bitwise-deterministic eval, so subsample variance cannot contribute to an inter-arm difference.
#
# Waits for the GPU to clear rather than assuming it is free. Two concurrent cells at mb16 do not
# fit in 80 GB, and this program has lost runs to exactly that.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
TAG=ce_free_0_1_14_15
FS="0,1,14,15"
EXT="${TAG}_200M"

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

echo "=== $(date +%H:%M) waiting for the GPU to clear"
while pgrep -f "analysis/ple/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done
echo "=== $(date +%H:%M) GPU clear"

# ---- 50M -> 200M --------------------------------------------------------------------------------
if [ -f "$DATA/ple_${EXT}.json" ]; then
  echo "[skip] $EXT (already has a result)"
else
  [ -f "$DATA/csurf_${TAG}_at50M.pt" ] || { echo "[FAIL] no 50M checkpoint for $TAG" >&2; exit 1; }
  echo "=== $(date +%H:%M) $EXT: resume $TAG at 50M, train to 200M"
  "$PY" analysis/ple/train_ple.py --tag "$EXT" --rank off --lora 32 --free-set "$FS" \
        --data-seed 0 --tokens 200000000 --eval-every 10000000 --mb 16 \
        --resume-c "$DATA/csurf_${TAG}_at50M.pt" > "$LOGS/${EXT}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${EXT}.json" ]; then
    echo "[FAIL] $EXT rc=$rc; tail:" >&2; tail -20 "$LOGS/${EXT}.log" >&2; exit 1
  fi
  grep -E "^\[resume\]|^\[DONE\]" "$LOGS/${EXT}.log"
  "$PY" analysis/ple/consolidate.py > /dev/null || { echo "[FAIL] consolidate aborted" >&2; exit 1; }
  _gitsafe "$EXT: {0,1,14,15} extended 50M -> 200M"
fi

# ---- downstream, 200M then the 50M control ------------------------------------------------------
for spec in "$EXT:csurf_${EXT}_at200M.pt" "$TAG:csurf_${TAG}_at50M.pt"; do
  tg=${spec%%:*}; ck=${spec##*:}
  [ -f "$DATA/$ck" ] || { echo "[skip] downstream $tg (no $ck)"; continue; }
  if grep -q ",$tg," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
    echo "[skip] downstream $tg (already scored)"; continue
  fi
  echo "=== $(date +%H:%M) downstream $tg"
  "$PY" analysis/ple/downstream.py --csurf "$ck" --free-set "$FS" --tag "$tg" \
        > "$LOGS/ds_${tg}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] downstream $tg rc=$rc"; tail -20 "$LOGS/ds_${tg}.log" >&2; continue; }
  grep -E "^\[ds\] (identity|warn|mean|wrote)" "$LOGS/ds_${tg}.log"
  _gitsafe "downstream 10-task: $tg"
done

echo "=== EXTENSION COMPLETE $(date +%H:%M) ==="
