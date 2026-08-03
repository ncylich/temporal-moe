#!/usr/bin/env bash
# Recover the last 10M tokens of ce_free_0_1_14_15_attn_250M, then score it downstream.
#
# At 10:44 the trainer was killed mid-checkpoint-write: csurf_..._at250M.pt landed at 469762048 bytes
# where every prior checkpoint of the same cell is 2946346190, and no result JSON was written, so
# consolidate.py could not see the cell at all. The driver scripts (attn_250M.sh, autonomous_queue.sh)
# died in the same instant while watchdog.sh survived. No traceback, no OOM, 240 TB free, 1.78 TB RAM
# available, host up 136 days -- the processes were killed from outside, not by anything in the code.
#
# The 250M eval itself DID run and print: BPB=0.783168. That number is not in question. What is
# missing is the artifact, and the honest way to get it is to recompute rather than to transcribe a
# log line into a JSON and call it a result.
#
# csurf_..._at240M.pt is intact (273 masters, step 3663, hist 24, metadata complete), so this resumes
# from it and trains the final ~10M. Constant LR, no schedule, optimizer state and data cursor both
# restored, so the recomputed cell is the same run -- and it should reproduce 0.783168 closely, which
# is itself a check that the kill corrupted nothing but the file it was writing.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
TAG=ce_free_0_1_14_15_attn_250M
FS="0,1,14,15"

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

while pgrep -f "analysis/ple/(train_ple|downstream)\.py" > /dev/null; do sleep 30; done

if [ -f "$DATA/ple_${TAG}.json" ]; then
  echo "[skip] $TAG (result already present)"
else
  echo "=== $(date -u +%H:%MZ) $TAG: resume at240M, finish the last 10M"
  "$PY" analysis/ple/train_ple.py --tag "$TAG" --rank off --lora 32 --lora-attn 32 \
        --free-set "$FS" --data-seed 0 --tokens 250000000 --eval-every 10000000 --mb 16 \
        --resume-c "$DATA/csurf_${TAG}_at240M.pt" > "$LOGS/${TAG}_recover.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${TAG}.json" ]; then
    echo "[FAIL] recovery rc=$rc; tail:" >&2; tail -25 "$LOGS/${TAG}_recover.log" >&2; exit 1
  fi
  grep -aE "^\[resume\]|^\[DONE\]" "$LOGS/${TAG}_recover.log"
  echo "--- killed run printed BPB=0.783168 at 250M; the line above is the recomputation ---"
  "$PY" analysis/ple/consolidate.py > /dev/null || { echo "[FAIL] consolidate" >&2; exit 1; }
  _gitsafe "$TAG: recovered the final 10M after the trainer was killed mid-checkpoint"
fi

ck="csurf_${TAG}_at250M.pt"
if [ -f "$DATA/$ck" ] && ! grep -q ",$TAG," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
  echo "=== $(date -u +%H:%MZ) downstream $TAG"
  "$PY" analysis/ple/downstream.py --csurf "$ck" --free-set "$FS" --tag "$TAG" \
        > "$LOGS/ds_${TAG}.log" 2>&1
  if [ $? -eq 0 ]; then
    grep -aE "^\[ds\] (attention|identity|warn|mean)" "$LOGS/ds_${TAG}.log"
    _gitsafe "downstream 10-task: $TAG"
  else
    echo "[FAIL] downstream; tail:" >&2; tail -20 "$LOGS/ds_${TAG}.log" >&2
  fi
fi

echo "=== RECOVERY COMPLETE $(date -u +%H:%MZ) ==="
