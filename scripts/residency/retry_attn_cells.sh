#!/usr/bin/env bash
# Re-run any attention cell that died of CUDA OOM, at a micro-batch that fits.
#
# ce_free_0_1_14_15_attn_ds1 died in loss.backward(): "Tried to allocate 12.28 GiB, 12.06 GiB free,
# this process has 67.11 GiB in use". Not a code fault -- the identical configuration
# (ce_free_0_1_14_15_attn) trained to 50M and then to 250M earlier today at the same --mb 16. The
# attention adapter adds 8.4M parameters but, more to the point, activations for four more tensors
# per block, and that puts mb16 within ~0.2 GiB of the ceiling. Whether a given run clears it is
# then down to allocator fragmentation, which is not something to leave to chance in a queue.
#
# --mb 8 --accum 2 holds the effective batch at 16, which train_ple.py's own help says is required
# to match the C recipe, and which is the documented remedy there: "the full-rank rung needs --mb 4
# --accum 4 because activations, not the table, dominate memory". Gradients accumulate across the
# inner loop and the optimizer steps once, so the training math is unchanged -- this is the same
# cell, not a cheaper one. expandable_segments:True is added because the failure message names
# fragmentation specifically.
#
# Order is by value, because the deadline may not fit both: the replicate first (the +0.0123
# attention result at 50M is ~1.6 sigma and everything leans on it), then the {0,1,15} substitution
# cell (whether attention can replace the fourth freed layer and its +43.8 points of memory).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
DEADLINE=$(date -u -d "2026-08-03 18:00:00" +%s)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

_left() { echo $(( (DEADLINE - $(date -u +%s)) / 60 )); }

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

# Whatever the queue finished before it was retired still needs recording, since its own
# consolidate/commit step goes with it.
while pgrep -f "analysis/residency/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done
"$PY" analysis/residency/consolidate.py > /dev/null 2>&1 && _gitsafe "record cells the queue completed before it was retired"

# tag : free-set : data-seed. Ordered by what the remaining time buys:
#   1. score ce_free_0_1_15_attn, which the queue trained but was retired before scoring (35min)
#   2. train + score the replicate at mb8, the cell the whole attention claim rests on (~130min)
CELLS=("ce_free_0_1_15_attn:0,1,15:0" "ce_free_0_1_14_15_attn_ds1:0,1,14,15:1")

_wait_gpu() { while pgrep -f "analysis/residency/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done; }

_score() {  # tag, free-set
  local tag=$1 fs=$2 ck="csurf_$1_at50M.pt"
  [ -f "$DATA/$ck" ] || { echo "[skip] downstream $tag (no checkpoint)"; return 0; }
  grep -q ",$tag," results/ablations/layer_freeing_downstream.csv 2>/dev/null &&     { echo "[skip] downstream $tag (already scored)"; return 0; }
  local l; l=$(_left)
  [ "$l" -lt 35 ] && { echo "=== SKIP downstream $tag: ${l}min left"; return 0; }
  echo "=== $(date -u +%H:%MZ) downstream $tag (${l}min left)"
  "$PY" analysis/residency/downstream.py --csurf "$ck" --free-set "$fs" --tag "$tag" \
        > "$LOGS/ds_${tag}.log" 2>&1 \
    && { grep -aE "^\[ds\] (attention|identity|mean)" "$LOGS/ds_${tag}.log"
         _gitsafe "downstream 10-task: $tag"; } \
    || { echo "[FAIL] downstream $tag" >&2; tail -15 "$LOGS/ds_${tag}.log" >&2; }
}

for spec in "${CELLS[@]}"; do
  IFS=':' read -r tag fs ds <<< "$spec"
  _wait_gpu
  if [ ! -f "$DATA/ple_${tag}.json" ]; then
    l=$(_left)
    if [ "$l" -lt 90 ]; then echo "=== SKIP $tag: needs ~90min, ${l}min left"; continue; fi
    # mb16 first, WITH expandable_segments. The OOM named fragmentation specifically and this exact
    # configuration cleared mb16 twice today, so the cheap attempt is likely to work and is ~55min
    # faster -- the difference between this cell being scored downstream and only having a BPB.
    # mb8 x accum2 is the fallback; effective batch is 16 either way, so both are the same cell.
    for split in "16 1" "8 2"; do
      set -- $split
      echo "=== $(date -u +%H:%MZ) $tag at --mb $1 --accum $2 ($(_left)min left)"
      "$PY" analysis/ple/train_ple.py --tag "$tag" --rank off --lora 32 --lora-attn 32 \
            --free-set "$fs" --data-seed "$ds" --tokens 50000000 --eval-every 10000000 \
            --mb "$1" --accum "$2" > "$LOGS/${tag}.log" 2>&1
      [ -f "$DATA/ple_${tag}.json" ] && break
      echo "[warn] $tag did not complete at mb$1"
    done
    if [ ! -f "$DATA/ple_${tag}.json" ]; then
      echo "[FAIL] $tag; tail:" >&2
      tail -c 1200 "$LOGS/${tag}.log" | tr '\r' '\n' | grep -avE "^Loading|^$" | tail -8 >&2
      continue
    fi
    grep -aE "^\[ple\]|^\[DONE\]" "$LOGS/${tag}.log"
    "$PY" analysis/residency/consolidate.py > /dev/null || { echo "[FAIL] consolidate" >&2; continue; }
    _gitsafe "$tag: attention LoRA at mb8 x accum2 after mb16 hit CUDA OOM"
  fi
  _score "$tag" "$fs"
done

echo "=== RETRY COMPLETE $(date -u +%H:%MZ), $(_left)min to deadline ==="
