#!/usr/bin/env bash
# Does adapting ATTENTION recover any of the residency constraint price?
#
# Every arm this program has ever run adapts the router, the RMSNorm gains, and the expert MLPs.
# Attention is frozen in all of them -- including F', the 6.92B-parameter "full finetune" whose
# 0.8106 is quoted as the constraint price and as evidence that no adaptation of existing weights
# can do better. F' unfroze every parameter, so attention WAS trainable there; but every efficient
# arm, and every cell in the layer-freeing line, has left it out. So the question of whether a cheap
# attention adapter buys anything on top of the CE surface has not been asked.
#
# The mechanism is at least plausible rather than arbitrary. Rolling residency restricts WHICH
# experts a token may reach. Attention determines what the token's representation contains by the
# time it gets there. If routing quality is bounded by the residency schedule, shaping the query
# that the schedule sees is a different lever from adapting the experts it lands on.
#
#   cell: ce_free_0_1_14_15_attn -- identical to ce_free_0_1_14_15 (0.786275) in every respect
#         except +8.4M parameters of LoRA r32 on q/k/v/o. Same free set, same budget, same data
#         seed, same recipe. So the difference IS the attention adapter.
#
# Runs after whatever else holds the GPU. Cheap: ~8.4M parameters against the 235M of expert LoRA.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
TAG=ce_free_0_1_14_15_attn
FS="0,1,14,15"

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

echo "=== $(date +%H:%M) waiting for the 250M chain and the GPU"
while pgrep -f "extend_250M\.sh" > /dev/null; do sleep 120; done
while pgrep -f "analysis/ple/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done
echo "=== $(date +%H:%M) GPU clear"

if [ -f "$DATA/ple_${TAG}.json" ]; then
  echo "[skip] $TAG (already has a result)"
else
  echo "=== $(date +%H:%M) $TAG: CE surface + attention LoRA r32, free {$FS}, 50M"
  "$PY" analysis/ple/train_ple.py --tag "$TAG" --rank off --lora 32 --lora-attn 32 \
        --free-set "$FS" --data-seed 0 --tokens 50000000 --eval-every 10000000 --mb 16 \
        > "$LOGS/${TAG}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${TAG}.json" ]; then
    echo "[FAIL] $TAG rc=$rc; tail:" >&2; tail -25 "$LOGS/${TAG}.log" >&2; exit 1
  fi
  grep -E "^\[ple\]|^\[eval\]|^\[DONE\]" "$LOGS/${TAG}.log"
  "$PY" analysis/ple/consolidate.py > /dev/null || { echo "[FAIL] consolidate aborted" >&2; exit 1; }
  _gitsafe "$TAG: attention LoRA on the free-{$FS} surface at 50M"
fi

# The comparison this cell exists for, printed rather than left to be reconstructed later, and
# used to decide whether the downstream hour is worth spending.
#
# GATE: score downstream only if attention actually helped. A null attention result is a complete
# answer on its own -- it says the constraint price is not attention-shaped -- and does not need
# ten tasks to characterise it.
#
# The threshold is 0.0005 BPB, about 2.5x the noise scale. Two estimates of that scale are
# available and they disagree by two orders of magnitude, so the more conservative one is used.
# The replicate pair (ce_free_0_1_15 vs ce_free_0_1_15_ds1, different corpus permutations) differs
# by 0.000004 at 50M -- but that is one pair at one point, and consecutive evals of
# ce_free_0_1_14_15_250M in its flattened region (120-140M, where training has stopped buying
# anything) differ by ~0.0002, once moving the wrong way. Treat ~0.0002 as the real precision on a
# cell's BPB and the 4e-6 agreement as fortunate.
#
# Both are far below the published 0.012 bar, which §6 established is the wrong instrument here
# because it was estimated from disjoint data subsamples while every arm is scored on the same
# fixed subset. 2.5x is deliberately permissive: a false positive costs half an hour of GPU, a
# false negative discards a real result.
GATE=0.0005
verdict=$("$PY" - "$GATE" <<'PYEOF'
import json, os, sys
D = "/workspace/olmoe-adapt/data"
BASE, IMPOSE, gate = 0.6727, 2.7507, float(sys.argv[1])
def load(t):
    p = os.path.join(D, f"ple_{t}.json")
    return json.load(open(p)) if os.path.exists(p) else None
ctrl, attn = load("ce_free_0_1_14_15"), load("ce_free_0_1_14_15_attn")
for t, r in (("ce_free_0_1_14_15", ctrl), ("ce_free_0_1_14_15_attn", attn)):
    if r is None:
        print(f"  {t:30s} (absent)", file=sys.stderr); continue
    b = r["final_bpb"]
    print(f"  {t:30s} BPB={b:.6f}  recovery={(1-(b-BASE)/(IMPOSE-BASE))*100:.2f}%  "
          f"attn_lora_r={r.get('lora_attn', 0)}", file=sys.stderr)
if ctrl is None or attn is None:
    print("SKIP"); raise SystemExit
gain = ctrl["final_bpb"] - attn["final_bpb"]          # positive = attention helped
print(f"  attention LoRA is worth {gain:+.6f} BPB (positive = better); gate is {gate}",
      file=sys.stderr)
print("SCORE" if gain > gate else "SKIP")
PYEOF
)

if [ "$verdict" = "SCORE" ]; then
  ck="csurf_${TAG}_at50M.pt"
  if [ ! -f "$DATA/$ck" ]; then
    echo "[FAIL] gate passed but no $ck to score" >&2
  elif grep -q ",$TAG," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
    echo "[skip] downstream $TAG (already scored)"
  else
    echo "=== $(date +%H:%M) downstream $TAG (attention helped; scoring)"
    "$PY" analysis/ple/downstream.py --csurf "$ck" --free-set "$FS" --tag "$TAG" \
          > "$LOGS/ds_${TAG}.log" 2>&1
    if [ $? -eq 0 ]; then
      grep -E "^\[ds\] (attention|identity|warn|mean|wrote)" "$LOGS/ds_${TAG}.log"
      _gitsafe "downstream 10-task: $TAG"
    else
      echo "[FAIL] downstream $TAG; tail:" >&2; tail -20 "$LOGS/ds_${TAG}.log" >&2
    fi
  fi
else
  echo "=== downstream SKIPPED: attention did not clear the $GATE BPB gate. A null here is the"
  echo "=== answer -- the constraint price is not attention-shaped -- and needs no task battery."
fi

echo "=== ATTN CELL COMPLETE $(date +%H:%M) ==="
