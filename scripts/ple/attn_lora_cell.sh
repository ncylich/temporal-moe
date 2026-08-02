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

# The comparison this cell exists for, printed rather than left to be reconstructed later.
"$PY" - <<'PYEOF'
import json, os
D = "/workspace/olmoe-adapt/data"
BASE, IMPOSE = 0.6727, 2.7507
for t in ("ce_free_0_1_14_15", "ce_free_0_1_14_15_attn"):
    p = os.path.join(D, f"ple_{t}.json")
    if not os.path.exists(p):
        print(f"  {t:30s} (absent)"); continue
    r = json.load(open(p))
    b = r["final_bpb"]
    print(f"  {t:30s} BPB={b:.6f}  recovery={(1-(b-BASE)/(IMPOSE-BASE))*100:.2f}%  "
          f"attn_lora_r={r.get('lora_attn', 0)}")
a = os.path.join(D, "ple_ce_free_0_1_14_15.json"); b = os.path.join(D, "ple_ce_free_0_1_14_15_attn.json")
if os.path.exists(a) and os.path.exists(b):
    d = json.load(open(b))["final_bpb"] - json.load(open(a))["final_bpb"]
    print(f"  attention LoRA is worth {-d:+.6f} BPB (positive = better). Replicate spread on this "
          f"program is 0.000004; the published 2-sigma bar of 0.012 is ~3000x that and is the wrong "
          f"comparison here (ple_RESULTS.md §6).")
PYEOF

echo "=== ATTN CELL COMPLETE $(date +%H:%M) ==="
