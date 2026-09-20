#!/usr/bin/env bash
# Does attention's downstream advantage survive matched budget, or is it a convergence effect?
#
# At 50M, adding 8.4M parameters of LoRA on q/k/v/o to the free-{0,1,14,15} surface moved BPB by
# 0.001074 -- small -- and mean 10-task accuracy by +0.0123, which is what the same surface needed
# 200M MORE TOKENS to achieve without it. The BPB curve says convergence speed: the gap between the
# two cells fell 0.00717 -> 0.00377 -> 0.00174 across the first 30M and flattened near 0.001. If that
# reading is right, the downstream advantage should shrink or vanish by 250M, where the no-attention
# cell reaches 0.6073. If it does not, attention is buying something tokens cannot.
#
# The comparison is exact: ce_free_0_1_14_15_250M is the same free set, same data seed, same recipe,
# same 250M budget, differing only in the attention adapter.
#
# Resumed from its own 50M checkpoint. LR is constant with no warmup or schedule and --resume-c
# restores optimizer state and the data cursor, so continuing is arithmetically the same run.
#
# Downstream is NOT gated here, unlike the 50M cell. The question is no longer "did attention do
# anything" -- it did -- but how the effect scales, and a null at 250M is exactly as informative as
# a win, so it gets scored either way.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/olmoe-adapt/venv/bin/python}
DATA=/workspace/olmoe-adapt/data
LOGS=/workspace/olmoe-adapt
TAG=ce_free_0_1_14_15_attn_250M
SRC=ce_free_0_1_14_15_attn
FS="0,1,14,15"

_gitsafe() {
  for _ in $(seq 1 40); do [ ! -f "$(git rev-parse --git-dir)/index.lock" ] && break; sleep 15; done
  git add -A results/ analysis/ scripts/ 2>/dev/null
  git commit -q -m "$1" 2>/dev/null
  git push -q origin ple-adaptation 2>/dev/null
  echo "[git] $1 -> $(git rev-parse --short HEAD)"
}

echo "=== $(date +%H:%M) waiting for the GPU"
while pgrep -f "analysis/residency/(train_ple|downstream)\.py" > /dev/null; do sleep 60; done
echo "=== $(date +%H:%M) GPU clear"

if [ -f "$DATA/ple_${TAG}.json" ]; then
  echo "[skip] $TAG (already has a result)"
else
  [ -f "$DATA/csurf_${SRC}_at50M.pt" ] || { echo "[FAIL] no 50M checkpoint for $SRC" >&2; exit 1; }
  echo "=== $(date +%H:%M) $TAG: resume $SRC at 50M, attention LoRA r32, train to 250M"
  "$PY" analysis/ple/train_ple.py --tag "$TAG" --rank off --lora 32 --lora-attn 32 \
        --free-set "$FS" --data-seed 0 --tokens 250000000 --eval-every 10000000 --mb 16 \
        --resume-c "$DATA/csurf_${SRC}_at50M.pt" > "$LOGS/${TAG}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$DATA/ple_${TAG}.json" ]; then
    echo "[FAIL] $TAG rc=$rc; tail:" >&2; tail -25 "$LOGS/${TAG}.log" >&2; exit 1
  fi
  grep -E "^\[resume\]|^\[ple\]|^\[DONE\]" "$LOGS/${TAG}.log"
  "$PY" analysis/residency/consolidate.py > /dev/null || { echo "[FAIL] consolidate aborted" >&2; exit 1; }
  _gitsafe "$TAG: attention LoRA to 250M"
fi

ck="csurf_${TAG}_at250M.pt"
if [ ! -f "$DATA/$ck" ]; then
  echo "[skip] downstream $TAG (no $ck)"
elif grep -q ",$TAG," results/ablations/layer_freeing_downstream.csv 2>/dev/null; then
  echo "[skip] downstream $TAG (already scored)"
else
  echo "=== $(date +%H:%M) downstream $TAG"
  "$PY" analysis/residency/downstream.py --csurf "$ck" --free-set "$FS" --tag "$TAG" \
        > "$LOGS/ds_${TAG}.log" 2>&1
  if [ $? -eq 0 ]; then
    grep -E "^\[ds\] (attention|identity|warn|mean|wrote)" "$LOGS/ds_${TAG}.log"
    _gitsafe "downstream 10-task: $TAG"
  else
    echo "[FAIL] downstream $TAG; tail:" >&2; tail -20 "$LOGS/ds_${TAG}.log" >&2
  fi
fi

# The four-way comparison this run completes: attention x budget, everything else held fixed.
"$PY" - <<'PYEOF'
import csv, json, os
D = "/workspace/olmoe-adapt/data"
BASE, IMPOSE = 0.6727, 2.7507
CELLS = [("ce_free_0_1_14_15", "no ", " 50M"), ("ce_free_0_1_14_15_attn", "r32", " 50M"),
         ("ce_free_0_1_14_15_250M", "no ", "250M"), ("ce_free_0_1_14_15_attn_250M", "r32", "250M")]
ds = {}
p = "results/ablations/layer_freeing_downstream.csv"
if os.path.exists(p):
    lines = [l for l in open(p) if not l.lstrip().lstrip('"').startswith("#")]
    for r in csv.DictReader(lines):
        if r["metric"] == "acc":
            ds.setdefault(r["cell"], []).append(float(r["cell_acc"]))
print(f"\n{'attn':5}{'tokens':7}{'BPB':>10}{'recovery':>10}{'mean acc':>10}")
for tag, a, tok in CELLS:
    f = os.path.join(D, f"ple_{tag}.json")
    if not os.path.exists(f):
        print(f"{a:5}{tok:7}{'(absent)':>10}"); continue
    b = json.load(open(f))["final_bpb"]
    acc = sum(ds[tag]) / len(ds[tag]) if tag in ds else float("nan")
    print(f"{a:5}{tok:7}{b:>10.6f}{(1-(b-BASE)/(IMPOSE-BASE))*100:>9.2f}%{acc:>10.4f}")
PYEOF

echo "=== ATTN 250M COMPLETE $(date +%H:%M) ==="
