#!/usr/bin/env bash
# T1 -- per-layer constraint sweep with co-adaptation, at s0/1e16 (grain 3: E=192, k=18, MoE layers 2-4).
#
# C3 perturbs a trained checkpoint, so it measures the cost of REMOVING freedom from a model that
# trained expecting it. T1 trains under the constraint and measures the cost of never having had it.
# That is the one comparison C3 structurally cannot make, and the two disagree: the endpoint spike
# that dominates every inference-time measurement does not appear here.
#
#   ARMS=A0,A5 SEEDS=1234,2 scripts/phase0/t1_sweep.sh     # subset
#   scripts/phase0/t1_sweep.sh                             # all eight arms, all three seeds
#
# Appends to results/ablations/t1_perlayer_training.csv, skipping any (arm, seed) already present, so
# it resumes after an interruption. Commits and pushes per arm -- a run that finishes and is lost to a
# crash is worse than one never started.
#
# The schedule field is written through _csv_field(), which quotes anything containing the delimiter.
# It previously emitted bare `2:18,3:18` into a comma-separated file, giving rows with 7 and 8 fields
# against a 6-field header; DictReader then read A5's test_CE as the string "3:18". Repairing the rows
# held only until this script ran again -- the fix has to be in the writer.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/FLAME-MoE/.venv/bin/python}
export PY DATA_DIR=${DATA_DIR:-/workspace/FLAME-MoE/data/tok16k_full}
OUT=results/ablations/t1_perlayer_training.csv
E=192; K=18

declare -A SCHED=(
  [A0]=""                       # unconstrained reference (R=E everywhere)
  [A1]="2:$K,3:$K,4:$K"         # fully constrained
  [A2]="2:$K"                   # first MoE layer only
  [A3]="3:$K"                   # interior only
  [A4]="4:$K"                   # last MoE layer only
  [A5]="2:$K,3:$K"              # exempt the last
  [A6]="3:$K,4:$K"              # exempt the first
  [A7]="2:76,3:76,4:76"         # uniform R matched to A5/A6 in resident slots (228)
)
IFS=',' read -ra ARMS  <<< "${ARMS:-A0,A1,A5,A6,A7,A2,A3,A4}"
IFS=',' read -ra SEEDS <<< "${SEEDS:-1234,2,3}"

_csv_field() {  # quote a value that contains the delimiter, a quote, or a newline
  local v=$1
  if [[ "$v" == *,* || "$v" == *'"'* || "$v" == *$'\n'* ]]; then
    printf '"%s"' "${v//\"/\"\"}"
  else
    printf '%s' "$v"
  fi
}

_gitsafe() {  # two queues may share this repo; wait for the index rather than racing it
  for _ in $(seq 1 40); do [ ! -f .git/index.lock ] && break; sleep 15; done
  git add -A "$OUT" >/dev/null 2>&1
  git commit -q -m "$1" >/dev/null 2>&1
  git push -q origin "$(git rev-parse --abbrev-ref HEAD)" >/dev/null 2>&1
}

[ -s "$OUT" ] || echo "run_name,arm,seed,schedule,test_CE,wall_s" > "$OUT"

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    name="t1_${arm}_s0_1e16_seed${seed}"
    if grep -q "^$name," "$OUT" 2>/dev/null; then echo "[skip] $name"; continue; fi
    st=$SECONDS
    TEMPORAL=1 TEMPORAL_RESIDENCY_R=$E TEMPORAL_R_SCHEDULE="${SCHED[$arm]}" \
      SHAPE=s0 TARGET_FLOPS=1e16 GRAIN=3 SEED="$seed" RUN_NAME="$name" \
      bash experiments/run.sh > "/tmp/${name}.log" 2>&1
    rc=$?; el=$((SECONDS-st))
    ce=$(grep -oE "on test set \| lm loss value: [0-9.E+]+" "/tmp/${name}.log" | tail -1 \
         | grep -oE "[0-9]+\.[0-9E+]+$")
    [ $rc -eq 0 ] && [ -n "$ce" ] || { ce="FAILED_rc$rc"; echo "[fail] $name rc=$rc ${el}s"; }
    printf '%s,%s,%s,%s,%s,%s\n' "$name" "$arm" "$seed" \
           "$(_csv_field "${SCHED[$arm]:-none}")" "$ce" "$el" >> "$OUT"
    [ "${ce:0:6}" != "FAILED" ] && echo "[ok]   $name CE=$ce ${el}s"
    _gitsafe "T1: $arm seed $seed"
  done
done
echo "=== T1 SWEEP COMPLETE $(date +%H:%M) ==="
