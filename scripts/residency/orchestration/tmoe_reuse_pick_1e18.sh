#!/usr/bin/env bash
# After the 1e17 reuse-fraction sweep (cur_g1_1e17_WK{2,3,4,5}_16k): pick the arm with the lowest
# final test CE under its own policy (ties within 0.005 -> the higher reuse fraction, i.e. fewer
# swaps), then run it at 1e18 in the paper's grain-1 flame38m config. Log: curriculum_reuse_1e18.out
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
until grep -q "sweep2 ALL DONE" /workspace/rerun-logs/curriculum_sweep2.out 2>/dev/null; do sleep 120; done
best=""; bestce=""
for s in 2 3 4 5; do
  f=results/phase0/runs/cur_g1_1e17_WK${s}_16k/train.log
  ce=$(grep "on test set" $f 2>/dev/null | tail -1 | grep -o "lm loss value: [0-9.E+]*" | awk '{print $4}')
  [ -z "$ce" ] && continue
  echo "WK$s (reuse $((6-s))/6): $ce"
  if [ -z "$best" ] || "$PY" -c "import sys; sys.exit(0 if float('$ce') < float('$bestce') - 0.005 else 1)"; then best=WK$s; bestce=$ce; fi
done
echo "### pick $best ($bestce) $(date -u +%H:%M)"
[ -z "$best" ] && { echo "no sweep results"; exit 1; }
ARM=$best GRAIN=1 bash scripts/residency/orchestration/tmoe_curriculum_1e18.sh
echo "### reuse 1e18 ALL DONE $(date -u +%H:%M)"
