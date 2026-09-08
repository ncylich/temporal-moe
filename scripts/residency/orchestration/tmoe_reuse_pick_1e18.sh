#!/usr/bin/env bash
# After the 1e17 reuse-fraction sweep (cur_g1_1e17_WK{2,3,4,5}_16k): pick the arm with the lowest
# final test CE under its own policy (no tie-break: serving cost is flat below reuse 5/6), run it at
# 1e18 in the paper's grain-1 flame38m config, and if it beats the recorded temporal triplet there
# (mean 3.9077) by 0.010, run it at 1e19 in the paper's coarse config. Log: curriculum_reuse_1e18.out
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
until grep -q "sweep2 ALL DONE" /workspace/rerun-logs/curriculum_sweep2.out 2>/dev/null; do sleep 120; done
best=""; bestce=""
for s in 2 3 4 5; do
  f=results/phase0/runs/cur_g1_1e17_WK${s}_16k/train.log
  ce=$(grep "on test set" $f 2>/dev/null | tail -1 | grep -o "lm loss value: [0-9.E+]*" | awk '{print $4}')
  [ -z "$ce" ] && continue
  echo "WK$s (reuse $((6-s))/6): $ce"
  if [ -z "$best" ] || "$PY" -c "import sys; sys.exit(0 if float('$ce') < float('$bestce') else 1)"; then best=WK$s; bestce=$ce; fi
done
echo "### pick $best ($bestce) $(date -u +%H:%M)"
[ -z "$best" ] && { echo "no sweep results"; exit 1; }
ARM=$best GRAIN=1 bash scripts/residency/orchestration/tmoe_curriculum_1e18.sh
CE18=$(grep "on test set" results/phase0/runs/cur_flame38m_g1_$best/train.log 2>/dev/null | tail -1 | grep -o "lm loss value: [0-9.E+]*" | awk '{print $4}')
echo "### reuse 1e18 result $best test_CE=$CE18 (temporal triplet mean 3.9077, free MoE mean 3.9235, bar 3.8977) $(date -u +%H:%M)"
if [ -n "$CE18" ] && "$PY" -c "import sys; sys.exit(0 if float('$CE18') <= 3.9077 - 0.010 else 1)"; then
  echo "### reuse 1e19 START $best $(date -u +%H:%M)"
  ARM=$best GRAIN=1 bash scripts/residency/orchestration/tmoe_curriculum_1e19.sh
else
  echo "### reuse 1e19 skipped: 1e18 did not clear the bar $(date -u +%H:%M)"
fi
echo "### reuse ALL DONE $(date -u +%H:%M)"
