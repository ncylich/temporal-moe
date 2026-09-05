#!/usr/bin/env bash
# After the running 1e18 arm (driver pid $1) finishes: run reuse 2/6 (WK4) at 1e18 too, take the lower
# final test CE of WK5 and WK4, and launch it at 1e19 if it is 0.010 below the recorded temporal
# triplet mean (3.9077). Log: curriculum_reuse_1e18b.out
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "### 1e18 WK5 finished $(date -u +%H:%M)"
ARM=WK4 GRAIN=1 bash scripts/residency/orchestration/tmoe_curriculum_1e18.sh
best=""; bestce=""
for a in WK5 WK4; do
  ce=$(grep "on test set" results/phase0/runs/cur_flame38m_g1_$a/train.log 2>/dev/null | tail -1 | grep -o "lm loss value: [0-9.E+]*" | awk '{print $4}')
  [ -z "$ce" ] && continue
  echo "1e18 $a: $ce"
  if [ -z "$best" ] || "$PY" -c "import sys; sys.exit(0 if float('$ce') < float('$bestce') else 1)"; then best=$a; bestce=$ce; fi
done
echo "### 1e18 pick $best ($bestce; temporal mean 3.9077, free mean 3.9235, bar 3.8977) $(date -u +%H:%M)"
if [ -n "$best" ] && "$PY" -c "import sys; sys.exit(0 if float('$bestce') <= 3.9077 - 0.010 else 1)"; then
  echo "### reuse 1e19 START $best $(date -u +%H:%M)"
  ARM=$best GRAIN=1 bash scripts/residency/orchestration/tmoe_curriculum_1e19.sh
else
  echo "### reuse 1e19 skipped: 1e18 did not clear the bar $(date -u +%H:%M)"
fi
echo "### reuse ALL DONE $(date -u +%H:%M)"
