#!/usr/bin/env bash
# Unattended driver of results/ablations/CURRICULUM_PLAN.md: round 1 on grain 3, the decision rules
# (analysis/residency/curriculum_decide.py) pick round 2, the best recipe transfers to grain 1, a win
# promotes to 1e18 and a 1e18 win to 1e19. Each stage writes curriculum_1e17.csv and a line here.
# Log: /workspace/rerun-logs/curriculum_supervisor.out
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
O=scripts/residency/orchestration
csv() { "$PY" analysis/residency/curriculum_csv.py 2>&1 | tail -n +1 | cut -c1-160; }
echo "### supervisor START $(date -u +%H:%M)"
echo "### stage round1 $(date -u +%H:%M)"
GRAIN=3 bash $O/tmoe_curriculum_1e17.sh
csv
echo "### verdict round1: $("$PY" analysis/residency/curriculum_decide.py best)"
R2=$("$PY" analysis/residency/curriculum_decide.py round2)
if [ -n "$R2" ]; then
  echo "### stage round2 arms=[$R2] $(date -u +%H:%M)"
  GRAIN=3 ARMS="$R2" bash $O/tmoe_curriculum_1e17.sh
  csv
  echo "### verdict round2: $("$PY" analysis/residency/curriculum_decide.py best)"
else
  echo "### round2 skipped: every round-1 arm lost by more than 0.005 $(date -u +%H:%M)"
fi
T=$("$PY" analysis/residency/curriculum_decide.py transfer)
if [ -n "$T" ]; then
  echo "### stage transfer grain1 arm=$T $(date -u +%H:%M)"
  GRAIN=1 ARMS="$T" bash $O/tmoe_curriculum_1e17.sh
  csv
fi
P=$("$PY" analysis/residency/curriculum_decide.py promote)
if [ -n "$P" ]; then
  echo "### stage promote 1e18 arm=$P $(date -u +%H:%M)"
  ARM=$P bash $O/tmoe_curriculum_1e18.sh
  # 1e18 verdict against the flame38m_g3_moe seed-triplet mean 4.0136 (bar 0.010)
  CE18=$(grep "on test set" results/phase0/runs/cur_flame38m_g3_$P/train.log 2>/dev/null | tail -1 | grep -o "lm loss value: [0-9.E+]*" | awk '{print $4}')
  echo "### 1e18 result arm=$P test_CE=$CE18 (baseline mean 4.0136, bar 4.0036) $(date -u +%H:%M)"
  if [ -n "$CE18" ] && "$PY" -c "import sys; sys.exit(0 if float('$CE18') <= 4.0136 - 0.010 else 1)"; then
    echo "### stage promote 1e19 arm=$P $(date -u +%H:%M)"
    ARM=$P bash $O/tmoe_curriculum_1e19.sh
  else
    echo "### 1e19 skipped: 1e18 did not clear the bar $(date -u +%H:%M)"
  fi
else
  echo "### promotion skipped $(date -u +%H:%M)"
fi
echo "### supervisor ALL DONE $(date -u +%H:%M)"
