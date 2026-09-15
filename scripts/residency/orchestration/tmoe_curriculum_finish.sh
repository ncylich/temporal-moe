#!/usr/bin/env bash
# Remaining curriculum stages after the speed change (TE cross-entropy, mb 128): the C0 replicate on
# grain 3 (noise floor and end-to-end check of the new configuration), then the grain-1 transfer of
# the best arm with its own C0. Log: /workspace/rerun-logs/curriculum_finish.out
set -uo pipefail; cd "$(dirname "$0")/../../.."
. scripts/env.sh
O=scripts/residency/orchestration
echo "### finish START $(date -u +%H:%M)"
echo "### stage replicate grain3 C0b $(date -u +%H:%M)"
GRAIN=3 ARMS="C0b" bash $O/tmoe_curriculum_1e17.sh
"$PY" analysis/residency/curriculum_csv.py 2>&1 | tail -n +1 | cut -c1-160
echo "### stage transfer grain1 C0 SHD0p01 $(date -u +%H:%M)"
GRAIN=1 ARMS="C0 SHD0p01" bash $O/tmoe_curriculum_1e17.sh
"$PY" analysis/residency/curriculum_csv.py 2>&1 | tail -n +1 | cut -c1-160
echo "### finish ALL DONE $(date -u +%H:%M)"
