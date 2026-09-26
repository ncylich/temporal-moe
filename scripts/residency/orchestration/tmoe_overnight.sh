#!/usr/bin/env bash
# Overnight driver, single GPU. Pools A and C are already spliced. Runs the two remaining
# data experiments back to back; each chain's GPU stages hold the device lease, so this
# driver cannot collide with EXP B's eval or with itself. A failed chain STOPS the driver
# (set -e) instead of being skipped -- the previous version marched past mathx2's failure
# and launched pool3x on top of B.
set -euo pipefail
cd /workspace/temporal-moe
E=/workspace/olmoe-adapt/data_exp
echo "### overnight: mathx2 $(date -u +%H:%M)"
/workspace/tmoe_data_arm.sh mathx2 $E/pool_mathx2.jsonl
echo "### overnight: pool3x $(date -u +%H:%M)"
/workspace/tmoe_data_arm.sh pool3x $E/pool_3x.jsonl
echo "### OVERNIGHT ALL DONE $(date -u +%H:%M)"
