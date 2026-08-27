#!/usr/bin/env bash
# Overnight driver, single GPU. Waits for the CPU-side pool builds, splices A and C
# correctly (realmath lane first), then runs the three data experiments back to back.
# B is already running on the GPU; A and C queue behind it via the free-device gate.
set -uo pipefail
cd /workspace/temporal-moe
D=/workspace/olmoe-adapt/data; E=/workspace/olmoe-adapt/data_exp; PY=/workspace/venv_fla/bin/python
until [ -s $E/realmath_4700.jsonl ]; do sleep 60; done
$PY analysis/residency/splice_pool.py --base $D/d7_prompts.jsonl --realmath $E/realmath_4700.jsonl \
   --math-rows 4700 --total 8482 --out $E/pool_mathx2.jsonl
echo "### pool A (math x2) spliced $(date -u +%H:%M)"
/workspace/tmoe_data_arm.sh mathx2 $E/pool_mathx2.jsonl
until [ -s $E/realmath_7023.jsonl ]; do sleep 60; done
$PY analysis/residency/splice_pool.py --base /workspace/olmoe-adapt/data_big/d7_prompts.jsonl \
   --realmath $E/realmath_7023.jsonl --math-rows 7023 --total 25446 --out $E/pool_3x.jsonl
echo "### pool C (3x, realmath-correct) spliced $(date -u +%H:%M)"
/workspace/tmoe_data_arm.sh pool3x $E/pool_3x.jsonl
echo "### OVERNIGHT ALL DONE $(date -u +%H:%M)"
