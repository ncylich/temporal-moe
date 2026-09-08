#!/usr/bin/env bash
set -uo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
echo "### smoke 1/2 reference dump from the merged W=3 checkpoint $(date -u +%H:%M) (expect ~3 min)"
[ -s /tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad/ref_digit3.json ] || $L $PY -u analysis/residency/parity_vllm.py --path /root/models/gemma4-digit3-merged --R 8 --n 8 --max-new 256 --out /tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad/ref_digit3.json
echo "### smoke 2/2 trainer + in-process engine, sync the W=3 adapter, greedy on the same prompts $(date -u +%H:%M) (expect ~8 min)"
cp /workspace/olmoe-adapt/data/gemma_ce_digit3_adapter.pt /tmp/smoke_adapter.pt
$L $PY -u analysis/residency/train_gemma_ce.py --model /dev/shm/gemma4-26b-it --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 \
  --expert-lora-r 32 --opt adamw --micro-batch 16 --out /tmp/smoke_adapter.pt --resume --tokens 1 \
  --online-every 16 --online-n 32 --online-smoke /tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad/ref_digit3.json
echo "### smoke DONE rc=$? $(date -u +%H:%M)"
