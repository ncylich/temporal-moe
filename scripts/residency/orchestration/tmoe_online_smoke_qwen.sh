#!/usr/bin/env bash
# qwen parity smoke: greedy generations from the in-process engine after syncing the W=3 qwen adapter must
# equal the merged qwen35-digit3 checkpoint's (eager mode, deterministic). ~15 min.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TEMPORAL_EAGER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad; L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
echo "### qwen smoke 1/2 reference dump (merged W=3 qwen, eager) $(date -u +%H:%M)"
[ -s $S/ref_qwen_digit3_eager.json ] || $L $PY -u analysis/residency/parity_vllm.py --path /root/models/qwen35-digit3-merged --R 8 --n 8 --max-new 256 --think off --out $S/ref_qwen_digit3_eager.json
echo "### qwen smoke 2/2 trainer + in-process engine, sync, greedy $(date -u +%H:%M)"
cp /workspace/olmoe-adapt/data/qwen_ce_digit3_adapter.pt /tmp/smoke_qwen_adapter.pt
$L $PY -u analysis/residency/train_gemma_ce.py --model /root/models/qwen35-35b-a3b --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 \
  --expert-lora-r 16 --opt adamw --micro-batch 16 --out /tmp/smoke_qwen_adapter.pt --resume --tokens 1 \
  --online-every 16 --online-n 32 --online-gpu-mem 0.45 --online-smoke $S/ref_qwen_digit3_eager.json
echo "### qwen smoke DONE rc=$? $(date -u +%H:%M)"
