#!/usr/bin/env bash
# Sampler-side cost of the residency walker at batch 256: greedy free arm (walker off) vs R8 (walker on), CUDA-graph path, qwen raw+adapter.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad
echo "### walker cost qwen n=256 max-new 768 $(date -u +%H:%M)"
scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/parity_vllm.py --path /root/models/qwen35-35b-a3b --adapter /workspace/olmoe-adapt/data/qwen_ce_digit3_adapter.pt --R 8 --n 256 --max-new 768 --think off --gpu-mem 0.85 --max-model-len 2560 --out $S/walker_cost_qwen.json
echo "### walker cost gemma n=256 max-new 768 $(date -u +%H:%M)"
scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/parity_vllm.py --path /dev/shm/gemma4-26b-it --R 8 --n 256 --max-new 768 --gpu-mem 0.85 --max-model-len 2560 --out $S/walker_cost_gemma.json
echo "### walker cost DONE $(date -u +%H:%M)"
