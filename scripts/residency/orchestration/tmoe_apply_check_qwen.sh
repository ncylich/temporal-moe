#!/usr/bin/env bash
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
echo "### apply_adapter qwen check vs qwen35-digit3-merged $(date -u +%H:%M) (expect ~5 min)"
scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/apply_adapter.py --base /root/models/qwen35-35b-a3b --adapter /workspace/olmoe-adapt/data/qwen_ce_digit3_adapter.pt --check /root/models/qwen35-digit3-merged
echo "### apply_adapter qwen check DONE rc=$? $(date -u +%H:%M)"
