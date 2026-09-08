#!/usr/bin/env bash
# 5-min smoke of the cache_bias walker inside vLLM (gemma base, 8 prompts, R8 arm = walker on): runs, masks k experts, reports loads/token.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
export TEMPORAL_WALKER=cache_bias TEMPORAL_CB_C=64 TEMPORAL_CB_LAMBDA=0.5 TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1
S=/tmp/claude-0/-workspace-temporal-moe/c5a8032b-6316-4ce6-b1ab-aa68b4e2178a/scratchpad
echo "### cache_bias smoke gemma C=64 lambda=0.5 $(date -u +%H:%M)"
scripts/residency/gpu_lease.sh /workspace/venv_vllm312/bin/python -u analysis/residency/parity_vllm.py --path /dev/shm/gemma4-26b-it --R 8 --n 8 --max-new 128 --gpu-mem 0.85 --out $S/cb_smoke_gemma.json
echo "### cache_bias smoke DONE rc=$? $(date -u +%H:%M)"
