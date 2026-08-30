#!/usr/bin/env bash
# Retry the gemma half of wb finals: the 06:04 stale free-arm response file (in /workspace/writingbench/responses,
# which the finals chain's rm missed by deleting in results/ablations/writingbench) made the engage check see
# R8 == free. Delete ALL gemma-final response files and regenerate under one lease at prio 3.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=3
GREC=gemma4_ce_online_scratch_e16_klT2_rho0
rm -f /workspace/writingbench/responses/${GREC}_*.jsonl
TMOE_ADAPTER=/workspace/olmoe-adapt/data/gemma_ce_online_scratch_e16_klT2_adapter.pt scripts/residency/gpu_lease.sh scripts/residency/wb_arm.sh /dev/shm/gemma4-26b-it $GREC free,R8,R16
echo "### wb gemma retry done rc=$? $(date -u +%H:%M)"
