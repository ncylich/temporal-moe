#!/usr/bin/env bash
# WB finals retry, both models. Root cause of the 21:50/21:54 failures: the finals chain passed
# "free,R8,R16" as wb_arm.sh's ARMS, but wb_arm generates free itself and ARMS must be constrained
# arms only -- the engage check then compared free vs free (50/50 identical, assert fires).
# Gemma also had a stale 06:04 free-arm file in /workspace/writingbench/responses (deleted here).
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=3
L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data
GREC=gemma4_ce_online_scratch_e16_klT2_rho0
rm -f /workspace/writingbench/responses/${GREC}_*.jsonl
TMOE_ADAPTER=$D/gemma_ce_online_scratch_e16_klT2_adapter.pt $L scripts/residency/wb_arm.sh /dev/shm/gemma4-26b-it $GREC R8,R16
echo "### wb gemma retry done rc=$? $(date -u +%H:%M)"
TMOE_ADAPTER=$D/qwen_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt $L scripts/residency/wb_arm.sh /root/models/qwen35-35b-a3b qwen35_ce_online_fullpool_full_rho0 R8,R32
echo "### wb qwen retry done rc=$? $(date -u +%H:%M)"
echo "### wb finals RETRY ALL DONE $(date -u +%H:%M)"
