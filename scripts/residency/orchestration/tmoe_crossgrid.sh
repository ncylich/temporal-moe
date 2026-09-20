#!/usr/bin/env bash
# Cross-setting grid for the fair comparison: each method at the OTHER's operating point.
# Skliar cache at OUR memory (C=8: gemma 6.25%, qwen 3.1%), lambda 0 and 0.4, loads counted;
# OUR rolling residency at THEIR 50% memory (gemma R64, qwen R128), base and adapted.
# GSM8K n=1319 everywhere.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMOE_PRIO=4
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data
GB=/dev/shm/gemma4-26b-it; QB=/root/models/qwen35-35b-a3b
QS="--think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"
# 1) Skliar at our memory
for LAM in 0 0.4; do
  ( export TEMPORAL_WALKER=cache_bias TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_C=8
    echo "### crossgrid skliar C8 lam$LAM gemma $(date -u +%H:%M)"
    $L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $GB --arms R8 --record-as gemma4_skliar_C8_lam${LAM/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
    echo "### crossgrid skliar C8 lam$LAM qwen $(date -u +%H:%M)"
    $L $PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path $QB $QS --arms R8 --record-as qwen35_skliar_C8_lam${LAM/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90 )
done
# 2) Ours at their memory (50% resident), base and adapted
echo "### crossgrid ours R64 gemma base+adapted $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $GB --arms R64 --record-as gemma4_instruct_R64_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $GB --adapter $D/gemma_ce_online_scratch_e16_klT2_adapter.pt --arms R64 --record-as gemma4_klT2_R64_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### crossgrid ours R128 qwen base+adapted $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path $QB $QS --arms R128 --record-as qwen35_instruct_R128_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path $QB $QS --adapter $D/qwen_ce_online_online_scratch_e16_fullpool_e16_full_adapter.pt --arms R128 --record-as qwen35_fullpool_R128_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### crossgrid DONE $(date -u +%H:%M)"
