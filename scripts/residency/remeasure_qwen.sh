#!/bin/bash
# qwen35 adapted grid, in THINK-OFF mode -- matching how the adapter was trained.
#
# The adapter trains on the model's own THINK-OFF trajectories, so the grid must be
# think-off too. Leaving --think at its default means thinking ON for qwen3.5: responses
# run past the 2048 cap (150 of 200 truncated), IFEval scores 0.27 against a real value
# near 0.85, and the whole grid is garbage that looks like catastrophic damage.
#
# Reference row is qwen35_think_off at cap 2048: free GSM8K 0.850, IFEval 0.850,
# humaneval_instruct 0.933, mmlu_flan 0.851. Sampling is qwen's non-thinking card recipe
# (0.7 / 0.8), as used by scripts/residency/final_reruns.sh for that record.
#
# HumanEval note: think-OFF emits no channel markers, so the stock humaneval_instruct task
# with pass@1,create_test is correct here -- humaneval_think is for the think-ON record.
set -euo pipefail
ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT
PY=/workspace/venv_vllm312/bin/python
LOG=${LOG_DIR:-/workspace/rerun-logs}
M=/dev/shm/qwen35-rebuild-merged
export CUDA_VISIBLE_DEVICES=${GPU:-2}

echo "### qwen REMEASURE think-off gsm8k/ifeval/humaneval $(date -u +%H:%M)"
$PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path $M \
    --arms free,R8,R16 --record-as qwen35_ce_rebuild \
    --think off --temperature 0.7 --top-p 0.8 \
    --tasks "gsm8k_cot_zeroshot=200,ifeval=200,humaneval_instruct=0" \
    --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94 2>&1 | tee $LOG/remeasure_qwen_gen.log

echo "### qwen REMEASURE MMLU (relaxed, the reported metric) $(date -u +%H:%M)"
$PY -u analysis/residency/mmlu_gptoss.py --model qwen35_instruct --path $M \
    --arms free,R8,R16 --record-as qwen35_ce_rebuild_dual --think off \
    --gpu-mem 0.94 2>&1 | tee $LOG/remeasure_qwen_mmlu.log
echo "### qwen REMEASURE DONE $(date -u +%H:%M)"
