#!/usr/bin/env bash
# Full think-on surface (user 2026-08-31: "don't just do GSM8K evals, do full surface").
# Chain's stage 2 already does GSM8K for the adapter; this adds IFEval/MMLU/HumanEval/MBPP for the
# adapter arms, then the BASE (no adapter) 5-task surface at full n for the deltas (the think
# ablation's base rows are n=200). Caps = the ablation's fair budgets: IFEval 16384, others 8192.
#   tmoe_think_surface.sh <gemma|qwen>
set -uo pipefail; cd /workspace/temporal-moe
MODEL=${1:?gemma|qwen}
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=4 HF_ALLOW_CODE_EVAL=1
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data
if [ "$MODEL" = gemma ]; then
  B=/dev/shm/gemma4-26b-it; AD=$D/gemma_ce_online_scratch_e16_think_adapter.pt; ARMS=free,R8,R16; PFX=gemma4_ce_online_think; BPFX=gemma4_think_on_fulln
  for PASS in "adapter" "base"; do
    if [ "$PASS" = adapter ]; then A="--adapter $AD"; T=$PFX; else A=""; T=$BPFX
      echo "### think-surface gemma BASE GSM8K $(date -u +%H:%M)"
      $L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $B --think on --arms $ARMS --record-as ${T}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90; fi
    echo "### think-surface gemma $PASS IFEval $(date -u +%H:%M)"
    $L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $B $A --think on --arms $ARMS --record-as ${T}_full --tasks "ifeval=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90
    echo "### think-surface gemma $PASS MMLU $(date -u +%H:%M)"
    $L $PY -u analysis/residency/mmlu_gptoss.py --model gemma4_instruct --path $B $A --think on --arms $ARMS --record-as ${T}_full_dual --gen-cap 8192 --max-model-len 10240 --gpu-mem 0.90
    echo "### think-surface gemma $PASS HumanEval $(date -u +%H:%M)"
    $L $PY -u analysis/residency/humaneval_gemma.py --path $B $A --think on --arms $ARMS --tag ${T}_he8192 --max-tokens 8192 --max-model-len 10240
    echo "### think-surface gemma $PASS MBPP $(date -u +%H:%M)"
    $L $PY -u analysis/residency/mbpp_gemma.py --path $B $A --think on --arms $ARMS --tag ${T}_m8192 --max-tokens 8192 --max-model-len 10240 --gpu-mem 0.90
  done
else
  B=/root/models/qwen35-35b-a3b; AD=$D/qwen_ce_online_scratch_e16_think_adapter.pt; ARMS=free,R8,R32; PFX=qwen35_ce_online_think; BPFX=qwen35_think_on_fulln
  Q="--model qwen35_instruct --path $B --think on --temperature 0.6 --top-p 0.95 --presence-penalty 1.5"
  for PASS in "adapter" "base"; do
    if [ "$PASS" = adapter ]; then A="--adapter $AD"; T=$PFX; else A=""; T=$BPFX
      echo "### think-surface qwen BASE GSM8K $(date -u +%H:%M)"
      $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q --arms $ARMS --record-as ${T}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90; fi
    echo "### think-surface qwen $PASS IFEval $(date -u +%H:%M)"
    $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q $A --arms $ARMS --record-as ${T}_full --tasks "ifeval=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90
    echo "### think-surface qwen $PASS MMLU $(date -u +%H:%M)"
    $L $PY -u analysis/residency/mmlu_gptoss.py --model qwen35_instruct --path $B $A --think on --arms $ARMS --record-as ${T}_n_dual --gen-cap 8192 --max-model-len 10240 --gpu-mem 0.90
    echo "### think-surface qwen $PASS code $(date -u +%H:%M)"
    $L $PY -u analysis/residency/instruct_genbench_vllm.py $Q $A --arms $ARMS --record-as ${T}_code --tasks "mbpp_instruct=0,humaneval_instruct=0" --gen-cap 8192 --max-model-len 10240 --gpu-mem 0.90
  done
fi
echo "### think-surface $MODEL ALL DONE $(date -u +%H:%M)"
