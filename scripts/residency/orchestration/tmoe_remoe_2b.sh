#!/usr/bin/env bash
# ReMoE 2b, the fair-surface variant: ReMoE's recency-reuse objective on D12's OWN training
# surface (expert-LoRA + attention-LoRA, NOT router-only), residency constraint OFF, lr = the
# incumbent surface's 1e-4 so only the objective varies. Triggered by the doc's rule: the
# faithful router-only arm underperformed at R8. GSM8K free/R8 n=1319 is the instrument.
#   tmoe_remoe_2b.sh <gemma|qwen>
set -uo pipefail; cd /workspace/temporal-moe
MODEL=$1
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMOE_PRIO=4
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python; D=/workspace/olmoe-adapt/data
if [ "$MODEL" = gemma ]; then B=/dev/shm/gemma4-26b-it; COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16"
  G="--model gemma4_instruct --path $B"; ML=4096; PFX=gemma4_remoe2b
else B=/root/models/qwen35-35b-a3b; COMMON="--model $B --family qwen35 --no-unsloth --traj qwen35_d7_seq4096 --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16"
  G="--model qwen35_instruct --path $B --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5"; ML=5632; PFX=qwen35_remoe2b; fi
A=$D/${PFX}_adapter.pt
echo "### remoe2b $MODEL train (LoRA surface, recency objective, residency off, lr 1e-4, 3.4M) $(date -u +%H:%M)"
[ -f $A.done ] || { $L $PY -u analysis/residency/train_gemma_ce.py $COMMON --out $A --no-constraint --remoe-lambda 1.0 --remoe-gamma 0.9 --extra-lr-div 1 --accum 16 --lr 1e-4 --tokens 3400000 && touch $A.done; }
echo "### remoe2b $MODEL GSM8K free,R8 n=1319 $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py $G --adapter $A --arms free,R8 --record-as ${PFX}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len $ML --gpu-mem 0.90
echo "### remoe2b $MODEL DONE $(date -u +%H:%M)"
