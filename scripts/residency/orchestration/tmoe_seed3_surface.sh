#!/usr/bin/env bash
# Second gemma run on the full five-benchmark surface. gemma_ce_rebuild's IFEval, MMLU and
# WritingBench cells rest on ONE run each; seed3 (the median D7 seed, GSM8K +3.1) already
# has GSM8K and both code surfaces. This fills IFEval, MMLU and WritingBench so the +1.6
# mean gets a second run behind every cell. Every GPU stage holds the lease.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-seed3-merged
A=/workspace/olmoe-adapt/data/gemma_ce_seed3_adapter.pt; L=scripts/residency/gpu_lease.sh
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { echo "### seed3 merge $(date -u +%H:%M)"
  $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
    --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096 \
    --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### seed3 IFEval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_seed3_full \
  --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### seed3 MMLU $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/mmlu_gptoss.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 \
  --record-as gemma4_ce_seed3_full_dual --gpu-mem 0.90
echo "### seed3 WritingBench $(date -u +%H:%M)"
export GPU=0
$L scripts/residency/wb_arm.sh $M gemma4_seed3 R8,R16
echo "### seed3 SURFACE DONE $(date -u +%H:%M)"
