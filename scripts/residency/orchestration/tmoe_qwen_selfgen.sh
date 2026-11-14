#!/usr/bin/env bash
# Cross-model test of the selfgen finding. D7 with its math lane swapped for the
# self-generated GSM8K-shaped problems (the prompts gemma wrote; qwen generates its own
# responses), everything else matching our qwen rebuild: expert-LoRA r16, KL 0.1, 3.4M
# tokens, think-off, arms free/R8/R32. If qwen's GSM8K delta jumps from +2.1 toward the
# published +6.5, the published qwen number has the same explanation as gemma's.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/root/models/qwen35-35b-a3b; TAG=qwen35_selfgen; L=scripts/residency/gpu_lease.sh
A=/workspace/olmoe-adapt/data/qwen_ce_selfgen_adapter.pt; M=/root/models/qwen35-selfgen-merged
KL=/workspace/instruct-traj/${TAG}_seq4096_klref.pt
P=/workspace/olmoe-adapt/data_exp/pool_selfgen_math.jsonl
scripts/residency/disk_budget.sh || exit 3
echo "### qwen-selfgen trajectories $(date -u +%H:%M)"
[ -s /workspace/instruct-traj/${TAG}.pt ] || $L /workspace/venv_vllm312/bin/python -u analysis/residency/gen_traj_vllm.py \
  --model $B --tag $TAG --prompts $P --max-new 3072 --max-prompt-tok 1024 --think off --gpu-mem 0.90
[ -s /workspace/instruct-traj/${TAG}_seq4096.pt ] || /workspace/venv_fla/bin/python analysis/residency/cut_trajectories.py --tag $TAG --max-seq 4096
COMMON="--model $B --family qwen35 --no-unsloth --traj ${TAG}_seq4096 --max-seq 4096 --expert-lora-r 16 --opt adamw --micro-batch 16 --out $A"
echo "### qwen-selfgen KL $(date -u +%H:%M)"
[ -s "$KL" ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --precompute-kl $KL
echo "### qwen-selfgen train $(date -u +%H:%M)"
[ -s "$A" ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
  --accum 16 --lr 3e-5 --tokens 3400000 --kl-anchor $KL --kl-weight 0.1
echo "### qwen-selfgen merge $(date -u +%H:%M)"
[ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --merge-out $M
  /workspace/venv_fla/bin/python analysis/residency/textify_qwen_merge.py $M; }
echo "### qwen-selfgen eval $(date -u +%H:%M)"
$L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model qwen35_instruct --path $M --arms free,R8,R32 --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
  --record-as qwen35_ce_selfgen_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
echo "### qwen-selfgen ALL DONE $(date -u +%H:%M)"
