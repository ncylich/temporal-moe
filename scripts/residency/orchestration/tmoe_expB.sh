#!/usr/bin/env bash
# EXP B: KL anchor on the CONSTRAINED arm. The rebuild anchors to the base's FREE-routing
# logprobs, which pulls the adapted model toward behaviour it cannot exhibit under R=8.
# Anchoring to the base's own constrained behaviour instead lets the adapter move away from
# free-mode habits that are unreachable anyway. Same pool, same trajectories, same budget.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/dev/shm/gemma4-26b-it; A=/workspace/olmoe-adapt/data/gemma_ce_klcons_adapter.pt
M=/root/models/gemma4-klcons-merged; KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref_constrained.pt
G=scripts/residency/wait_gpu_free.sh
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 --max-seq 4096
        --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A"
scripts/residency/disk_budget.sh || exit 3
echo "### klcons KL precompute (constrained arm) $(date -u +%H:%M)"
[ -s "$KL" ] || { $G 120 3600; /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  $COMMON --precompute-kl $KL --kl-arm constrained; }
echo "### klcons train $(date -u +%H:%M)"
[ -s "$A" ] || { $G 120 3600; /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  $COMMON --accum 16 --lr 3e-5 --tokens 3400000 --kl-anchor $KL --kl-weight 0.05 --kl-arm constrained; }
echo "### klcons merge $(date -u +%H:%M)"
[ -d $M ] || { $G 120 3600; /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  $COMMON --merge-out $M; cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### klcons eval $(date -u +%H:%M)"
$G 120 3600
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_klcons_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### klcons ALL DONE $(date -u +%H:%M)"
