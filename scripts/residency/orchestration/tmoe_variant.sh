#!/bin/bash
# One rebuild variant: train, merge, verify, score GSM8K at n=1319 against the
# contemporaneous base (gemma4_instruct_n1319, same settings, same questions).
# Target: beat the current rebuild's same-arm R8 delta of +3.1 and close on D12's +6.0.
set -euo pipefail
cd /workspace/temporal-moe
NAME=$1; shift
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/dev/shm/gemma4-26b-it
A=/workspace/olmoe-adapt/data/gemma_ce_${NAME}_adapter.pt
M=/root/models/gemma4-${NAME}-merged
KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref.pt
COMMON="--model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096
        --max-seq 4096 --opt adamw --micro-batch 16 --out $A"
scripts/residency/disk_budget.sh || exit 3
echo "### variant $NAME: $* $(date -u +%H:%M)"
[ -s "$A" ] || /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  $COMMON --accum 16 --kl-anchor $KL "$@"
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  $COMMON $(echo "$@" | grep -oE '\-\-expert-lora-r [0-9]+') --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
scripts/residency/gpu_lease.sh /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_${NAME}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### variant $NAME DONE $(date -u +%H:%M)"
