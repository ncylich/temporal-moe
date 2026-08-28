#!/usr/bin/env bash
# TRUE on-policy distillation loop (gemma). Each round: sample from the CURRENT adapter under
# R8 -> teacher (frozen free base) log-probs on those samples -> continue the same adapter on
# the reverse-KL sampled-token objective (+ the free-arm anchor, no CE) -> merge -> GSM8K.
# The next round samples from the adapter just trained, so the behaviour policy IS the
# policy being trained at the start of every round.
#   tmoe_gemma_onpolicy_loop.sh [start-adapter=digit3] [rounds=3] [tokens-per-round=850000]
# Expectation per round (fast serving path): sample ~11 min, teacher ~5, train ~18 (0.85M at
# ~800 tok/s), merge+verify ~3, GSM8K x3 arms ~8  => ~45 min.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
START=${1:-digit3}; ROUNDS=${2:-3}; T=${3:-850000}
B=/dev/shm/gemma4-26b-it; L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data
KL=/workspace/instruct-traj/gemma4_d7_seq4096_klref.pt
Q="mathlane_v2=2341,d5_fewshot=1183,domain8k=1000"
COMMON="--model $B --family gemma4 --no-unsloth --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16"
CUR_A=$D/gemma_ce_${START}_adapter.pt; CUR_M=/root/models/gemma4-${START}-merged
SEEN=$(/workspace/venv_fla/bin/python -c "import torch,sys; print(int(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['seen']))" $CUR_A)
echo "### onpolicy loop from $START (seen=$SEEN), $ROUNDS rounds x $T tokens $(date -u +%H:%M)"
for r in $(seq 1 $ROUNDS); do
  NAME=onp_r$r; TRAJ=gemma4_${NAME}_samples; TP=/workspace/instruct-traj/$TRAJ.pt; AKL=/workspace/instruct-traj/${TRAJ}_klref.pt
  A=$D/gemma_ce_${NAME}_adapter.pt; M=/root/models/gemma4-${NAME}-merged
  scripts/residency/disk_budget.sh || exit 3
  echo "### $NAME 1/5 sample from $(basename $CUR_M) under R8 $(date -u +%H:%M) (expect ~11 min)"
  [ -s $TP ] || $L /workspace/venv_vllm312/bin/python -u analysis/residency/selfgen_traj.py \
    --path $CUR_M --R 8 --prompts $D/d7_prompts.jsonl --quota "$Q" --max-new 1024 --max-model-len 2560 --gpu-mem 0.85 --out $TP
  echo "### $NAME 2/5 teacher log-probs $(date -u +%H:%M) (expect ~5 min)"
  [ -s $AKL ] || $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --traj $TRAJ \
    --out /tmp/gce_precompute_unused.pt --precompute-kl $AKL
  SEEN=$((SEEN + T))
  echo "### $NAME 3/5 reverse-KL distillation, continue $(basename $CUR_A) to seen=$SEEN $(date -u +%H:%M) (expect ~18 min)"
  [ -f $A.done ] || { cp $CUR_A $A
    $L /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --traj gemma4_d7_seq4096 --out $A --resume \
      --accum 16 --lr 3e-5 --tokens $SEEN --kl-only --kl-anchor $KL --kl-weight 0.05 \
      --aux-traj $TRAJ --aux-kl-anchor $AKL --aux-loss revkl --aux-kl-weight 1.0
    touch $A.done; }
  echo "### $NAME 4/5 merge + verify $(date -u +%H:%M)"
  [ -d $M ] || { $L /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py $COMMON --traj gemma4_d7_seq4096 --out $A --merge-out $M
    cp $B/processor_config.json $M/ 2>/dev/null || true; }
  /workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
  echo "### $NAME 5/5 GSM8K n=1319 $(date -u +%H:%M) (expect ~8 min)"
  $L /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
    --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_${NAME}_n1319 \
    --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
  echo "### $NAME ROUND DONE $(date -u +%H:%M)"
  [ "$r" -gt 1 ] && rm -rf /root/models/gemma4-onp_r$((r-1))-merged      # adapters are kept; merges are regenerable
  CUR_A=$A; CUR_M=$M
done
echo "### onpolicy loop ALL DONE $(date -u +%H:%M)"
