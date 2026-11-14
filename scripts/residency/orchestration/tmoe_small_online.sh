#!/usr/bin/env bash
# On-policy adaptation (the gemma/qwen winner recipe: from scratch, reverse KL T=2, lr 1e-4, no anchor,
# 3.4M sampled tokens, refresh every 16 steps x 256 rows, budget on sampled tokens) for the two released
# models the paper leaves unadapted: OLMoE-1B-7B-Instruct (R=k=8) and LFM2.5-8B-A1B (R=k=4).
# Then GSM8K n=1319 on free and R=k, adapter-direct, then the surface (IFEval/MMLU at each model's recorded
# budgets, inside OLMoE's 4096 window; no HumanEval for LFM: its thinking breaks the primed-fence task, see DATA_CONTRACT).
#   tmoe_small_online.sh <olmoe|lfm> [tokens=3400000] [every=16] [n=256]
set -uo pipefail; cd /workspace/temporal-moe
FAM=$1; T=${2:-3400000}; EVERY=${3:-16}; N=${4:-256}
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_ALLOW_CODE_EVAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMOE_PRIO=${TMOE_PRIO:-4}
L=scripts/residency/gpu_lease.sh; D=/workspace/olmoe-adapt/data; PY=/workspace/venv_vllm312/bin/python
TRAJ=$([ "$1" = lfm ] && echo lfm25_d7_seq4096 || echo olmoe_d7_seq4096)
case $FAM in
  olmoe) B=/workspace/instruct-models/olmoe-0125-instruct; MODEL=olmoe_instruct; R=8; ARMS=free,R8; GEN=2048; MML=4096; IFG=2048; IFM=4096; MMG=2048; MMM=4096; OMML=2560; PFX=olmoe_ce_online ;;
  lfm)   B=/workspace/instruct-models/lfm25-8b-a1b;        MODEL=lfm25_instruct; R=4; ARMS=free,R4; GEN=4096; MML=8192; IFG=8192; IFM=10240; MMG=4096; MMM=6144; OMML=4096; PFX=lfm25_ce_online ;;
  *) echo "unknown family $FAM"; exit 1 ;;
esac
NAME=${PFX}_scratch_e${EVERY}${TMOE_NAME_SUFFIX:-}; A=$D/${NAME}_adapter.pt
echo "### $NAME 1/3 train (family $FAM, R=$R, $T sampled tokens) $(date -u +%H:%M)"
[ -f $A.done ] || { rm -f $A.tmp
  $L $PY -u analysis/residency/train_gemma_ce.py --model $B --family $FAM --no-unsloth --traj ${TRAJ} --max-seq 4096 \
    --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --accum 16 --tokens $T --R $R --kl-only \
    --aux-loss ${TMOE_AUX_LOSS:-revkl_full} --aux-kl-weight 1.0 --aux-kl-temp ${TMOE_KL_TEMP:-2} --lr ${TMOE_LR:-1e-4} \
    --online-every $EVERY --online-n $N --online-max-new ${TMOE_MAXNEW:-1024} --online-think off --online-max-model-len $OMML \
    --online-temp 0.7 --online-quota "mathlane_v2=2341,d5_fewshot=1183,domain8k=1000" --online-prompts $D/d7_prompts.jsonl --budget-on sampled \
    && touch $A.done; }
[ -f $A.done ] || { echo "### $NAME training failed $(date -u +%H:%M)"; exit 1; }
echo "### $NAME 2/3 GSM8K n=1319 $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model $MODEL --path $B --adapter $A --arms $ARMS --record-as ${NAME}_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap $GEN --max-model-len $MML --gpu-mem 0.90
echo "### $NAME 3/3 surface $(date -u +%H:%M)"
$L $PY -u analysis/residency/instruct_genbench_vllm.py --model $MODEL --path $B --adapter $A --arms $ARMS --record-as ${NAME}_full \
  --tasks "ifeval=0" --gen-cap $IFG --max-model-len $IFM --gpu-mem 0.90
$L $PY -u analysis/residency/mmlu_gptoss.py --model $MODEL --path $B --adapter $A --arms $ARMS --record-as ${NAME}_n_dual --gen-cap $MMG --max-model-len $MMM --gpu-mem 0.90
[ $FAM = lfm ] || $L $PY -u analysis/residency/instruct_genbench_vllm.py --model $MODEL --path $B --adapter $A --arms $ARMS --record-as ${NAME}_code \
  --tasks "humaneval_instruct=0" --gen-cap $GEN --max-model-len $MML --gpu-mem 0.90
$L $PY -u analysis/residency/mbpp_chat.py --model $MODEL --path $B --adapter $A --arms $ARMS --tag ${NAME}_mbpp --think off --gpu-mem 0.90
echo "### $NAME ALL DONE $(date -u +%H:%M)"
