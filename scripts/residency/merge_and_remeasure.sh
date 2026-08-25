#!/bin/bash
# RECOVER_DATA_PLAN section 1.5 -- merge each retrained adapter and regenerate its grid.
#
#     merge_and_remeasure.sh gemma    # GPU 1
#     merge_and_remeasure.sh qwen     # GPU 2
#
# TWO producers per model, which is why every record in this program has a _dual sibling:
#   instruct_genbench_vllm.py  -> GSM8K 200, IFEval 200, HumanEval full, under <record>
#   mmlu_gptoss.py             -> MMLU, under <record>_dual
# MMLU must come from mmlu_gptoss.py. It is NOT an lm_eval task: the reported metric is
# acc,relaxed-extract, which only that producer emits, while instruct_genbench_vllm.py's
# mmlu_flan_cot_fewshot records the STRICT filter alone. Wiring the strict number in as if
# it were the reported one is exactly the metric-provenance bug that hit the Group A gemma
# MMLU re-run (strict said damage widened -9.2 to -12.3; relaxed says it shrank -4.4 to
# -1.3) and, separately, OLMoE's headline mean (-14.8 vs -11.8). Check the producer, not
# just the CSV.
set -euo pipefail

ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1 HF_ALLOW_CODE_EVAL=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT

$ROOT/scripts/residency/disk_budget.sh || exit 3
# SPACE CHECK: a merge writes a full model copy (49-67GB). /dev/shm filling mid-write kills
# the merge with a bare "No space left on device" from safetensors, after minutes of work.
free_kb=$(df -k /root/models | tail -1 | awk '{print $4}')
if [ "$free_kb" -lt 75000000 ]; then
  echo "### ABORT: /root/models has only $((free_kb/1024/1024))GB free; a merge needs ~70GB." >&2
  echo "### Delete finished merged checkpoints (they are reproducible from adapter+base)." >&2
  exit 3
fi

VPY=/workspace/venv_vllm312/bin/python      # serving stack
TPY=/workspace/venv_fla/bin/python          # training stack (merge)
G=analysis/residency/instruct_genbench_vllm.py
M=analysis/residency/mmlu_gptoss.py
DATA=/workspace/olmoe-adapt/data
LOG=${LOG_DIR:-/workspace/rerun-logs}
mkdir -p $LOG

# GPU overridable, same as the other lanes: this box is shared AND our own lanes move
# around. A merge launched without honouring GPU= landed on top of the think-on trajectory
# job and OOM'd against its 136 GiB.
case "${1:?usage: merge_and_remeasure.sh gemma|qwen}" in
  gemma)
    DEV=${GPU:-1}; BASE=/dev/shm/gemma4-26b-it; MERGED=/root/models/gemma4-realmath-merged
    ADAPTER=$DATA/gemma_ce_realmath_adapter.pt; REC=gemma4_ce_realmath
    MODEL_KEY=gemma4_instruct; ARMS=free,R8,R16; RANK=32; TRAJ=gemma4_d7_seq4096 ;;
  qwen)
    DEV=${GPU:-2}; BASE=/dev/shm/qwen35-35b-a3b; MERGED=/root/models/qwen35-realmath-merged
    ADAPTER=$DATA/qwen_ce_realmath_adapter.pt; REC=qwen35_ce_realmath
    MODEL_KEY=qwen35_instruct; ARMS=free,R8,R16; RANK=16; TRAJ=qwen35_d7_seq4096 ;;
  *) echo "unknown: $1" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$DEV
[ -s "$ADAPTER" ] || { echo "adapter missing: $ADAPTER" >&2; exit 2; }

if [ ! -d "$MERGED" ]; then
  echo "### $1 MERGE $(date -u +%H:%M)"
  if [ "$1" = "qwen" ]; then
    # NEVER --merge-out for qwen. The CE trainer only ever holds the text-only submodule,
    # so save_pretrained writes a checkpoint that never contained visual.* and this vLLM
    # has no working text-only serving path for the family (TODO section 6 / task #78).
    # qwen_ce_patch streams the full multimodal base shard-by-shard and patches only the
    # text side, so the vision tower survives by never being dropped.
    ADAPTER_PATH=$ADAPTER DST_PATH=$MERGED SRC_PATH=$BASE \
      $TPY analysis/residency/qwen_ce_patch.py 2>&1 | tee $LOG/merge_${1}.log
  else
    # --expert-lora-r MUST be passed at merge time or unsloth builds attention-only LoRA
    # modules and silently misses elora_gu_A/B and elora_dp_A/B (commit ae505b7).
    # --traj is required even for a merge: train_gemma_ce.py loads the trajectory file
    # unconditionally at startup, before it reaches the merge branch, so omitting it falls
    # back to the gemma4_train5k default and dies with FileNotFoundError. The published
    # wb_matrix3.sh merge omitted it and only worked because that file existed on that pod.
    # --no-unsloth at MERGE time too, matching how the adapter was TRAINED. peft on the
    # HF stack targets gemma4's wrapped projections as q_proj.linear (transformers 5.x
    # wraps them in Gemma4ClippableLinear), so the checkpoint's attention-LoRA keys are
    # named for that module path. Loading under unsloth builds a differently-named module
    # tree, and --merge-out has no every-tensor-consumed assertion the way qwen_ce_patch.py
    # does -- a mismatch would drop the attention LoRA silently and yield a checkpoint that
    # looks fine and is wrong. Stacks must match across train and merge.
    $TPY analysis/residency/train_gemma_ce.py --model $BASE --family gemma4 --no-unsloth \
        --traj $TRAJ --max-seq 4096 \
        --expert-lora-r $RANK --out $ADAPTER --merge-out $MERGED \
        2>&1 | tee $LOG/merge_${1}.log
    # vLLM's engine boot fails on gemma4's multimodal processor class without this
    # (commit d0f67aa); save_pretrained does not carry it across.
    cp $BASE/processor_config.json $MERGED/ 2>/dev/null || true
  fi
fi
echo "### $1 MERGE DONE $(date -u +%H:%M)"

# Never trust a merge that "succeeded". On 2026-08-25 a merge reported success while
# carrying expert-LoRA and NO attention LoRA, because peft had attached the attention
# adapter to gemma4's vision tower. Diff every trained surface against base before any
# grid is generated from it -- a wrong grid is far more expensive than a failed merge.
$TPY analysis/residency/verify_merge.py --base $BASE --merged $MERGED \
    2>&1 | tee $LOG/verify_${1}.log
echo "### $1 VERIFY DONE $(date -u +%H:%M)"

echo "### $1 REMEASURE gsm8k/ifeval/humaneval $(date -u +%H:%M)"
$VPY -u $G --model $MODEL_KEY --path $MERGED --arms $ARMS --record-as $REC \
    --tasks "gsm8k_cot_zeroshot=200,ifeval=200,humaneval=0" \
    --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94 \
    2>&1 | tee $LOG/remeasure_${1}_gen.log
echo "### $1 REMEASURE MMLU (relaxed, the reported metric) $(date -u +%H:%M)"
$VPY -u $M --model $MODEL_KEY --path $MERGED --arms $ARMS --record-as ${REC}_dual \
    --gpu-mem 0.94 2>&1 | tee $LOG/remeasure_${1}_mmlu.log
echo "### $1 ALL DONE $(date -u +%H:%M)"
