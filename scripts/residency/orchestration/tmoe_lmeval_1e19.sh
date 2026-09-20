#!/usr/bin/env bash
# 0-shot lm-eval for a 1e19 phase0 checkpoint on the six t19 tasks, the way t19_lmeval_stderr.csv
# was produced for the July cells (analysis/probes/run_lmeval.py, megatron_lm backend, native
# regime: TEMPORAL=1 for temporal runs). Geometry comes from the run's run.meta (s19opt: H=800,
# L=14, 50 heads, ffn 4272, experts/topk/moe_ffn/shared as trained). Writes
# results/phase0/runs/<run>/lmeval_0shot/, then analysis/probes/lmeval_to_csv.py regenerates
# results/ablations/t19_lmeval_stderr.csv from every 1e19 run it knows.
#
#   bash scripts/residency/orchestration/tmoe_lmeval_1e19.sh moe_fine_g3_1e19
set -uo pipefail
RUN=${1:?run name}
cd "$(dirname "$0")/../../.."; . scripts/env.sh
OUT=$ROOT/results/phase0/runs/$RUN; META=$(cat "$OUT/run.meta")
mg() { echo "$META" | grep -oE "[[:space:]]$1=[^ ]+" | head -1 | sed 's/.*=//'; }
H=$(mg H); L=$(mg L); HEADS=$(mg heads); FFN=$(mg ffn); MOE_FFN=$(mg moe_ffn)
NE=$(mg num_experts); TOPK=$(mg topk); SHARED=$(mg shared_int); TP=$(mg temporal)
TASKS=${TASKS:-arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande}
OUT_TAG=${OUT_TAG:-lmeval_0shot}
echo "### lmeval $RUN H=$H L=$L heads=$HEADS ffn=$FFN experts=$NE topk=$TOPK moe_ffn=$MOE_FFN shared=$SHARED temporal=$TP tasks=$TASKS $(date -u +%H:%M)"
MODEL_ARGS=(
  --hidden-size "$H" --ffn-hidden-size "$FFN" --num-layers "$L" --num-attention-heads "$HEADS"
  --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
  --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
  --moe-ffn-hidden-size "$MOE_FFN" --num-experts "$NE" --moe-router-topk "$TOPK"
  --moe-shared-expert-intermediate-size "$SHARED" --moe-layer-freq "[0]+[1]*$((L-1))"
  --moe-router-dtype fp32 --moe-router-pre-softmax --moe-router-score-function softmax
  --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001 --moe-grouped-gemm --moe-use-legacy-grouped-gemm
  --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model EleutherAI/pythia-12b
  --transformer-impl transformer_engine --no-gradient-accumulation-fusion --no-rope-fusion
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --use-distributed-optimizer
  --moe-token-dispatcher-type alltoall
)
export OMP_NUM_THREADS=16 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export TEMPORAL=$TP TEMPORAL_EVICT=min_logit MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1
export PYTHONPATH=$ROOT/lm-evaluation-harness:$ROOT/Megatron-LM:${PYTHONPATH:-}
cd "$ROOT/lm-evaluation-harness"
"$PY" -m torch.distributed.run --nproc_per_node=1 --rdzv-endpoint=localhost:${RDZV_PORT:-29541} \
  "$ROOT/analysis/probes/run_lmeval.py" "${MODEL_ARGS[@]}" \
  --bf16 --seq-length 2048 --micro-batch-size 32 --batch_size "${LM_BATCH:-16}" \
  --max-tokens-to-oom 10000000 --seed 42 --load "$OUT/ckpt" --model megatron_lm \
  --num_fewshot 0 --tasks "$TASKS" ${LIMIT:+--limit $LIMIT} \
  --output_path "$OUT/$OUT_TAG" 2>&1 | tee "$OUT/${OUT_TAG}.log" | grep -E "^\|.*\|.*\||Error|error|Traceback" | tail -40
echo "### lmeval $RUN rc=${PIPESTATUS[0]} $(date -u +%H:%M)"
cd "$ROOT" && "$PY" analysis/probes/lmeval_to_csv.py "$RUN"
