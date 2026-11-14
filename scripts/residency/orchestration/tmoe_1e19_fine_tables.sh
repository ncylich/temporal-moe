#!/usr/bin/env bash
# Complete the matched-pair tables with the 1e19 fine full MoE (moe_fine_g3_1e19), matching the
# July protocol for the other 1e19 cells: delex capture (structural table), activation capture
# (activation kurtosis), and the fake-quant eval at 16/8/4/3 bits on 16 x 256 test sequences.
# Runs on the fast sync-free expert path; fakequant_eval.py quantizes the legacy weight layout
# through per-expert transposes so the RTN groups are the same as on the TE layout. The first
# stage is a validation cell: the coarse full MoE at 4 bits must reproduce July's 3.137491.
set -uo pipefail
cd "$(dirname "$0")/../../.."; . scripts/env.sh
export CE_FUSION=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=${TMOE_PRIO:-4}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export TOKENIZER_MODEL=EleutherAI/pythia-12b DATA_DIR=/root/data/dclm_tokenized
export EXTRA_ARGS="--no-rope-fusion --moe-use-legacy-grouped-gemm --dist-ckpt-strictness log_all"
export MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1
stage() { # run, name, log, env...
  local run=$1 name=$2 log=$3; shift 3; local out=results/phase0/runs/$run
  echo "### $name $(date -u +%H:%M)"; cp $out/run.meta $out/run.meta.pretables
  env RUN_NAME=$run SHAPE=s19opt TARGET_FLOPS=1e19 SHARED_MULT=2 PEAK_LR=3e-4 WARMUP_FRAC=0.01 LR_DECAY_STYLE=WSD TEMPORAL_EVICT=min_logit SEED=1234 DENSE=0 "$@" \
      timeout -k 60 ${STAGE_TIMEOUT:-5400} scripts/residency/gpu_lease.sh bash experiments/run.sh > /workspace/rerun-logs/$log 2>&1
  local rc=$?; mv -f $out/run.meta.pretables $out/run.meta; echo "### $name rc=$rc $(date -u +%H:%M)"
}
R=moe_fine_g3_1e19; O=results/phase0/runs/$R
# validation: July's coarse full MoE, 4 bits, on the fast path -> quanteval_b4_fastcheck.log
V=moe_coarse_1e19
if [ "${SKIP_VALIDATE:-0}" != 1 ]; then
  cp results/phase0/runs/$V/quanteval_b4.log /workspace/rerun-logs/quanteval_b4_july_$V.log 2>/dev/null
  stage $V validate_b4_$V tables_validate_b4.out QUANTEVAL=1 QUANT_BITS=4 QUANT_GROUP=128 EVAL_ITERS=16 GRAIN=1 TOPK=6 TEMPORAL=0 MICRO_BATCH=8 GLOBAL_BATCH=256
  mv -f results/phase0/runs/$V/quanteval_b4.log results/phase0/runs/$V/quanteval_b4_fastcheck.log; cp -f /workspace/rerun-logs/quanteval_b4_july_$V.log results/phase0/runs/$V/quanteval_b4.log 2>/dev/null
  echo "[validate] fast path b4: $(grep -oE 'on test set \| lm loss value: [0-9.E+-]+' results/phase0/runs/$V/quanteval_b4_fastcheck.log | tail -1)  July: 3.137491"
fi
[ -f $O/act_log.pt ] || stage $R act_capture tables_act_$R.out ACTPROBE=1 GRAIN=3 TOPK=18 TEMPORAL=0 MICRO_BATCH=16 GLOBAL_BATCH=1024
for b in 16 8 4 3; do
  grep -q "on test set" $O/quanteval_b$b.log 2>/dev/null && { echo "[skip] quanteval b$b done"; continue; }
  stage $R quanteval_b$b tables_quant_b${b}_$R.out QUANTEVAL=1 QUANT_BITS=$b QUANT_GROUP=128 EVAL_ITERS=16 GRAIN=3 TOPK=18 TEMPORAL=0 MICRO_BATCH=8 GLOBAL_BATCH=256
  grep -oE "\[fakequant\][^|]*|on test set \| lm loss value: [0-9.E+-]+" $O/quanteval_b$b.log | tail -2
done
echo "### tables ALL DONE $(date -u +%H:%M)"
