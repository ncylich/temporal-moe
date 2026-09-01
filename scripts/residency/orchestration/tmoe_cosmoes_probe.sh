#!/usr/bin/env bash
# After the CoSMoEs sweep: router-probe each trained cell (one fixed batch, logs raw routing),
# then compute fair-usage metrics (effective experts, over-use, neglect, switching, union).
set -uo pipefail; cd /workspace/temporal-moe
. scripts/env.sh
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7568 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=5 EXTRA_ARGS="--no-rope-fusion"
L=scripts/residency/gpu_lease.sh
cells="g1_ref_moe_1e16:1:0:0 g1_ref_tmoe_1e16:1:1:0 g1_bles0.1_1e16:1:1:64 g1_bles1_1e16:1:1:64 g1_bles10_1e16:1:1:64 g1_bles100_1e16:1:1:64 g3_ref_moe_1e16:3:0:0 g3_ref_tmoe_1e16:3:1:0 g3_bles0.1_1e16:3:1:192 g3_bles1_1e16:3:1:192 g3_bles10_1e16:3:1:192 g3_bles100_1e16:3:1:192"
# wait until every cell has its final checkpoint
for c in $cells; do IFS=: read RN G TP RR <<<"$c"
  read _N IT < <(GRAIN=$G "$PY" analysis/shapes.py iters s0 1e16 256)
  until [ -d "results/phase0/runs/$RN/ckpt/$(printf iter_%07d $IT)" ]; do sleep 600; done
done
echo "### cosmoes probe: all 12 ckpts present $(date -u +%H:%M)"
for c in $cells; do IFS=: read RN G TP RR <<<"$c"
  [ -f "results/phase0/runs/$RN/router_log.pt" ] || {
    echo "### probe $RN $(date -u +%H:%M)"
    PROBE=1 GRAIN=$G TEMPORAL=$TP TEMPORAL_EVICT=min_logit TEMPORAL_RESIDENCY_R=$RR \
      SHAPE=s0 TARGET_FLOPS=1e16 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=256 SEED=1234 \
      RUN_NAME=$RN $L bash experiments/run.sh; }
  "$PY" analysis/probes/cosmoes_metrics.py "$RN" || echo "[metrics] $RN FAILED"
done
echo "### cosmoes probe ALL DONE $(date -u +%H:%M)"
