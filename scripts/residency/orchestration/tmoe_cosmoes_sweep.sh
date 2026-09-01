#!/usr/bin/env bash
# CoSMoEs BlES baseline on the phase0 isoFLOP venue (BASELINE_METHODS_COMPARISON #1).
# Waits for the rebuilt tok16k corpus and the transformer_engine install, then runs, serial
# under the GPU lease, all at s0@1e16 with the locked phase0 HPs (lr 3e-3, wu 0.05, gb 256):
#   references on the SAME corpus (old-frontier BPBs are not comparable across a rebuilt
#   tokenizer): vanilla MoE and temporal MoE, at GRAIN=1 and GRAIN=3;
#   BlES sweep: COSMOES_LAMBDA in {0.1, 1, 10, 100} x GRAIN in {1, 3}, patched router at R=E
#   (their setting: no residency mask; the paper gives no lambda, so log-spaced coverage).
# Achieved swaps/token-layer come from the run logs; the comparison point is (R needed, BPB).
set -uo pipefail; cd /workspace/temporal-moe
until grep -q "### tok16k rebuild DONE" /workspace/rerun-logs/tok16k_rebuild.out 2>/dev/null; do sleep 300; done
until (cd /workspace/temporal-moe && .venv/bin/python -c "import torch, transformer_engine.pytorch") 2>/dev/null; do sleep 300; done
. scripts/env.sh
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7568 EVAL_AT_END=1 CUDA_VISIBLE_DEVICES=0 GPU=0 TMOE_PRIO=5
# TE 2.16 (the only cu12+torch2.8 build NVIDIA ships) fails Megatron's fused-rope probe; fusion
# off for EVERY cell, so all references and BlES cells stay internally consistent.
export EXTRA_ARGS="--no-rope-fusion"
L=scripts/residency/gpu_lease.sh
run_one () { # NAME GRAIN TEMPORAL COSMOES_LAMBDA RESIDENCY_R(optional, "E" = patched router unconstrained)
  local tag=iter_done_marker; local d=$ROOT/results/phase0/runs/$1
  read _N ITERS < <("$PY" analysis/shapes.py iters s0 1e16 256)
  [ -d "$d/ckpt/$(printf iter_%07d $ITERS)" ] && { echo "[cosmoes] SKIP $1"; return 0; }
  echo "### cosmoes cell $1 (grain=$2 temporal=$3 lambda=$4) $(date -u +%H:%M)"
  GRAIN=$2 TEMPORAL=$3 TEMPORAL_EVICT=min_logit COSMOES_LAMBDA=$4 TEMPORAL_RESIDENCY_R=${5:-0} \
    SHAPE=s0 TARGET_FLOPS=1e16 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=256 SEED=1234 \
    RUN_NAME=$1 $L bash experiments/run.sh
  "$PY" analysis/parse_run.py "$d" 2>/dev/null | grep '^SUMMARY' || echo "[cosmoes] $1 parse failed"
}
for G in 1 3; do
  run_one g${G}_ref_moe_1e16    $G 0 0
  run_one g${G}_ref_tmoe_1e16   $G 1 0
  # BlES cells run the PATCHED router at R=E (unconstrained, mask all-true: the documented
  # "vanilla full MoE" mode) so the COSMOES_LAMBDA hook is live -- TEMPORAL=0 would route through
  # stock pretrain_gpt.py and silently train with no BlES loss.
  RE=$((64*G))  # router accepts numeric R only; R=num_experts == unconstrained
  for LAM in 0.1 1 10 100; do run_one g${G}_bles${LAM}_1e16 $G 1 $LAM $RE; done
done
echo "### cosmoes sweep ALL DONE $(date -u +%H:%M)"
