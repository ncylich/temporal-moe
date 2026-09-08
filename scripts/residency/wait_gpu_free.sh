#!/usr/bin/env bash
# Block until the GPU has actually released its memory.
#
# A multi-stage chain (merge -> eval, or eval -> eval) starts its next stage the instant
# the previous python process returns, but CUDA memory is freed asynchronously: the next
# vLLM boot then dies in the memory-utilisation check with "Free memory on device cuda:0
# (73/139 GiB) is less than desired GPU memory utilization". That is what killed the qwen
# MMLU chain's own eval on 2026-08-27. The multi-GPU queue runner had this gate; the
# hand-written chains did not.
#
#     wait_gpu_free.sh [free_gb] [timeout_s]
# Note nvidia-smi honours CUDA_VISIBLE_DEVICES for -i, so the probe unsets it.
set -uo pipefail
NEED=${1:-120}; TIMEOUT=${2:-1800}; G=${CUDA_VISIBLE_DEVICES:-0}
start=$(date +%s)
while :; do
  free=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -i "$G" \
           --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  [ -n "${free:-}" ] && [ "$free" -ge $((NEED*1024)) ] && { echo "[gpu] ${free}MiB free, proceeding"; exit 0; }
  [ $(( $(date +%s) - start )) -ge "$TIMEOUT" ] && { echo "[gpu] TIMEOUT: only ${free:-?}MiB free" >&2; exit 1; }
  sleep 20
done
