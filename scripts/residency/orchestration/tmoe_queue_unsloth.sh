#!/bin/bash
# Fires the unsloth non-determinism diagnosis on the first GPU that goes idle.
while true; do
  for g in 0 1 2 3; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader | tr -dc 0-9)
    if [ "${used:-99999}" -lt 2000 ]; then
      echo "$(date -u +%H:%M) GPU $g free -> running unsloth diagnosis" \
        >> /workspace/rerun-logs/unsloth_diag.log
      cd /workspace/temporal-moe
      CUDA_VISIBLE_DEVICES=$g PATH=/workspace/venv_fla/bin:$PATH \
        TMOE_ROOT=/workspace/temporal-moe \
        LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH \
        /workspace/venv_fla/bin/python -u analysis/residency/diagnose_unsloth_nondet.py \
        >> /workspace/rerun-logs/unsloth_diag.log 2>&1
      exit 0
    fi
  done
  sleep 300
done
