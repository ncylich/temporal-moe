#!/bin/bash
# Unattended overnight chaining. Each lane waits for its predecessor's completion marker
# and starts the next stage on a named GPU, so a GPU never sits idle between stages --
# on 2026-08-25 four H200s idled from 03:28 until someone asked, because nothing was
# watching for completion.
#
#     night_chain.sh gemma-merge     # wait for gemma adapter  -> merge + 12-cell grid
#     night_chain.sh qwen-merge      # wait for qwen adapter   -> merge + 12-cell grid
#     night_chain.sh gpu0-think      # wait until GPU 0 is free -> gemma think-on traj
#
# Waits are bounded and loud: if the upstream process dies without its marker, the chain
# says so and exits rather than waiting forever looking healthy.
set -u
ROOT=/workspace/temporal-moe
LOG=/workspace/rerun-logs
cd $ROOT

wait_for () {   # marker file, marker string, process pattern, human label
  local f=$1 marker=$2 pat=$3 label=$4
  echo "[chain] waiting for: $label"
  while ! grep -q "$marker" "$f" 2>/dev/null; do
    if ! ps -eo cmd | grep -q "[${pat:0:1}]${pat:1}"; then
      echo "[chain] ABORT: '$label' process gone and marker '$marker' never appeared"
      return 1
    fi
    sleep 30
  done
  echo "[chain] reached: $label"
}

case "${1:?usage: night_chain.sh gemma-merge|qwen-merge|gpu0-think}" in

  gemma-merge)
    wait_for $LOG/adapt_gemma.out "### gemma ALL DONE" "train_adapters.sh gemma" \
             "gemma adapter trained + mirrored" || exit 1
    GPU=3 exec $ROOT/scripts/residency/merge_and_remeasure.sh gemma ;;

  qwen-merge)
    wait_for $LOG/adapt_qwen.out "### qwen ALL DONE" "train_adapters.sh qwen" \
             "qwen adapter trained + mirrored" || exit 1
    GPU=2 exec $ROOT/scripts/residency/merge_and_remeasure.sh qwen ;;

  # GPU 0 belongs to another agent until 08:30 UTC (01:30 PDT). Wait for the clock AND
  # for the card to actually be empty -- a stated handover time is not proof it is free.
  gpu0-think)
    echo "[chain] GPU 0 handover at 08:30 UTC; polling clock and card"
    while [ "$(date -u +%H%M)" \< "0830" ]; do sleep 300; done
    while [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader | tr -dc 0-9)" -gt 2000 ]; do
      echo "[chain] 08:30 passed but GPU 0 still busy; waiting"
      sleep 300
    done
    echo "[chain] GPU 0 free, starting gemma think-on trajectories"
    GPU=0 exec $ROOT/scripts/residency/regen_trajectories.sh gemma-think ;;

  *) echo "unknown: $1" >&2; exit 2 ;;
esac
