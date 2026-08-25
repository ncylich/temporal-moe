#!/bin/bash
# Block until a GPU is genuinely free, then print its index. Never returns GPU 0, which is
# reserved for other users on this shared box.
#
# Exists because "the previous job printed ALL DONE" is not the same as "the GPU is free":
# a vLLM engine takes time to release, and two launches this session died with
# "Free memory on device (10.54/139.81 GiB) ... less than desired GPU memory utilization".
# Checking once is not enough either -- require the card to look free on consecutive polls.
#
#     g=$(wait_for_gpu.sh) && CUDA_VISIBLE_DEVICES=$g python ...
#     wait_for_gpu.sh 2      # wait for one SPECIFIC gpu
NEED_GB=${NEED_GB:-100}
WANT=${1:-}
deadline=$(( $(date +%s) + ${TIMEOUT:-3600} ))
declare -A ok
while [ "$(date +%s)" -lt "$deadline" ]; do
  for g in ${WANT:-1 2 3}; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -dc 0-9)
    tot=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader 2>/dev/null | tr -dc 0-9)
    free_gb=$(( (${tot:-0} - ${used:-0}) / 1024 ))
    if [ "$free_gb" -ge "$NEED_GB" ]; then
      ok[$g]=$(( ${ok[$g]:-0} + 1 ))
      # two consecutive clean polls: a releasing engine can look free for an instant
      if [ "${ok[$g]}" -ge 2 ]; then echo "$g"; exit 0; fi
    else
      ok[$g]=0
    fi
  done
  sleep 15
done
echo "TIMEOUT waiting for a free GPU" >&2
exit 1
