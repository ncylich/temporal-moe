#!/bin/bash
# Standing queue runner: keeps a GPU busy forever. One runner per GPU, each pulls the next
# job file from /workspace/tmoe_queue atomically and executes it. Hand-queueing left GPUs
# idle every time a job finished between my check-ins; this does not.
set -u
G=$1
Q=/workspace/tmoe_queue
mkdir -p $Q/pids; echo $$ > $Q/pids/runner$G.pid
trap "rm -f $Q/pids/runner$G.pid" EXIT
export CUDA_VISIBLE_DEVICES=$G
while :; do
  JOB=""
  for f in $(ls $Q/*.job 2>/dev/null | sort); do
    if mv "$f" "$f.running.$G" 2>/dev/null; then JOB="$f.running.$G"; break; fi
  done
  if [ -z "$JOB" ]; then sleep 20; continue; fi
  # wait until THIS gpu is actually free -- the earlier hand-queued jobs still hold some,
  # and starting on an occupied device just dies in vLLM's memory-utilisation check
  while :; do
    # nvidia-smi honours CUDA_VISIBLE_DEVICES for -i, so querying with it set indexes the
    # VISIBLE set and reads the wrong device (or errors). Unset it for the probe only.
    used=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -i $G \
             --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ "${used:-999999}" -lt 2000 ] && break
    sleep 30
  done
  LOG=/workspace/rerun-logs/q_$(basename "$JOB" .job.running.$G).out
  echo "### runner GPU $G start $(date -u +%H:%M)" > "$LOG"
  bash "$JOB" >> "$LOG" 2>&1
  RC=$?
  echo "### runner GPU $G done rc=$RC $(date -u +%H:%M)" >> "$LOG"
  # A job that died because the device was busy is not a failed experiment -- older
  # hand-queued runs still hold GPUs, and a lost job silently shrinks the batch. Requeue
  # it (up to 3 attempts) instead of marking it done.
  if [ $RC -ne 0 ] && grep -q "less than desired GPU memory utilization" "$LOG" 2>/dev/null; then
    BASE=$(basename "$JOB" .job.running.$G)
    N=$(echo "$BASE" | grep -oE "retry[0-9]+$" | tr -dc 0-9); N=${N:-0}
    if [ "$N" -lt 3 ]; then
      NEW=$Q/$(echo "$BASE" | sed -E "s/retry[0-9]+$//")retry$((N+1)).job
      mv "$JOB" "$NEW" 2>/dev/null
      echo "### requeued as $(basename $NEW) (device was busy)" >> "$LOG"
      sleep 60
      continue
    fi
  fi
  mv "$JOB" "$JOB.done" 2>/dev/null
done
