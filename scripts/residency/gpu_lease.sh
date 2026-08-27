#!/usr/bin/env bash
# Hold the GPU for the duration of a command. Replaces wait_gpu_free.sh, whose check-then-run
# had a window: chain A's next stage passed the "free" check between chain B's train
# finishing and B's merge starting, then B's merge took the device and A died in vLLM's
# memory check (2026-08-27, 05:46). A poll cannot reserve; a lock can.
#
#     gpu_lease.sh <cmd...>
# Blocks on an exclusive flock, THEN waits for memory to actually be released (CUDA frees
# asynchronously after the previous holder exits), then runs the command holding the lock.
set -uo pipefail
G=${CUDA_VISIBLE_DEVICES:-0}
LOCK=/var/lock/tmoe_gpu${G}.lease
mkdir -p /var/lock
exec {fd}>"$LOCK"
flock "$fd"
for i in $(seq 1 90); do
  free=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  [ "${free:-0}" -ge 122880 ] && break
  sleep 20
done
echo "[lease] GPU $G acquired (${free:-?}MiB free) $(date -u +%H:%M)"
"$@"; rc=$?
echo "[lease] GPU $G released rc=$rc $(date -u +%H:%M)"
exit $rc
