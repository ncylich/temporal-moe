#!/usr/bin/env bash
# Hold-the-device gate for one GPU, ORDERED.
#   gpu_lease.sh <cmd...>            run <cmd> once the GPU is ours and >=120 GB are free
# Ordering: waiters take a ticket <prio>-<ns-timestamp>-<pid> in /var/lock/tmoe_gpu<G>.q and
# the GPU goes to the smallest live ticket: strict FIFO within a priority, lower TMOE_PRIO
# first (default 5; use 1 for "next", 9 for "whenever"). Dead waiters' tickets are pruned.
# Chain-level hold: run a whole chain under the lease (gpu_lease.sh bash chain.sh); inner
# gpu_lease.sh calls see TMOE_LEASE_HELD and exec directly, so no other job can interleave
# between the chain's stages. The flock is kept for mutual exclusion with older waiters.
set -uo pipefail
G=${CUDA_VISIBLE_DEVICES:-0}
if [ "${TMOE_LEASE_HELD:-}" = "$G" ]; then
  exec "$@"
fi
LOCK=/var/lock/tmoe_gpu${G}.lease; Q=/var/lock/tmoe_gpu${G}.q
mkdir -p /var/lock "$Q"
PRIO=${TMOE_PRIO:-5}
TICKET="$Q/$(printf '%02d' "$PRIO")-$(date +%s%N)-$$"
: > "$TICKET"
trap 'rm -f "$TICKET"' EXIT
exec {fd}>"$LOCK"
while :; do
  first=$(for f in "$Q"/*; do [ -e "$f" ] || continue; p=${f##*-}; if kill -0 "$p" 2>/dev/null; then echo "$f"; else rm -f "$f"; fi; done | sort | head -1)
  if [ "$first" = "$TICKET" ] && flock -n "$fd"; then break; fi
  sleep 3
done
for i in $(seq 1 90); do
  [ "${TMOE_LEASE_NOMEM:-}" = 1 ] && { free=999999; break; }     # tests: skip the memory wait
  # host RAM guard: the container limit is ~251 GB and /dev/shm counts; a model load or a
  # sleeping vLLM engine needs ~60-110 GB, so require 140 GB of headroom before starting
  lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  if [ -n "$lim" ] && [ "$lim" != max ] && [ -n "$cur" ] && [ $(( (lim - cur) / 1073741824 )) -lt 140 ]; then
    [ $((i % 6)) -eq 1 ] && echo "[lease] waiting for host RAM: $(( (lim - cur) / 1073741824 )) GB free of $((lim / 1073741824))"
    sleep 20; continue
  fi
  free=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  [ "${free:-0}" -ge 122880 ] && break
  sleep 20
done
echo "[lease] GPU $G acquired (${free:-?}MiB free, prio $PRIO) $(date -u +%H:%M)"
export TMOE_LEASE_HELD=$G
"$@"; rc=$?
echo "[lease] GPU $G released rc=$rc $(date -u +%H:%M)"
exit $rc
