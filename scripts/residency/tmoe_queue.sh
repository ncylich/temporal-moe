#!/usr/bin/env bash
# Queue control for the GPU lease (scripts/residency/gpu_lease.sh).
#   tmoe_queue.sh list                    tickets in service order, the current holder, pause state
#   tmoe_queue.sh pause | resume          pause: the running job finishes its stage; nothing new starts until resume
#   tmoe_queue.sh kill <ticket|pid>       kill a waiting or running lease entry and its process tree (the ticket disappears)
#   tmoe_queue.sh killchain <pid>         kill a chain script (bash ...chain.sh) and everything under it
#   tmoe_queue.sh add <prio> <log> <cmd>  queue one command under the lease at <prio> (0 = first), output to <log>
#   tmoe_queue.sh hold <prio> <log> <chain.sh args>   queue a whole chain under ONE lease (no interleaving between its stages)
G=${CUDA_VISIBLE_DEVICES:-0}; Q=/var/lock/tmoe_gpu${G}.q; PAUSE=/var/lock/tmoe_gpu${G}.paused; HOLDER=/var/lock/tmoe_gpu${G}.holder
L=/workspace/temporal-moe/scripts/residency/gpu_lease.sh
tree() { local p=$1; echo "$p"; for c in $(pgrep -P "$p"); do tree "$c"; done; }
case "${1:-list}" in
  list)
    [ -e "$PAUSE" ] && echo "PAUSED (tmoe_queue.sh resume to continue)" || echo "running"
    if [ -e "$HOLDER" ]; then read -r t p c < "$HOLDER"; kill -0 "$p" 2>/dev/null && echo "HOLDER  $t pid $p  $(echo "$c" | grep -oE '[a-z_]+\.(py|sh)( [^ ]+){0,3}' | head -1)"; fi
    for f in $(ls "$Q" 2>/dev/null | sort); do p=${f##*-}; kill -0 "$p" 2>/dev/null || continue
      grep -q "^$f " "$HOLDER" 2>/dev/null && continue
      age=$(( ($(date +%s) - ${f:3:10}) / 60 )); echo "WAIT    $f pid $p  ${age}m  $(head -c 200 "$Q/$f" | grep -oE '[a-z_]+\.(py|sh)( [^ ]+){0,3}' | head -1)"; done ;;
  pause) touch "$PAUSE"; echo "paused: the running stage finishes, nothing new starts";;
  resume) rm -f "$PAUSE"; echo "resumed";;
  kill) t=$2; if [ -e "$Q/$t" ]; then p=${t##*-}; else p=$t; fi
    for x in $(tree "$p" | tac); do kill "$x" 2>/dev/null; done; sleep 3; for x in $(tree "$p" | tac); do kill -9 "$x" 2>/dev/null; done; rm -f "$Q/$t" 2>/dev/null; echo "killed $p and its tree";;
  killchain) p=$2; for x in $(tree "$p" | tac); do kill "$x" 2>/dev/null; done; sleep 3; for x in $(tree "$p" | tac); do kill -9 "$x" 2>/dev/null; done; echo "killed chain $p and its tree";;
  add) prio=$2; log=$3; shift 3; TMOE_PRIO=$prio nohup "$L" "$@" > "$log" 2>&1 & echo "queued pid $! at prio $prio -> $log";;
  hold) prio=$2; log=$3; shift 3; TMOE_PRIO=$prio nohup "$L" bash "$@" > "$log" 2>&1 & echo "queued chain pid $! at prio $prio (held lease) -> $log";;
  *) sed -n 2,9p "$0";;
esac
