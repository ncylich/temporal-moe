#!/usr/bin/env bash
# Start one llama-server for one routing arm, wait until it is healthy.
#   server.sh <arm> <model.gguf> <port> <logfile> [extra env KEY=VAL ...]
# arms: base | ours | skliar (all experts in VRAM) and ours_stream | skliar_stream (K slots + CPU pool).
# Skliar settings come from SK_C and SK_LAM (environment). Prompt batches stay whole: -ub = -b = 4096.
set -euo pipefail
ARM=$1; MODEL=$2; PORT=$3; LOG=$4; shift 4
BIN=${BIN:-/workspace/rel/eng/llama.cpp/build/bin}
NMOE=${NMOE:-40}
declare -a ENVV=()
case $ARM in
  base) ;;
  ours) ENVV+=(LLAMA_TEMPORAL_ROUTE=decode) ;;
  skliar) ENVV+=(LLAMA_TEMPORAL_ROUTE=skliar TEMPORAL_CB_C=${SK_C:?} TEMPORAL_CB_LAMBDA=${SK_LAM:?} TEMPORAL_CB_J=1) ;;
  ours_stream) ENVV+=(LLAMA_TEMPORAL_ROUTE=decode TEMPORAL_UNIFIED=1) ;;
  skliar_stream) ENVV+=(LLAMA_TEMPORAL_ROUTE=skliar TEMPORAL_CB_C=${SK_C:?} TEMPORAL_CB_LAMBDA=${SK_LAM:?} TEMPORAL_CB_J=1 TEMPORAL_UNIFIED=1) ;;
  *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac
EXTRA=()
[[ $ARM == *_stream ]] && EXTRA=(-ncmoe "$NMOE")
env "${ENVV[@]}" "$@" "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -t 8 -c 8192 -np 1 -b 4096 -ub 4096 \
    --jinja --no-cache-prompt --port "$PORT" "${EXTRA[@]}" > "$LOG" 2>&1 &
echo $! > "$LOG.pid"
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "server $ARM up on $PORT (pid $(cat "$LOG.pid"))"; exit 0; fi
  if ! kill -0 "$(cat "$LOG.pid")" 2>/dev/null; then echo "server $ARM died"; tail -30 "$LOG"; exit 1; fi
  sleep 1
done
echo "server $ARM not healthy after 600 s"; tail -30 "$LOG"; exit 1
