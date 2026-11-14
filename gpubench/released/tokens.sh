#!/usr/bin/env bash
# Real-text token files for the timing runs (run on the pod).
#   tokens.sh <model.gguf> <outdir> <C> <lambda> [ctx ...]
# Contexts are built once with the stock server; then each routing method (base, ours, skliar, all experts in
# VRAM) writes its own greedy continuation of every context: <outdir>/ctx<N>_<method>.{ctx,gen,txt}.
set -euo pipefail
MODEL=$1; OUT=$2; export SK_C=$3 SK_LAM=$4; shift 4
CTX=("${@:-1024}"); [ $# -eq 0 ] && CTX=(1024 2048 4096)
REPO=/workspace/rel/repo; G=/workspace/rel/data/gsm8k_test.parquet
mkdir -p "$OUT"
stop() { kill -INT "$(cat "$1.pid")" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$(cat "$1.pid")" 2>/dev/null || return 0; sleep 1; done; kill -9 "$(cat "$1.pid")"; }
for m in base ours skliar; do
  bash $REPO/gpubench/released/server.sh $m "$MODEL" 8093 "$OUT/srv_$m.log"
  [ $m = base ] && python3 $REPO/gpubench/released/serve_client.py contexts --url http://127.0.0.1:8093 --gsm8k $G --ctx "${CTX[@]}" --out "$OUT/prompts.json"
  python3 $REPO/gpubench/released/serve_client.py generate --url http://127.0.0.1:8093 --prompts "$OUT/prompts.json" --tag $m --out "$OUT" --n-gen 100
  stop "$OUT/srv_$m.log"
done
echo TOKENS_DONE
