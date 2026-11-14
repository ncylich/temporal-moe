#!/usr/bin/env bash
# Batch-1 decode speed for the 1e19 fine models on an RTX A6000 (pod layout /workspace/rel/{repo,eng,m}).
#   plain MoE (own gates, GGUF without weights norm): stock top-k; our rule and Skliar (C = 18, lambda*) with
#     a normal top-k prefill and the rule on decoded tokens (LLAMA_TEMPORAL_ROUTE=decode / skliar);
#   trained model: its rule on every token (LLAMA_TEMPORAL_ROUTE=trained).
# R = C = 18 slots per layer; streamed arms keep all routed experts in the CPU pool. Every arm replays the
# same real text: 1024 tokens of a held-out test row as the prompt, its own next 100 tokens decoded.
# -fa 0 (no CUDA flash attention for head size 16), pinned to the GPU's NUMA node, four trials, first dropped.
# Each streamed method is timed at every kernel setting (TEMPORAL_ROUTE_SLOTS 0/1/2 x TEMPORAL_SHEXP_FIRST
# 0/1); the ledger reports all of them and compares each method at its fastest.
#   speed.sh <lambda*>
set -uo pipefail
LAM=$1; O=${OUT:-/workspace/rel/out2}; export SK_C=18 SK_LAM=$LAM NMOE=14 BIN=/workspace/rel/eng/llama.cpp/build/bin
cd /workspace/rel; R=repo/gpubench; mkdir -p $O
MOE=m/moe_fine_1e19-full-q4_0.gguf; TM=m/fine_1e19-q4_0.gguf; T=m/test_tokens_2048.npz
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
echo "### GPU0 CPU affinity: ${CPUS:-unknown} $(date -u +%T)"
stop() { [ -f "$1.pid" ] || return 0; kill -INT "$(cat "$1.pid")" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$(cat "$1.pid")" 2>/dev/null || { rm -f "$1.pid"; return 0; }; sleep 1; done; kill -9 "$(cat "$1.pid")" 2>/dev/null; rm -f "$1.pid"; }

echo "### gate 1: engine routing equals decode_state.py / cache_bias.py $(date -u +%T)"
for S in 0 2; do for arm in ours skliar ours_stream skliar_stream; do
  mode=${arm%_stream}; [ "$mode" = ours ] && mode=decode
  bash $R/released/server.sh $arm $MOE 8091 $O/srv_dump_$arm.log GGML_CUDA_DISABLE_GRAPHS=1 TEMPORAL_ROUTE_SLOTS=$S LLAMA_TEMPORAL_ROUTE_DUMP2=$O/route_$arm.bin > /dev/null || continue
  python3 $R/pretrained/tokens_base.py --url http://127.0.0.1:8091 --tokens $T --row 700 --ctx 512 --tag dump --out $O/dump_$arm > /dev/null
  stop $O/srv_dump_$arm.log
  echo "slots $S $arm: $(python3 $R/released/route_check.py --dump $O/route_$arm.bin --mode $mode --K 18 --E 192 --C 18 --lam $LAM --out $O/check_s${S}_$arm.json | tail -1)"
  rm -f $O/route_$arm.bin
done; done

echo "### gate 3: batch-1 perplexity, streamed (each slot setting) against all in VRAM $(date -u +%T)"
P() { $BIN/llama-perplexity -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8 "$@" 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+"; }
echo "stock: $(P -m $MOE)"
for mode in decode skliar trained; do
  M=$MOE; [ $mode = trained ] && M=$TM
  E="LLAMA_TEMPORAL_ROUTE=$mode TEMPORAL_CB_C=18 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_J=1"
  echo "$mode vram: $(env $E $BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+')"
  for S in 0 1 2; do
    echo "$mode stream slots $S: $(env $E TEMPORAL_UNIFIED=1 TEMPORAL_ROUTE_SLOTS=$S $BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8 -ncmoe 14 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+')"
  done
done

echo "### timing $(date -u +%T)"
for v in 1 2 3 4 5; do
  TD=$O/tok_v$v; ROW=$((600 + v))
  for m in base ours skliar trained; do          # one real continuation, the same file content for every arm
    python3 $R/pretrained/tokens_base.py --tokens $T --row $ROW --ctx 1024 --tag $m --out $TD > /dev/null
  done
  RR() { python3 $R/released/run_released.py --bin $BIN --nmoe 14 --C 18 --lam $LAM --tokens $TD --out $O/speed.jsonl --fa 0 \
         --prefix "taskset -c $CPUS" --ctx 1024 --note "row $ROW" "$@" | cut -c1-150; }
  RR --model $MOE --tag vram-v$v --arms base ours_vram skliar_vram
  RR --model $TM  --tag vram-v$v --arms trained_vram
  for S in 0 1 2; do for X in 0 1; do
    echo "-- slots $S shexp_first $X row $ROW"
    RR --model $MOE --tag s${S}x${X}-v$v --env TEMPORAL_ROUTE_SLOTS=$S TEMPORAL_SHEXP_FIRST=$X --arms ours_stream skliar_stream
    RR --model $TM  --tag s${S}x${X}-v$v --env TEMPORAL_ROUTE_SLOTS=$S TEMPORAL_SHEXP_FIRST=$X --arms trained_stream
  done; done
  for X in 0 1; do      # the kernels patch 0009 replaced, at the copy setting that was fastest for them
    RR --model $MOE --tag k2-x${X}-v$v --env TEMPORAL_ROUTE_SLOTS=0 TEMPORAL_SHEXP_FIRST=$X TEMPORAL_DECODE_K2=1 --arms ours_stream
    RR --model $MOE --tag fullsel-x${X}-v$v --env TEMPORAL_ROUTE_SLOTS=0 TEMPORAL_SHEXP_FIRST=$X TEMPORAL_CB_FULLSEL=1 --arms skliar_stream
  done
  RR --model $MOE --tag k2-v$v --env TEMPORAL_DECODE_K2=1 --arms ours_vram
  RR --model $MOE --tag fullsel-v$v --env TEMPORAL_CB_FULLSEL=1 --arms skliar_vram
done
echo "### SPEED_DONE $(date -u +%T)"
