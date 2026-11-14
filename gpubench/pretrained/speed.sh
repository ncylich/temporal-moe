#!/usr/bin/env bash
# Batch-1 decode speed for the 1e19 fine models on an RTX A6000 (run on the pod, layout as the released runs:
# /workspace/rel/{repo,eng,m}). Plain MoE (moe_fine) under stock top-k, our rule (decode) and Skliar
# (C = 18, lambda*); the temporally trained model (fine) under its own rule. R = C = 18 slots per layer,
# all 14 layers' routed experts in the CPU pool when streamed. Every arm: TEMPORAL_SHEXP_FIRST=1, pinned to
# the GPU's NUMA node, real text (test-split rows, own greedy continuation), four trials, first discarded.
#   speed.sh <lambda*>
set -uo pipefail
LAM=$1; O=${OUT:-/workspace/rel/out}; export SK_C=18 SK_LAM=$LAM NMOE=14 BIN=/workspace/rel/eng/llama.cpp/build/bin
cd /workspace/rel; R=repo/gpubench; mkdir -p $O
MOE=${MOE_GGUF:-m/moe_fine_1e19-q4_0.gguf}; TM=m/fine_1e19-q4_0.gguf; T=m/test_tokens_2048.npz
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
echo "### GPU0 CPU affinity: ${CPUS:-unknown} $(date -u +%T)"
stop() { [ -f "$1.pid" ] || return 0; kill -INT "$(cat "$1.pid")" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$(cat "$1.pid")" 2>/dev/null || { rm -f "$1.pid"; return 0; }; sleep 1; done; kill -9 "$(cat "$1.pid")" 2>/dev/null; rm -f "$1.pid"; }

if [ "${SKIP_GATES:-0}" != 1 ]; then
echo "### gate 1: engine routing equals decode_state.py / cache_bias.py $(date -u +%T)"
for arm in ours skliar ours_stream skliar_stream; do
  mode=${arm%_stream}; [ "$mode" = ours ] && mode=decode
  bash $R/released/server.sh $arm $MOE 8091 $O/srv_dump_$arm.log GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_TEMPORAL_ROUTE_DUMP2=$O/route_$arm.bin || continue
  python3 $R/pretrained/tokens_base.py --url http://127.0.0.1:8091 --tokens $T --row 700 --ctx 512 --tag dump --out $O/dump_$arm > /dev/null
  stop $O/srv_dump_$arm.log
  python3 $R/released/route_check.py --dump $O/route_$arm.bin --mode $mode --K 18 --E 192 --C 18 --lam $LAM --out $O/check_$arm.json | tail -2
  rm -f $O/route_$arm.bin
done

echo "### gate 3: batch-1 perplexity, streamed against all in VRAM $(date -u +%T)"
for mode in decode skliar trained; do
  M=$MOE; [ $mode = trained ] && M=$TM
  for where in vram stream; do
    ex=(); [ $where = stream ] && ex=(-ncmoe 14)
    env LLAMA_TEMPORAL_ROUTE=$mode TEMPORAL_CB_C=18 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_J=1 $( [ $where = stream ] && echo TEMPORAL_UNIFIED=1 ) \
      $BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa on -t 8 "${ex[@]}" > $O/ppl_${mode}_$where.log 2>&1
    echo "$mode $where: $(grep -E 'Final estimate' $O/ppl_${mode}_$where.log)"
  done
done

fi
echo "### timing $(date -u +%T)"
for v in 1 2 3 4 5; do
  TD=$O/tok_v$v; ROW=$((600 + v)); CTX=1024     # the model was trained at 2048 positions; longer prompts are refused
  for m in base ours skliar trained; do
    M=$MOE; [ $m = trained ] && M=$TM
    [ -s $TD/ctx1024_$m.gen ] && continue
    if [ "${REAL_TEXT:-0}" = 1 ]; then                    # same real continuation for every arm
      python3 $R/pretrained/tokens_base.py --tokens $T --row $ROW --ctx $CTX --tag $m --out $TD; continue
    fi
    bash $R/released/server.sh $m $M 8093 $O/srv_tok_$m.log || continue
    python3 $R/pretrained/tokens_base.py --url http://127.0.0.1:8093 --tokens $T --row $ROW --ctx $CTX --tag $m --out $TD
    stop $O/srv_tok_$m.log
  done
  RR() { python3 $R/released/run_released.py --bin $BIN --nmoe 14 --C 18 --lam $LAM --tokens $TD --out $O/${SPEED_OUT:-speed.jsonl} --fa 0 \
         --prefix "taskset -c $CPUS" --env TEMPORAL_SHEXP_FIRST=1 --note "row $ROW" "$@" | cut -c1-150; }
  ARMS="base ours_stream skliar_stream"; [ $v = 1 ] && ARMS="base ours_vram skliar_vram ours_stream skliar_stream ours_nobytes skliar_nobytes"
  RR --model $MOE --tag pre-v$v --ctx $CTX --arms $ARMS
  RR --model $TM --tag pre-v$v --ctx $CTX --arms trained_vram trained_stream
done
echo "### SPEED_DONE $(date -u +%T)"
