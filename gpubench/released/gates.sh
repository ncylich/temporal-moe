#!/usr/bin/env bash
# Correctness gates for one released model (run on the pod).
#   gates.sh <model.gguf> <outdir> <E> <C> <lambda> <nmoe> [n_gsm8k]
# 1. routing equality: GSM8K generations with LLAMA_TEMPORAL_ROUTE_DUMP2 (graphs off), all in VRAM and
#    streamed, replayed through decode_state.py / cache_bias.py by route_check.py
# 2. load histograms on n_gsm8k sampled GSM8K generations (all in VRAM; routing does not depend on where
#    the experts live, gate 1 checks the streamed path selects the same experts)
# 3. batch-1 perplexity, streamed against all in VRAM, per routing mode
set -uo pipefail
MODEL=$1; OUT=$2; E=$3; SK_C=$4; SK_LAM=$5; NMOE=$6; NG=${7:-200}
export SK_C SK_LAM NMOE
REPO=/workspace/rel/repo; BIN=/workspace/rel/eng/llama.cpp/build/bin; G=/workspace/rel/data/gsm8k_test.parquet
mkdir -p "$OUT"; cd "$OUT"
stop() { kill -INT "$(cat "$1.pid")" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$(cat "$1.pid")" 2>/dev/null || return 0; sleep 1; done; kill -9 "$(cat "$1.pid")"; }

echo "### gate 1: routing equality $(date -u +%T)"
for arm in ours skliar ours_stream skliar_stream; do
  mode=${arm%_stream}; [ "$mode" = ours ] && mode=decode
  rm -f route_$arm.bin
  bash $REPO/gpubench/released/server.sh $arm "$MODEL" 8091 srv_dump_$arm.log GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_TEMPORAL_ROUTE_DUMP2=$OUT/route_$arm.bin || { echo "GATE1 $arm server failed"; continue; }
  python3 $REPO/gpubench/released/serve_client.py gsm8k --url http://127.0.0.1:8091 --gsm8k $G --tag dump_$arm --out gen_dump_$arm.jsonl --n 3 --max-tokens 400 > /dev/null
  stop srv_dump_$arm.log
  grep "temporal-route2:" srv_dump_$arm.log | tail -1
  python3 $REPO/gpubench/released/route_check.py --dump route_$arm.bin --mode $mode --K 8 --E $E --C $SK_C --lam $SK_LAM --out check_$arm.json
  rm -f route_$arm.bin
done

echo "### gate 2: GSM8K load histograms ($NG items) $(date -u +%T)"
for arm in skliar ours base; do
  bash $REPO/gpubench/released/server.sh $arm "$MODEL" 8092 srv_gsm_$arm.log TEMPORAL_ROUTE_HIST=$OUT/hist_$arm.json || { echo "GATE2 $arm server failed"; continue; }
  python3 $REPO/gpubench/released/serve_client.py gsm8k --url http://127.0.0.1:8092 --gsm8k $G --tag gsm_$arm --out gen_gsm_$arm.jsonl --n $NG > /dev/null
  stop srv_gsm_$arm.log
  grep "temporal-route2:" srv_gsm_$arm.log | tail -1
done

echo "### gate 3: batch-1 perplexity $(date -u +%T)"
python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('$G').to_pydict()
open('ppl.txt','w').write('\n\n'.join(q + '\n' + a for q, a in zip(t['question'], t['answer'])))"
for mode in decode skliar; do
  for where in vram stream; do
    ex=(); [ $where = stream ] && ex=(-ncmoe $NMOE)
    env LLAMA_TEMPORAL_ROUTE=$mode TEMPORAL_CB_C=$SK_C TEMPORAL_CB_LAMBDA=$SK_LAM TEMPORAL_CB_J=1 $( [ $where = stream ] && echo TEMPORAL_UNIFIED=1 ) \
      $BIN/llama-perplexity -m "$MODEL" -f ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa on -t 8 "${ex[@]}" > ppl_${mode}_$where.log 2>&1
    echo "$mode $where: $(grep -E 'Final estimate' ppl_${mode}_$where.log) | $(grep 'temporal-route2:' ppl_${mode}_$where.log | cut -c1-120)"
    grep -oE "\[[0-9]+\][0-9.]+" ppl_${mode}_$where.log | tr '\n' ' '; echo
  done
done
$BIN/llama-perplexity -m "$MODEL" -f ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa on -t 8 > ppl_base_vram.log 2>&1
echo "base vram: $(grep -E 'Final estimate' ppl_base_vram.log)"
echo "### GATES_DONE $(date -u +%T)"
