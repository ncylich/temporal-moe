#!/usr/bin/env bash
# The iteration loop on the 1.26 B models (trained model and plain MoE, Q4_0), P1-9 protocol through
# released/run_released.py: real text (rows 601.. of the test tokens), 1,024-token prompt as one batch, decode one
# token at a time, four trials with the first dropped. Arms: ours on the plain MoE, Skliar on the plain MoE at
# matched loads, the trained model; streamed (18 slots) unless VRAM=1.
#   fast_small.sh <tag>     env: FUSED (default 1), X="extra env", ROWS="1 2", VRAM=1, REV=1
# gates:  fast_small.sh gates   prints batch-1 perplexity (4 x 1,024 tokens) of each model, all in VRAM with the
#         stock fused op and nothing else, then streamed with FUSED and X. They must agree in every digit.
set -uo pipefail
cd /workspace/rel; R=repo/gpubench; BIN=eng/llama.cpp/build/bin; O=ov/small; mkdir -p $O
MOE=m/moe_fine_1e19-full-q4_0.gguf; TM=m/fine_1e19-q4_0.gguf; T=m/test_tokens_2048.npz; LAM=${LAM:-0.739453125}
CPUS=$(nvidia-smi topo -m | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+-[0-9]+(,[0-9]+-[0-9]+)*$/){print $i; exit}}')
TAG=${1:-try}; FUSED=${FUSED:-1}
if [ "$TAG" = gates ]; then
  PPL() { env "$@" > $O/last.log 2>&1; echo "$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/last.log || echo FAILED) $(grep -ohE 'temporal-qkv[^|]*|loads_per_layer_token=[0-9.]+|swaps_per_layer_token=[0-9.]+' $O/last.log | head -2 | tr '\n' ' ')"; }
  for mode in decode skliar trained; do
    M=$MOE; [ $mode = trained ] && M=$TM
    E="LLAMA_TEMPORAL_ROUTE=$mode TEMPORAL_CB_C=18 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_J=1"
    C="taskset -c $CPUS $BIN/llama-perplexity -m $M -f m/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa off -t 8"
    echo "$mode vram, patch 0014 op only:  $(PPL $E TEMPORAL_FUSED=1 $C)"
    echo "$mode vram, FUSED=$FUSED ${X:-}:  $(PPL $E TEMPORAL_FUSED=$FUSED ${X:-} $C)"
    for rep in 1 2; do echo "$mode stream, FUSED=$FUSED ${X:-} rep $rep:  $(PPL $E TEMPORAL_UNIFIED=1 TEMPORAL_FUSED=$FUSED ${X:-} $C -ncmoe 14)"; done
  done
  echo GATES_DONE; exit 0
fi
for v in ${ROWS:-1 2}; do [ -d $O/tok_v$v ] || for m in base ours skliar trained; do
  python3 $R/pretrained/tokens_base.py --tokens $T --row $((600 + v)) --ctx 1024 --tag $m --out $O/tok_v$v > /dev/null; done; done
SUF=stream; [ "${VRAM:-0}" = 1 ] && SUF=vram
L="ours|$MOE|ours skliar|$MOE|skliar trained|$TM|trained"; [ "${REV:-0}" = 1 ] && L=$(echo $L | tr ' ' '\n' | tac | tr '\n' ' ')
for a in $L; do IFS='|' read -r n M arm <<< "$a"
  for v in ${ROWS:-1 2}; do
    python3 $R/released/run_released.py --bin $BIN --nmoe 14 --C 18 --lam $LAM --tokens $O/tok_v$v --out $O/$TAG.jsonl --fa 0 \
       --prefix "taskset -c $CPUS" --ctx 1024 --model $M --tag $n --note "$TAG row $((600 + v))" --env TEMPORAL_SHEXP_FIRST=1 TEMPORAL_FUSED=$FUSED ${X:-} --arms ${arm}_$SUF > /dev/null
  done
done
python3 - $O/$TAG.jsonl "$TAG" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; out = []
for n in ("ours", "skliar", "trained"):
    v = [r["tok_s"] if "tok_s" in r else r.get("tps") for r in rows if r.get("tag") == n]
    v = [x for x in v if x]
    if v: out.append(f"{n} {sum(v)/len(v):.1f} tok/s ({1000*len(v)/sum(v):.4f} ms, n={len(v)})")
print(sys.argv[2], " | ".join(out), flush=True)
PY
