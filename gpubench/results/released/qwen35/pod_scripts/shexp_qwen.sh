cd /workspace/rel
B=eng/llama.cpp/build/bin; M=/root/m/qwen35-q4km.gguf; G=gates_qwen
for v in 0 1; do
  TEMPORAL_SHEXP_FIRST=$v LLAMA_TEMPORAL_ROUTE=decode TEMPORAL_UNIFIED=1 taskset -c 0-31,64-95 $B/llama-perplexity -m $M -f $G/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa on -t 8 -ncmoe 40 > logs/ppl_shexp${v}_ours_stream.log 2>&1
  echo "shexp_first=$v ours streamed: $(grep 'Final estimate' logs/ppl_shexp${v}_ours_stream.log)"
done
TEMPORAL_SHEXP_FIRST=1 $B/llama-perplexity -m $M -f $G/ppl.txt -c 1024 --chunks 4 -b 1 -ub 1 -ngl 99 -fa on -t 8 > logs/ppl_shexp1_base.log 2>&1
echo "shexp_first=1 base: $(grep 'Final estimate' logs/ppl_shexp1_base.log)"
R() { python3 repo/gpubench/released/run_released.py --bin $B --model $M --nmoe 40 --C 8 --lam 0.4 --tokens /workspace/rel/tok_qwen --out /workspace/rel/res/qwen_decode_shexp.jsonl --prefix "taskset -c 0-31,64-95" --env TEMPORAL_SHEXP_FIRST=1 "$@" | cut -c1-120; }
R --ctx 1024 --tag R1-q-shexp-p1 --arms base ours_vram skliar_vram ours_stream skliar_stream
R --ctx 1024 --tag R1-q-shexp-p2 --arms skliar_stream ours_stream skliar_vram ours_vram base
echo SHEXP_DONE
