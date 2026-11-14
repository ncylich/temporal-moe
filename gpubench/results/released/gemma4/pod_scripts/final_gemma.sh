cd /workspace/rel
B=eng/llama.cpp/build/bin; M=/root/m/gemma4-q4km.gguf; MF=/root/m/gemma4-q4km-fused.gguf
NMOE=30 bash repo/gpubench/released/tokens.sh $M /workspace/rel/tok_gemma 8 0.5 1024 2048 4096 || exit 1
A1="base ours_vram skliar_vram ours_stream skliar_stream ours_nobytes skliar_nobytes ours_nocopy skliar_nocopy"
A2="skliar_nocopy ours_nocopy skliar_nobytes ours_nobytes skliar_stream ours_stream skliar_vram ours_vram base"
for cfg in default shexp; do
  E=(); OUT=/workspace/rel/res/gemma_decode_final.jsonl; [ $cfg = shexp ] && { E=(--env TEMPORAL_SHEXP_FIRST=1); OUT=/workspace/rel/res/gemma_decode_shexp.jsonl; }
  R() { python3 repo/gpubench/released/run_released.py --bin $B --model $M --nmoe 30 --C 8 --lam 0.5 --tokens /workspace/rel/tok_gemma --out $OUT --prefix "taskset -c 0-31,64-95" "${E[@]}" "$@" | cut -c1-120; }
  for p in 1 2 3; do A=$A1; [ $p = 2 ] && A=$A2; R --ctx 1024 --tag R1-g-$cfg-p$p --arms $A; done
  for c in 2048 4096; do for p in 1 2; do A=$A1; [ $p = 2 ] && A=$A2; R --ctx $c --tag R1-g-$cfg-p$p --arms $A; done; done
  echo "CFG_${cfg}_DONE"
done
# the checkpoint's own fused gate/up layout for the stock baseline (the streamed path needs split tensors)
for cfg in default shexp; do
  E=(); [ $cfg = shexp ] && E=(--env TEMPORAL_SHEXP_FIRST=1)
  for c in 1024 2048 4096; do for p in 1 2 3; do
    python3 repo/gpubench/released/run_released.py --bin $B --model $MF --nmoe 30 --C 8 --lam 0.5 --tokens /workspace/rel/tok_gemma --out /workspace/rel/res/gemma_decode_fusedbase.jsonl --prefix "taskset -c 0-31,64-95" "${E[@]}" --ctx $c --tag R1-g-fused-$cfg-p$p --arms base | cut -c1-120
  done; done
done
echo FINAL_GEMMA_DONE
