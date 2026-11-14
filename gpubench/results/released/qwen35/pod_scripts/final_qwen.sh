cd /workspace/rel
R() { python3 repo/gpubench/released/run_released.py --bin eng/llama.cpp/build/bin --model /root/m/qwen35-q4km.gguf --nmoe 40 --C 8 --lam 0.4 --tokens /workspace/rel/tok_qwen --out /workspace/rel/res/qwen_decode_final.jsonl --prefix "taskset -c 0-31,64-95" "$@" | cut -c1-120; }
A1="base ours_vram skliar_vram ours_stream skliar_stream ours_nobytes skliar_nobytes ours_nocopy skliar_nocopy"
A2="skliar_nocopy ours_nocopy skliar_nobytes ours_nobytes skliar_stream ours_stream skliar_vram ours_vram base"
for p in 1 2 3; do A=$A1; [ $p = 2 ] && A=$A2; R --ctx 1024 --tag R1-q-final-p$p --arms $A; done
echo PASS_1K_DONE
for c in 2048 4096; do for p in 1 2; do A=$A1; [ $p = 2 ] && A=$A2; R --ctx $c --tag R1-q-final-p$p --arms $A; done; done
echo FINAL_DONE
