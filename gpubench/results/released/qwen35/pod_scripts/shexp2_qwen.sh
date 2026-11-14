cd /workspace/rel
B=eng/llama.cpp/build/bin; M=/root/m/qwen35-q4km.gguf
R() { python3 repo/gpubench/released/run_released.py --bin $B --model $M --nmoe 40 --C 8 --lam 0.4 --tokens /workspace/rel/tok_qwen --out /workspace/rel/res/qwen_decode_shexp.jsonl --prefix "taskset -c 0-31,64-95" --env TEMPORAL_SHEXP_FIRST=1 "$@" | cut -c1-120; }
R --ctx 1024 --tag R1-q-shexp-p1 --arms ours_nobytes skliar_nobytes ours_nocopy skliar_nocopy
R --ctx 1024 --tag R1-q-shexp-p2 --arms skliar_nocopy ours_nocopy skliar_nobytes ours_nobytes
R --ctx 1024 --tag R1-q-shexp-p3 --arms base ours_vram skliar_vram ours_stream skliar_stream ours_nobytes skliar_nobytes ours_nocopy skliar_nocopy
for c in 2048 4096; do
R --ctx $c --tag R1-q-shexp-p1 --arms base ours_vram skliar_vram ours_stream skliar_stream ours_nobytes skliar_nobytes ours_nocopy skliar_nocopy
R --ctx $c --tag R1-q-shexp-p2 --arms skliar_nocopy ours_nocopy skliar_nobytes ours_nobytes skliar_stream ours_stream skliar_vram ours_vram base
done
echo SHEXP2_DONE
