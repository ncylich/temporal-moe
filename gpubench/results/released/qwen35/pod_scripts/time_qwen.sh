set -x
cd /workspace/rel
bash repo/gpubench/released/tokens.sh /root/m/qwen35-q4km.gguf /workspace/rel/tok_qwen 8 0.4 1024 2048 4096 || exit 1
python3 repo/gpubench/released/run_released.py --bin eng/llama.cpp/build/bin --model /root/m/qwen35-q4km.gguf --nmoe 40 --C 8 --lam 0.4 --tokens /workspace/rel/tok_qwen --ctx 1024 --out /workspace/rel/res/qwen_decode.jsonl --tag R1-q-dec1
echo TIMING_1K_DONE
