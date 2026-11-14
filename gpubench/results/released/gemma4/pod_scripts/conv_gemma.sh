set -e
cd /root/conv
PYTHONPATH=/root/conv/gguf-py python3 /workspace/rel/repo/gpubench/released/convert_rawconfig.py /root/conv/convert_hf_to_gguf.py /root/hf/gemma4 --outtype bf16 --outfile /root/m/gemma4-bf16-fused.gguf
echo CONV_DONE
PYTHONPATH=/root/conv/gguf-py python3 /workspace/rel/repo/gpubench/released/split_gate_up.py /root/m/gemma4-bf16-fused.gguf /root/m/gemma4-bf16.gguf
echo SPLIT_DONE
cd /workspace/rel/eng/llama.cpp
build/bin/llama-quantize /root/m/gemma4-bf16.gguf /root/m/gemma4-q4km.gguf Q4_K_M 16
build/bin/llama-quantize /root/m/gemma4-bf16-fused.gguf /root/m/gemma4-q4km-fused.gguf Q4_K_M 16
echo QUANT_DONE
