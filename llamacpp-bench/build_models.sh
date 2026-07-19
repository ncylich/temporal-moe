#!/usr/bin/env bash
# Reconstruct both random-weight Qwen3-MoE GGUF benchmark models (fine + coarse), Q4_K_M.
# Usage: build_models.sh <llama.cpp-dir> [out-root]
# Produces: <out-root>/qwen3moe-rand-<fine|coarse>-Q4_K_M.gguf
set -euo pipefail
LLAMA_DIR="${1:?usage: build_models.sh <llama.cpp-dir> [out-root]}"
OUT_ROOT="${2:-/workspace/models}"
HERE="$(cd "$(dirname "$0")" && pwd)"
QUANT="$LLAMA_DIR/build/bin/llama-quantize"

python3 "$HERE/gen_random_qwen3moe.py" --variant both --out-root "$OUT_ROOT"

for var in fine coarse; do
  HF="$OUT_ROOT/qwen3moe-rand-$var"
  F16="$OUT_ROOT/qwen3moe-rand-$var-f16.gguf"
  Q4="$OUT_ROOT/qwen3moe-rand-$var-Q4_K_M.gguf"
  echo "[$var] convert HF -> f16 gguf"
  python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF" --outfile "$F16" --outtype f16
  echo "[$var] quantize f16 -> Q4_K_M"
  "$QUANT" "$F16" "$Q4" Q4_K_M
  rm -f "$F16"   # keep only the quantized model; f16 is large and disposable
  ls -la "$Q4"
done
echo "MODELS_DONE"
