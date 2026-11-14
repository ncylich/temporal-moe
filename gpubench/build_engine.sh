#!/usr/bin/env bash
# Build the CUDA engine for the trained-model A6000 runs from the patch files in this repo.
# Usage: build_engine.sh <repo-dir> <work-dir>      -> <work-dir>/llama.cpp/build/bin
set -euo pipefail
REPO="$(cd "${1:?repo dir}" && pwd)"; WORK="${2:?work dir}"
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$WORK"; cd "$WORK"
[ -d llama.cpp ] || git clone -q https://github.com/ncylich/llama.cpp
cd llama.cpp
git fetch -q origin temporal-moe-win
git checkout -q -B cuda-trained 43bec5f3
git -c user.name=build -c user.email=build@localhost am -q "$REPO"/scripts/serving_convert/engine_patch/000*.patch
[ "$(git rev-parse --short=8 'HEAD^{tree}')" = b69151b6 ] || { echo "tree mismatch after CPU patches" >&2; exit 1; }
git apply "$REPO"/llamacpp-bench/systems_bench.patch
git add -A && git -c user.name=build -c user.email=build@localhost commit -q -m "systems bench patch"
git -c user.name=build -c user.email=build@localhost am -q "$REPO"/scripts/serving_convert/engine_patch_cuda/000*.patch
echo "engine tree $(git rev-parse --short=8 'HEAD^{tree}')"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF > build_cmake.log 2>&1
cmake --build build --target llama llama-bench llama-perplexity llama-server llama-quantize -j 48 2>&1 | tail -40
g++ -std=c++17 -O2 "$REPO"/scripts/serving_convert/gguf_logits.cpp -I include -I ggml/include \
    -L build/bin -lllama -Wl,-rpath,"$PWD/build/bin" -o build/bin/gguf_logits
sha256sum build/bin/llama-bench build/bin/llama-perplexity build/bin/libllama.so build/bin/libggml-cuda.so 2>/dev/null
echo BUILD_DONE
