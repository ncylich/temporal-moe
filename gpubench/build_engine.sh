#!/usr/bin/env bash
# Build the CUDA engine for the trained-model A6000 runs from the patch files in this repo.
# Usage: build_engine.sh <repo-dir> <work-dir>      -> <work-dir>/llama.cpp/build/bin
set -euo pipefail
REPO="$(cd "${1:?repo dir}" && pwd)"; WORK="${2:?work dir}"
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$WORK"; cd "$WORK"
if [ -n "${ENGINE_BASE_TGZ:-}" ]; then
    # pod without a route to GitHub: the base commit as a `git archive 43bec5f3` tarball (tree 56fc59a1)
    if [ ! -d llama.cpp ]; then
        mkdir llama.cpp; tar xzf "$ENGINE_BASE_TGZ" -C llama.cpp
        git -C llama.cpp init -q; git -C llama.cpp add -A -f
        git -C llama.cpp -c user.name=build -c user.email=build@localhost commit -q -m "base 43bec5f3"; git -C llama.cpp tag base
    fi
    cd llama.cpp
    [ "$(git rev-parse --short=8 'base^{tree}')" = 56fc59a1 ] || { echo "base tree mismatch" >&2; exit 1; }
    git checkout -q -f -B cuda-trained base
else
[ -d llama.cpp ] || git clone -q https://github.com/ncylich/llama.cpp
cd llama.cpp
git fetch -q origin temporal-moe-win
git checkout -q -B cuda-trained 43bec5f3
fi
git -c user.name=build -c user.email=build@localhost am -q "$REPO"/scripts/serving_convert/engine_patch/000*.patch
[ "$(git rev-parse --short=8 'HEAD^{tree}')" = b69151b6 ] || { echo "tree mismatch after CPU patches" >&2; exit 1; }
git apply "$REPO"/llamacpp-bench/systems_bench.patch
git add -A && git -c user.name=build -c user.email=build@localhost commit -q -m "systems bench patch"
git -c user.name=build -c user.email=build@localhost am -q "$REPO"/scripts/serving_convert/engine_patch_cuda/00*.patch
echo "engine tree $(git rev-parse --short=8 'HEAD^{tree}')"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF > build_cmake.log 2>&1
cmake --build build --target llama llama-bench llama-perplexity llama-server llama-quantize -j 48 2>&1 | tail -40
g++ -std=c++17 -O2 "$REPO"/scripts/serving_convert/gguf_logits.cpp -I include -I ggml/include \
    -L build/bin -lllama -Wl,-rpath,"$PWD/build/bin" -o build/bin/gguf_logits
sha256sum build/bin/llama-bench build/bin/llama-perplexity build/bin/libllama.so build/bin/libggml-cuda.so 2>/dev/null
echo BUILD_DONE
