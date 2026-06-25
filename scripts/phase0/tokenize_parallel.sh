#!/bin/bash
# Tokenize all data/dclm_parts/part*.jsonl in parallel (one preprocess_data.py per file),
# producing data/dclm_tokenized/part*_text_document.{bin,idx}. FLAME-style multi-bin corpus.
# Usage: tokenize_parallel.sh [WORKERS_PER_JOB]   (default 2; 32 files x 2 = 64-way)
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
WPJ=${1:-2}
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export PATH=$ROOT/.venv/bin:$PATH CUDNN_PATH=$NV/cudnn \
  LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} HF_TOKEN=${HF_TOKEN:-}
mkdir -p data/dclm_tokenized
LOGD=$ROOT/results/phase0/tok_logs; mkdir -p "$LOGD"
S=$(date +%s)
pids=()
for f in data/dclm_parts/part*.jsonl; do
  name=$(basename "$f" .jsonl)
  out=$ROOT/data/dclm_tokenized/$name
  # skip if already done
  if [ -f "${out}_text_document.idx" ]; then echo "skip $name (done)"; continue; fi
  ( cd Megatron-LM && python tools/preprocess_data.py --input "$ROOT/$f" \
      --output-prefix "$out" --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model EleutherAI/pythia-12b --json-keys text --append-eod \
      --workers "$WPJ" > "$LOGD/$name.log" 2>&1 ) &
  pids+=($!)
done
echo "launched ${#pids[@]} tokenize jobs (workers=$WPJ each)"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE elapsed=$(($(date +%s)-S))s failures=$fail"
ls data/dclm_tokenized/part*_text_document.idx 2>/dev/null | wc -l
