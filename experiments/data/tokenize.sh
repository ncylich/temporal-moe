#!/bin/bash
# Tokenize the dclm jsonl into Megatron bin/idx using the pythia-12b tokenizer.
# Output: data/dclm_tokenized/dclm_text_document.{bin,idx}
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
IN=${1:-$ROOT/data/dclm_jsonl/dclm.jsonl}
OUTPREFIX=${2:-$ROOT/data/dclm_tokenized/dclm}
WORKERS=${WORKERS:-32}
mkdir -p "$(dirname "$OUTPREFIX")"
export HF_TOKEN=${HF_TOKEN:-}
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export CUDNN_PATH=$NV/cudnn
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd Megatron-LM
$ROOT/.venv/bin/python tools/preprocess_data.py \
  --input "$IN" \
  --output-prefix "$OUTPREFIX" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/pythia-12b \
  --json-keys text \
  --append-eod \
  --workers "$WORKERS"
