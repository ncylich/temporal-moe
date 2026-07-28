#!/bin/bash
# Tokenize the dclm jsonl into Megatron bin/idx using the pythia-12b tokenizer.
# Output: data/dclm_tokenized/dclm_text_document.{bin,idx}
set -euo pipefail
cd "$(dirname "$0")/../.."
IN=${1:-$ROOT/data/dclm_jsonl/dclm.jsonl}
OUTPREFIX=${2:-$ROOT/data/dclm_tokenized/dclm}
WORKERS=${WORKERS:-32}
mkdir -p "$(dirname "$OUTPREFIX")"
export HF_TOKEN=${HF_TOKEN:-}
cd Megatron-LM
"$PY" tools/preprocess_data.py \
  --input "$IN" \
  --output-prefix "$OUTPREFIX" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/pythia-12b \
  --json-keys text \
  --append-eod \
  --workers "$WORKERS"
