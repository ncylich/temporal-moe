#!/usr/bin/env python3
"""Fast dclm tokenizer: HF batch encoding (Rust) + bulk-write Megatron IndexedDatasetBuilder.

Heavy libs (torch/transformers/megatron) and the tokenizer are imported/loaded ONCE in the parent;
multiprocessing.Pool uses fork, so workers inherit them copy-on-write (no per-worker re-import or
re-load — that was the startup bottleneck). Each worker batch-encodes one part and bulk-writes.

Output is byte-identical to tools/preprocess_data.py (--append-eod, uint16). Resumable: skips done.
Usage: fast_tokenize.py [NPROC]
"""
import os, sys, glob, json

NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 24
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # fork-safe; parallelism comes from NPROC
os.environ.setdefault("RAYON_NUM_THREADS", "2")

ROOT = "/workspace/FLAME-MoE"
sys.path.insert(0, f"{ROOT}/Megatron-LM")
# PARTS_GLOB may contain multiple whitespace-separated glob patterns.
_pat = os.environ.get("PARTS_GLOB", f"{ROOT}/data/dclm_parts/part*.jsonl")
PARTS = sorted({p for g in _pat.split() for p in glob.glob(g)})
OUTDIR = os.environ.get("OUT_DIR", f"{ROOT}/data/dclm_tokenized")
os.makedirs(OUTDIR, exist_ok=True)
BATCH = 2000
EOD = int(os.environ.get("EOD", "0"))
TOKENIZER_MODEL = os.environ.get("TOKENIZER_MODEL", "EleutherAI/pythia-12b")

# --- imported ONCE in parent; forked workers inherit ---
import numpy
from transformers import AutoTokenizer
from megatron.core.datasets import indexed_dataset
TOK = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

def tokenize_part(part_path):
    name = os.path.basename(part_path)[:-6]
    prefix = f"{OUTDIR}/{name}_text_document"
    if os.path.exists(prefix + ".idx"):
        return (name, "skip", 0)
    builder = indexed_dataset.IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.uint16)
    n_tok = 0
    batch = []
    def flush(texts):
        nonlocal n_tok
        if not texts:
            return
        enc = TOK(texts, add_special_tokens=False)["input_ids"]
        flat = []
        for ids in enc:
            flat.extend(ids); flat.append(EOD)
            builder.sequence_lengths.append(len(ids) + 1)
            builder.document_indices.append(len(builder.sequence_lengths))
            n_tok += len(ids) + 1
        arr = numpy.fromiter(flat, dtype=numpy.uint16, count=len(flat))
        builder.data_file.write(arr.tobytes(order="C"))
    with open(part_path) as f:
        for line in f:
            try:
                t = json.loads(line).get("text")
            except Exception:
                continue
            if t:
                batch.append(t)
            if len(batch) >= BATCH:
                flush(batch); batch = []
    flush(batch)
    builder.finalize(prefix + ".idx")
    return (name, "done", n_tok)

def main():
    import multiprocessing as mp
    todo = [p for p in PARTS if not os.path.exists(f"{OUTDIR}/{os.path.basename(p)[:-6]}_text_document.idx")]
    print(f"{len(PARTS)} parts, {len(todo)} to do, NPROC={NPROC}", flush=True)
    with mp.Pool(NPROC) as pool:   # fork: inherits TOK + imports
        for name, status, ntok in pool.imap_unordered(tokenize_part, todo):
            print(f"{name}: {status} tokens={ntok}", flush=True)
    total = sum(os.path.getsize(f"{OUTDIR}/{os.path.basename(p)[:-6]}_text_document.bin") // 2
                for p in PARTS if os.path.exists(f"{OUTDIR}/{os.path.basename(p)[:-6]}_text_document.bin"))
    print(f"TOTAL tokens (uint16): {total/1e9:.3f}B", flush=True)

if __name__ == "__main__":
    main()
