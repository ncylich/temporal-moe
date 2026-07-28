#!/usr/bin/env python3
"""Download dclm-baseline parquet shards into data/dclm_parts/partNN.jsonl ({"text":...} per line).

Groups SHARDS_PER_PART consecutive shards into one part file (matches the original layout: part00+01
~= 500k docs for tokenizer training). Resumable: skips part files that already exist & are non-empty.
Parquet is deleted right after read to keep disk small.

Usage: download_parts.py <first_part> <last_part_inclusive> [shards_per_part=4]
"""
import sys, os, json
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor

REPO = "mlfoundations/dclm-baseline-1.0-parquet"
PREFIX = ("filtered/OH_eli5_vs_rw_v2_bigram_200k_train/"
          "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/"
          "global-shard_01_of_10/local-shard_0_of_10/")
ROOT = os.environ.get("TMOE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
PARTS_DIR = f"{ROOT}/data/dclm_parts"
RAW_DIR = f"{ROOT}/data/dclm_raw"

def shard_path(i):
    return f"{PREFIX}shard_{i:08d}_processed.parquet"

def fetch_shard(i):
    p = hf_hub_download(REPO, shard_path(i), repo_type="dataset", local_dir=RAW_DIR)
    t = pq.read_table(p, columns=["text"])
    texts = [v.as_py() for v in t.column("text")]
    os.remove(p)
    return texts

def build_part(part_idx, spp):
    out = f"{PARTS_DIR}/part{part_idx:02d}.jsonl"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"part{part_idx:02d}: skip (exists)", flush=True)
        return
    first = part_idx * spp
    shards = list(range(first, first + spp))
    n_docs = 0
    tmp = out + ".tmp"
    with open(tmp, "w") as fout, ThreadPoolExecutor(max_workers=spp) as ex:
        for texts in ex.map(fetch_shard, shards):
            for s in texts:
                if s:
                    fout.write(json.dumps({"text": s}) + "\n"); n_docs += 1
    os.rename(tmp, out)
    print(f"part{part_idx:02d}: docs={n_docs} shards={shards}", flush=True)

def main():
    first, last = int(sys.argv[1]), int(sys.argv[2])
    spp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(PARTS_DIR, exist_ok=True); os.makedirs(RAW_DIR, exist_ok=True)
    for p in range(first, last + 1):
        build_part(p, spp)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
