#!/usr/bin/env python3
"""Download dclm-baseline parquet shards and write a combined JSONL ({"text": ...} per line).

Usage: build_jsonl.py <start_shard> <num_shards> <out.jsonl>
Each shard ~144MB parquet / 61k docs / ~65M tokens. Resumable: skips if out exists & non-empty.
"""
import sys, json, os
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "mlfoundations/dclm-baseline-1.0-parquet"
# global-shard_01_of_10 / local-shard_0_of_10 holds shard_00000000..N
PREFIX = ("filtered/OH_eli5_vs_rw_v2_bigram_200k_train/"
          "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/"
          "global-shard_01_of_10/local-shard_0_of_10/")

def shard_path(i):
    return f"{PREFIX}shard_{i:08d}_processed.parquet"

def main():
    start, num, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    raw_dir = "data/dclm_raw"
    n_docs = 0
    with open(out, "w") as fout:
        for i in range(start, start + num):
            try:
                p = hf_hub_download(REPO, shard_path(i), repo_type="dataset", local_dir=raw_dir)
            except Exception as e:
                print(f"shard {i}: download failed {e}", flush=True)
                continue
            t = pq.read_table(p, columns=["text"])
            for v in t.column("text"):
                s = v.as_py()
                if s:
                    fout.write(json.dumps({"text": s}) + "\n")
                    n_docs += 1
            os.remove(p)  # free disk; keep only jsonl
            print(f"shard {i}: cumulative docs={n_docs}", flush=True)
    print(f"DONE {out}: {n_docs} docs", flush=True)

if __name__ == "__main__":
    main()
