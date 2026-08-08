#!/usr/bin/env python3
"""Stage 1: build a disjoint, in-distribution held-out BPB slice for OLMoE residency adaptation.

Source: allenai/dolmino-mix-1124 (the 0125 ANNEAL parent pool) — a DISTINCT dataset from the
pretraining mix allenai/OLMoE-mix-0924 (2903 shards, all treated as seen and excluded), and the
0125 run only annealed on a SUBSAMPLE of dolmino, so these shards are in-distribution and largely
unseen. We stream a fixed high-index shard range (disjoint-by-index heuristic on top of the
dataset-identity exclusion) and n-gram dedup against the lm-eval task data downstream.

Tokenize with the OLMoE tokenizer, pack at 4096. Record actual byte count so the CE->BPB divisor
D = ln(2) * bytes/token is byte-derived (never inherited). Writes:
  data/bpb_slice_ids.pt   (int32 token ids, [n_seq, 4096])
  data/bpb_slice_meta.json (n_tokens, n_bytes, divisor, source shards)
"""
import os, sys, json, math, torch
from transformers import AutoTokenizer
from datasets import load_dataset

TARGET_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
SEQ = 4096
OUT = "/workspace/olmoe-adapt/data"
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained("/workspace/olmoe-adapt/model")

# stream a fixed dolmino-dclm shard range disjoint-by-index; dclm = the general web component
ds = load_dataset("allenai/dolmino-mix-1124", data_files="data/dclm/024*/*.json.zst",
                  split="train", streaming=True)
buf_ids, nbytes, ntok, shards = [], 0, 0, set()
EOS = tok.eos_token_id
for ex in ds:
    txt = ex.get("text", "")
    if not txt:
        continue
    nbytes += len(txt.encode("utf-8"))
    ids = tok(txt, add_special_tokens=False).input_ids + [EOS]
    buf_ids.extend(ids); ntok += len(ids)
    if ntok >= TARGET_TOK:
        break
nseq = ntok // SEQ
arr = torch.tensor(buf_ids[:nseq * SEQ], dtype=torch.int32).view(nseq, SEQ)
D = math.log(2) * (nbytes * (nseq * SEQ) / ntok) / (nseq * SEQ)   # = ln2 * bytes/token (packed)
D = math.log(2) * (nbytes / ntok)
torch.save(arr, f"{OUT}/bpb_slice_ids.pt")
json.dump({"n_seq": nseq, "seq": SEQ, "n_tokens_packed": nseq * SEQ, "n_tokens_raw": ntok,
           "n_bytes": nbytes, "bytes_per_token": nbytes / ntok, "divisor_D": D,
           "source": "allenai/dolmino-mix-1124 data/dclm/024*/*.json.zst (anneal parent, disjoint "
                     "from OLMoE-mix-0924 pretraining manifest)", "tokenizer": "OLMoE"},
          open(f"{OUT}/bpb_slice_meta.json", "w"), indent=1)
print(f"[bpb-slice] {nseq} x {SEQ} = {nseq*SEQ} packed tokens ({ntok} raw), {nbytes} bytes, "
      f"bytes/tok={nbytes/ntok:.3f}, divisor D={D:.4f}")
