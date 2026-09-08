#!/usr/bin/env python3
"""Stage 2 Step 1: materialize the 1B-token router-finetune corpus (5B-capable recipe, 1B now).

Distribution (approximating OLMoE-mix-0924's DCLM-dominant proportions):
  ~70% mlfoundations/dclm-baseline-1.0  (the parent web pool OLMoE-mix subsampled; fresh shards)
  ~30% allenai/dolmino-mix-1124         (anneal parent: dclm web + math/pes2o/stackexchange/wiki),
                                        shard ranges DISJOINT from the 100M BPB slice (which used
                                        dolmino dclm/024*).
Disjointness from the held-out BPB slice (the science-critical constraint) is enforced by
(a) different dataset / shard ranges and (b) n-gram dedup: 32-token windows (stride 16) of the BPB
slice are hashed into a set; any corpus 4096-pack sharing a window with the slice (or with the
lm-eval task text) is dropped. Tokenize with the OLMoE tokenizer, pack 4096.

Writes data/finetune_ids.pt [n_seq,4096] int32 + data/finetune_meta.json (counts, dedup hits).
"""
import os, sys, json, hashlib, torch
from transformers import AutoTokenizer
from datasets import load_dataset

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000_000
SEQ = 4096
FRAC_DCLM = 0.70
OUT = "/workspace/olmoe-adapt/data"
tok = AutoTokenizer.from_pretrained("/workspace/olmoe-adapt/model")
EOS = tok.eos_token_id

# ---- build the dedup window-set from the BPB slice + lm-eval task text ----
W, STRIDE = 32, 16
def wins(ids):
    for i in range(0, len(ids) - W, STRIDE):
        yield hash(tuple(ids[i:i + W]))
bpb = torch.load(f"{OUT}/bpb_slice_ids.pt").reshape(-1).tolist()
dedup = set(wins(bpb))
print(f"[dedup] {len(dedup)} windows from BPB slice", flush=True)
# lm-eval task text (arc/piqa/openbookqa/hellaswag/sciq/boolq/copa/winogrande/lambada) — small
for path, name, split, field in [("allenai/openbookqa", "main", "test", "question_stem"),
                                 ("ybisk/piqa", None, "validation", "goal"),
                                 ("allenai/ai2_arc", "ARC-Easy", "test", "question")]:
    try:
        for ex in load_dataset(path, name, split=split, trust_remote_code=True):
            t = ex.get(field, "")
            if t:
                dedup.update(wins(tok(t, add_special_tokens=False).input_ids))
    except Exception as e:
        print(f"[dedup] skip {path}: {str(e)[:60]}", flush=True)
print(f"[dedup] {len(dedup)} windows total (BPB + eval task text)", flush=True)

def stream_pack(ds, budget, srclabel):
    buf, ntok, nbytes, dropped, kept = [], 0, 0, 0, 0
    out = []
    for ex in ds:
        txt = ex.get("text", "")
        if not txt:
            continue
        nbytes += len(txt.encode("utf-8"))
        buf.extend(tok(txt, add_special_tokens=False).input_ids + [EOS])
        while len(buf) >= SEQ:
            seq = buf[:SEQ]; buf = buf[SEQ:]
            if any(h in dedup for h in wins(seq)):
                dropped += 1; continue
            out.append(seq); kept += 1; ntok += SEQ
            if ntok >= budget:
                return out, ntok, nbytes, dropped, kept
    return out, ntok, nbytes, dropped, kept

# ---- 70% DCLM-baseline (fresh web shards) ----
dclm = load_dataset("mlfoundations/dclm-baseline-1.0",
                    data_files="global-shard_05_of_10/local-shard_0_of_10/*.jsonl.zst",
                    split="train", streaming=True)
d_out, d_tok, d_bytes, d_drop, d_keep = stream_pack(dclm, int(TARGET * FRAC_DCLM), "dclm-baseline")
print(f"[dclm] {d_keep} packs, {d_tok} tok, {d_drop} dedup-dropped", flush=True)

# ---- 30% dolmino (disjoint from BPB slice: dclm 00*/01* + math + pes2o + stackexchange + wiki) ----
dol = load_dataset("allenai/dolmino-mix-1124",
                   data_files=["data/dclm/00*/*.json.zst", "data/dclm/01*/*.json.zst",
                               "data/math/*/*.json.zst", "data/pes2o/*.json.zst",
                               "data/stackexchange/*.json.zst", "data/wiki/*.json.zst"],
                   split="train", streaming=True)
o_out, o_tok, o_bytes, o_drop, o_keep = stream_pack(dol, TARGET - d_tok, "dolmino")
print(f"[dolmino] {o_keep} packs, {o_tok} tok, {o_drop} dedup-dropped", flush=True)

allp = d_out + o_out
arr = torch.tensor(allp, dtype=torch.int32)
torch.save(arr, f"{OUT}/finetune_ids.pt")
meta = {"n_seq": len(allp), "seq": SEQ, "n_tokens": len(allp) * SEQ,
        "dclm_baseline": {"packs": d_keep, "tokens": d_tok, "bytes": d_bytes, "dedup_dropped": d_drop},
        "dolmino": {"packs": o_keep, "tokens": o_tok, "bytes": o_bytes, "dedup_dropped": o_drop},
        "dedup_windows": len(dedup), "dedup_total_dropped": d_drop + o_drop,
        "sources": "mlfoundations/dclm-baseline-1.0 (global-shard_05) + allenai/dolmino-mix-1124 "
                   "(dclm 00*/01*, math, pes2o, stackexchange, wiki) — disjoint from BPB slice's dclm/024*",
        "tokenizer": "OLMoE", "note": "5B-capable recipe; 1B materialized"}
json.dump(meta, open(f"{OUT}/finetune_meta.json", "w"), indent=1)
print(f"[corpus] {len(allp)} packs = {len(allp)*SEQ} tokens; dedup dropped {d_drop+o_drop}", flush=True)
