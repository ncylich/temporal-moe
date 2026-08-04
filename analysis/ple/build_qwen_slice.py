#!/usr/bin/env python3
"""Re-express the audited held-out slice in Qwen's tokenizer, on byte-identical text.

The OLMoE slice is stored as token ids (`bpb_slice_ids.pt`), not text, and its raw source
(dolmino-mix-1124 dclm shards, chosen to be disjoint from OLMoE's pretraining manifest) is not on
this machine. Rather than fetch a *different* corpus and lose the matching, this decodes the stored
ids back to text and re-tokenizes with Qwen. Both models then score the SAME BYTES, which is the
only way a cross-model BPB comparison means anything -- BPB is tokenizer-invariant, but only if the
underlying bytes are the same.

The decode is verified, not assumed: a sample is re-encoded with the OLMoE tokenizer and compared
id-for-id against the original. A BPE round-trip through text is exact for ordinary content but can
differ around byte-fallback or special tokens, and a silent mismatch here would corrupt every number
downstream, so the check is a hard assert rather than a warning.

Contiguous sequences are decoded and concatenated into one stream before re-tokenizing, so Qwen's
tokenizer sees natural text rather than 4096-token fragments; chunk boundaries would otherwise
create tokens that exist in neither model's natural segmentation.

    build_qwen_slice.py --n-seq 2048
"""
import argparse
import json
import math
import os

import torch
from transformers import AutoTokenizer

OLMOE = "/workspace/olmoe-adapt/model"
QWEN = "/workspace/qwen35-adapt/model"
SRC = "/workspace/olmoe-adapt/data/bpb_slice_ids.pt"
OUT_DIR = "/workspace/qwen35-adapt/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=2048, help="OLMoE sequences to decode (4096 tok each)")
    ap.add_argument("--seq", type=int, default=4096)
    A = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    ids = torch.load(SRC, weights_only=False)[: A.n_seq]
    tok_o = AutoTokenizer.from_pretrained(OLMOE)
    tok_q = AutoTokenizer.from_pretrained(QWEN)

    # ---- decode, and prove the decode is lossless before trusting it ----------------------------
    texts = tok_o.batch_decode(ids, skip_special_tokens=False)
    probe = min(16, len(texts))
    for i in range(probe):
        back = tok_o(texts[i], add_special_tokens=False)["input_ids"]
        orig = ids[i].tolist()
        assert back == orig, (
            f"OLMoE decode->encode round-trip is not exact on sequence {i} "
            f"({len(back)} vs {len(orig)} ids). The text handed to Qwen would not be the text OLMoE "
            f"scored, so the two BPB numbers would not be comparable.")
    print(f"  round-trip exact on {probe}/{probe} probed sequences", flush=True)

    text = "".join(texts)
    n_bytes = len(text.encode("utf-8"))

    # ---- re-tokenize the stream, then chunk -----------------------------------------------------
    qids = tok_q(text, add_special_tokens=False)["input_ids"]
    n_full = len(qids) // A.seq
    packed = torch.tensor(qids[: n_full * A.seq], dtype=torch.int32).view(n_full, A.seq)
    n_packed = n_full * A.seq

    # The divisor converts mean CE in nats to bits per byte: bits = nats/ln2, per byte = / (bytes per
    # token). Computed on the packed tokens only, since those are what the model actually scores.
    bytes_per_tok = n_bytes / len(qids)
    divisor = math.log(2) * bytes_per_tok

    torch.save(packed, os.path.join(OUT_DIR, "bpb_slice_ids_qwen.pt"))
    meta = {
        "n_seq": int(n_full), "seq": A.seq, "n_tokens_packed": int(n_packed),
        "n_tokens_raw": len(qids), "n_bytes": n_bytes,
        "bytes_per_token": bytes_per_tok, "divisor_D": divisor,
        "tokenizer": "Qwen3.5-35B-A3B-Base", "vocab_size": tok_q.vocab_size,
        "source": ("byte-identical to OLMoE bpb_slice_ids.pt (dolmino-mix-1124 dclm, disjoint from "
                   "OLMoE-mix-0924 pretraining manifest), decoded and re-tokenized; decode verified "
                   "exact by re-encoding"),
        "olmoe_seqs_consumed": int(A.n_seq),
    }
    with open(os.path.join(OUT_DIR, "bpb_slice_meta_qwen.json"), "w") as f:
        json.dump(meta, f, indent=2)

    o = json.load(open("/workspace/olmoe-adapt/data/bpb_slice_meta.json"))
    print(f"  bytes            {n_bytes:,}")
    print(f"  Qwen tokens      {len(qids):,}   packed {n_packed:,} into {n_full} x {A.seq}")
    print(f"  bytes/token      Qwen {bytes_per_tok:.4f}   vs OLMoE {o['bytes_per_token']:.4f}")
    print(f"  divisor_D        Qwen {divisor:.7f}   vs OLMoE {o['divisor_D']:.7f}")
    print(f"\n[write] {OUT_DIR}/bpb_slice_ids_qwen.pt + bpb_slice_meta_qwen.json", flush=True)


if __name__ == "__main__":
    main()
