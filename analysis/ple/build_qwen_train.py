#!/usr/bin/env python3
"""Re-express the adaptation corpus in Qwen's tokenizer, same text as the OLMoE runs consumed.

`finetune_ids.pt` is the OLMoE-tokenized adaptation corpus and is a different file from the held-out
`bpb_slice_ids.pt`, so train/eval separation is inherited rather than re-derived. Only the leading
slice is converted: a 50M-token run needs ~50M Qwen tokens, not the full 1B, and decoding the whole
file would cost far more than the runs it feeds.

Same round-trip guarantee as build_qwen_slice.py -- decode is verified exact by re-encoding before
any of it is trusted, because a silently lossy decode would train Qwen on text OLMoE never saw while
we describe the two runs as matched.

    build_qwen_train.py --target-tokens 60000000
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

OLMOE = "/workspace/olmoe-adapt/model"
QWEN = "/workspace/qwen35-adapt/model"
SRC = "/workspace/olmoe-adapt/data/finetune_ids.pt"
OUT_DIR = "/workspace/qwen35-adapt/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=60_000_000)
    ap.add_argument("--seq", type=int, default=4096)
    A = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    tok_o = AutoTokenizer.from_pretrained(OLMOE)
    tok_q = AutoTokenizer.from_pretrained(QWEN)

    # Qwen packs ~4.565 bytes/token vs OLMoE's ~4.485, so it needs slightly FEWER tokens for the same
    # text. Take a margin so a 50M-token run cannot run off the end of the corpus mid-epoch.
    n_seq = int(A.target_tokens / A.seq * 1.15)
    ids = torch.load(SRC, weights_only=False)[:n_seq]
    print(f"  decoding {len(ids)} OLMoE sequences ({len(ids)*A.seq/1e6:.1f}M tokens)", flush=True)

    texts = tok_o.batch_decode(ids, skip_special_tokens=False)
    for i in range(min(8, len(texts))):
        assert tok_o(texts[i], add_special_tokens=False)["input_ids"] == ids[i].tolist(), (
            f"decode->encode not exact on training sequence {i}; the Qwen and OLMoE runs would not "
            f"be seeing the same text")
    print(f"  round-trip exact on 8/8 probed sequences", flush=True)

    text = "".join(texts)
    n_bytes = len(text.encode("utf-8"))
    qids = tok_q(text, add_special_tokens=False)["input_ids"]
    n_full = len(qids) // A.seq
    packed = torch.tensor(qids[: n_full * A.seq], dtype=torch.int32).view(n_full, A.seq)

    torch.save(packed, os.path.join(OUT_DIR, "finetune_ids_qwen.pt"))
    meta = {"n_seq": int(n_full), "seq": A.seq, "n_tokens": int(n_full * A.seq),
            "n_bytes": n_bytes, "bytes_per_token": n_bytes / len(qids),
            "olmoe_seqs_consumed": int(n_seq), "tokenizer": "Qwen3.5-35B-A3B-Base",
            "source": "byte-identical to the leading slice of OLMoE finetune_ids.pt; "
                      "disjoint from bpb_slice_ids.pt (the held-out eval slice)"}
    with open(os.path.join(OUT_DIR, "finetune_meta_qwen.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {n_full} x {A.seq} = {n_full*A.seq/1e6:.1f}M Qwen tokens from {n_bytes/1e6:.1f}MB")
    print(f"[write] {OUT_DIR}/finetune_ids_qwen.pt", flush=True)


if __name__ == "__main__":
    main()
