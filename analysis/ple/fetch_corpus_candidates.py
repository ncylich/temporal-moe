#!/usr/bin/env python3
"""Fetch candidate adaptation corpora and tokenize them for both Qwen models.

Qwen3's own pretraining data is proprietary (36T tokens, web + PDF-extracted documents + synthetic
textbooks from Qwen2.5-Math/Coder, annotated for educational value -- arXiv 2505.09388), so the best
available proxy is a public corpus built on the same principle, chosen by measured BPB rather than by
argument. Candidates:

    fineweb-edu   web filtered by an educational-value classifier; closest public analogue to Qwen's
                  own educational-value annotation axis, but discards ~90% of raw data
    dclm          high-quality filtered CommonCrawl
    nemotron-cc   ensembles the FineWeb-Edu and DCLM classifiers and adds synthetic rephrasing,
                  which parallels Qwen's synthetic component without the aggressive filtering

The incumbent is the OLMoE adaptation corpus (OLMo's Dolma mixture), used because it matched OLMoE,
not Qwen. Its cost is visible: OLMoE's unconstrained null degraded 0.0224 BPB over 50M tokens of it,
so ~20% of what we attribute to the residency constraint is the corpus and recipe.

Only tokenization happens here -- BPB scoring needs the GPU and the GPU is training. Byte counts are
matched across candidates so BPB is comparable: BPB = CE_nats / (ln2 * bytes_per_token), and an
unmatched byte count changes the divisor rather than the model's actual surprise.

    fetch_corpus_candidates.py --mb-per-corpus 40
"""
import argparse
import json
import os

CANDIDATES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "dclm":        ("mlfoundations/dclm-baseline-1.0-parquet", None, "text"),
    "nemotron-cc": ("nvidia/Nemotron-CC-v2", None, "text"),
}
TOKENIZERS = {"qwen3": "/dev/shm/qwen3-30b", "qwen": "/workspace/qwen35-adapt/model"}
OUT = "/workspace/corpus_candidates"


def fetch(name, target_bytes):
    from datasets import load_dataset
    repo, cfg, col = CANDIDATES[name]
    ds = load_dataset(repo, cfg, split="train", streaming=True)
    parts, n = [], 0
    for row in ds:
        t = row.get(col) or ""
        if not t:
            continue
        parts.append(t)
        n += len(t.encode("utf-8"))
        if n >= target_bytes:
            break
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb-per-corpus", type=int, default=40)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--only", default="")
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import torch
    from transformers import AutoTokenizer

    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZERS.items()}
    names = [A.only] if A.only else list(CANDIDATES)
    target = A.mb_per_corpus * 1_000_000
    for name in names:
        try:
            text = fetch(name, target)
        except Exception as e:
            print(f"  {name:14} FETCH FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        # Truncate to exactly the target so every candidate is scored on the same byte count.
        b = text.encode("utf-8")[:target]
        text = b.decode("utf-8", errors="ignore")
        nb = len(text.encode("utf-8"))
        for tk, tokzr in toks.items():
            ids = tokzr(text, add_special_tokens=False)["input_ids"]
            nfull = len(ids) // A.seq
            packed = torch.tensor(ids[: nfull * A.seq], dtype=torch.int32).view(nfull, A.seq)
            torch.save(packed, f"{OUT}/{name}_{tk}.pt")
            meta = {"corpus": name, "tokenizer": tk, "n_seq": nfull, "seq": A.seq,
                    "n_bytes": nb, "bytes_per_token": nb / len(ids),
                    "divisor_D": 0.6931471805599453 * (nb / len(ids))}
            json.dump(meta, open(f"{OUT}/{name}_{tk}.json", "w"), indent=2)
            print(f"  {name:14} {tk:6} {nfull:5} x {A.seq}  {nb/1e6:.1f}MB  "
                  f"{nb/len(ids):.4f} B/tok  D={meta['divisor_D']:.7f}", flush=True)
    print("=== CORPUS FETCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
