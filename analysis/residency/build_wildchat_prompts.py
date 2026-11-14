#!/usr/bin/env python3
"""Freeze the instruct-program prompt set: 500 WildChat prompts, one artifact for every model.

Filters (in stream order, first 500 kept -- deterministic, no sampling):
    language == English, turn == 1, toxic == False, redacted == False,
    first message role == user, 30 <= len(text) <= 2000 chars (proxy for the <=512-token
    prompt cap enforced per tokenizer at generation time), deduped on conversation_hash
    and on normalized text.

Writes wildchat_prompts_500.jsonl ({idx, conversation_hash, text}) plus a meta json with
the sha256 of the jsonl, so any later regeneration can be checked byte-identical.

    build_wildchat_prompts.py [--n 500] [--out DIR]
"""
import argparse
import hashlib
import json
import os

os.environ.setdefault("HF_TOKEN", open(os.path.expanduser(
    "~/.cache/huggingface/token")).read().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data")
    A = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    kept, seen_hash, seen_text, scanned = [], set(), set(), 0
    for r in ds:
        scanned += 1
        if len(kept) >= A.n:
            break
        if r["language"] != "English" or r["turn"] != 1 or r["toxic"] or r["redacted"]:
            continue
        conv = r["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue
        text = conv[0]["content"].strip()
        if not (30 <= len(text) <= 2000):
            continue
        norm = " ".join(text.lower().split())
        if r["conversation_hash"] in seen_hash or norm in seen_text:
            continue
        seen_hash.add(r["conversation_hash"])
        seen_text.add(norm)
        kept.append({"idx": len(kept), "conversation_hash": r["conversation_hash"],
                     "text": text})

    path = os.path.join(A.out, f"wildchat_prompts_{A.n}.jsonl")
    with open(path, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {"n": len(kept), "scanned": scanned, "dataset": "allenai/WildChat-1M",
            "split": "train (stream order, first-N kept)", "sha256": sha,
            "filters": "English, turn==1, not toxic, not redacted, first role user, "
                       "30-2000 chars, dedup on conversation_hash + normalized text"}
    json.dump(meta, open(os.path.join(A.out, f"wildchat_prompts_{A.n}_meta.json"), "w"),
              indent=1)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
