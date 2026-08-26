#!/usr/bin/env python3
"""Build an ADDITIONAL code lane to test whether code residency damage is fixable by data.

Motivation (2026-08-26): MBPP put base R8 code damage at -14.6 and HumanEval at -5.5,
both far larger than the D7 adapter recovers (+1.2 +/- 2.6 and +1.2 +/- 2.7). The D7 pool
carries 431 code rows out of 8,482 -- 5.1%. The hypothesis is that code is underrepresented,
not unfixable.

This builder produces ONLY the new code prompts. The existing 8,471 D7 trajectories are
reused unchanged and the new rows are appended, so the comparison against the D7 arm is
controlled: same data plus code, not a different pool.

Lineage, unchanged and extended
-------------------------------
Same provenance rule as build_d7_prompts.py -- WildChat-1M and oasst2 only, no
benchmark-derived corpora. The 8-gram screen is EXTENDED to MBPP test here, because MBPP
is now an evaluation surface and a code lane is exactly where MBPP contamination would
enter (screen_mbpp.py found the current pool clean at 12/8471 boilerplate-only hits, but
that pool was not built to be code-heavy).

Prompts already in d7_prompts.jsonl are excluded by source_id and by normalised text, so
the new lane cannot silently duplicate rows the model already trained on.

    build_codelane.py --n 2500 [--scan-cap 400000]
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_d7_prompts import (grams, clean, lane_of, wildchat, oasst2_pairs,  # noqa: E402
                              cached_screen, NGRAM)


def mbpp_screen_grams():
    from datasets import load_dataset
    out = set()
    for split in ("test", "validation", "prompt"):
        try:
            d = load_dataset("google-research-datasets/mbpp", "full", split=split)
        except Exception:
            continue
        for r in d:
            for f in ("text", "code"):
                if r.get(f):
                    out |= grams(r[f])
    print(f"[codelane] MBPP screen: {len(out)} distinct {NGRAM}-grams", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--scan-cap", type=int, default=400000)
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data")
    ap.add_argument("--screen-cache", default="/workspace/olmoe-adapt/data/d7_screen.json")
    A = ap.parse_args()

    existing_ids, existing_norm = set(), set()
    with open(os.path.join(A.out, "d7_prompts.jsonl")) as fh:
        for l in fh:
            r = json.loads(l)
            existing_ids.add(r.get("source_id"))
            existing_norm.add(" ".join(r["text"].lower().split()))
    print(f"[codelane] excluding {len(existing_ids)} prompts already in D7", flush=True)

    screen = cached_screen(A.screen_cache) | mbpp_screen_grams()
    print(f"[codelane] combined screen: {len(screen)} grams "
          f"(four D7 test sets + MBPP)", flush=True)

    kept, seen_norm, scanned, rej = [], set(), 0, {"lane": 0, "dup": 0, "screen": 0}
    for src, sid, text in wildchat(A.scan_cap):
        scanned += 1
        if len(kept) >= A.n:
            break
        if lane_of(text) != "code":
            rej["lane"] += 1
            continue
        norm = " ".join(text.lower().split())
        if sid in existing_ids or norm in existing_norm or norm in seen_norm:
            rej["dup"] += 1
            continue
        if grams(text) & screen:
            rej["screen"] += 1
            continue
        seen_norm.add(norm)
        kept.append({"lane": "codelane", "source": src, "source_id": sid, "text": text})

    if len(kept) < A.n:      # top up from oasst2, same filters
        for src, sid, text in oasst2_pairs():
            if len(kept) >= A.n:
                break
            if lane_of(text) != "code":
                continue
            norm = " ".join(text.lower().split())
            if sid in existing_ids or norm in existing_norm or norm in seen_norm:
                continue
            if grams(text) & screen:
                rej["screen"] += 1
                continue
            seen_norm.add(norm)
            kept.append({"lane": "codelane", "source": src, "source_id": sid, "text": text})

    path = os.path.join(A.out, f"codelane_{len(kept)}.jsonl")
    with open(path, "w") as fh:
        for i, r in enumerate(kept):
            fh.write(json.dumps({"idx": i, **r}, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {"path": path, "sha256": sha, "n": len(kept), "scanned": scanned,
            "rejected": rej,
            "screen": {"n": NGRAM, "test_sets": ["gsm8k", "mmlu", "humaneval",
                                                 "ifeval", "mbpp"]},
            "sources": ["allenai/WildChat-1M (ODC-BY)",
                        "OpenAssistant/oasst2 (Apache-2.0)"],
            "builder": "analysis/residency/build_codelane.py",
            "purpose": "test whether code residency damage is a data-mix problem"}
    json.dump(meta, open(path.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
