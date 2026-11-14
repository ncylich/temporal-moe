#!/usr/bin/env python3
"""Is the D7 training pool clean against MBPP test? Answer before MBPP is used as an eval.

MBPP was never part of the lineage screen -- build_d7_prompts.py screens GSM8K, MMLU,
HumanEval and IFEval only. D7's code lane is Magicoder (OSS-Instruct), which is derived
from public repositories and has documented overlap with MBPP. Adopting MBPP as a code
surface without checking would repeat exactly the mistake the lineage ban exists to
prevent (the Orca-Math episode, where benchmark-family data produced a fake +8 on GSM8K).

Reports, for the D7 trajectory rows actually trained on, how many contain an 8-gram that
also appears in MBPP test -- and prints the worst offenders so a hit can be judged as
boilerplate ("for i in range len") versus real problem/solution content.

    screen_mbpp.py [--traj gemma4_d7_seq4096] [--top 12]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_d7_prompts import grams, NGRAM                            # noqa: E402


def mbpp_grams():
    from datasets import load_dataset
    out = set()
    n = 0
    for split in ("test", "validation", "prompt"):
        try:
            d = load_dataset("google-research-datasets/mbpp", "full", split=split)
        except Exception:
            continue
        for r in d:
            n += 1
            for f in ("text", "code"):
                if r.get(f):
                    out |= grams(r[f])
    print(f"[mbpp] {n} problems, {len(out)} distinct {NGRAM}-grams (text + code)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="gemma4_d7_seq4096")
    ap.add_argument("--top", type=int, default=12)
    A = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    rows = torch.load(f"/workspace/instruct-traj/{A.traj}.pt", weights_only=False)["rows"]
    tok = AutoTokenizer.from_pretrained("/dev/shm/gemma4-26b-it")
    screen = mbpp_grams()

    hits, worst = 0, []
    for i, r in enumerate(rows):
        text = tok.decode(r["ids"], skip_special_tokens=True)
        overlap = grams(text) & screen
        if overlap:
            hits += 1
            worst.append((len(overlap), i, sorted(overlap)[:3]))
    worst.sort(reverse=True)
    print(f"\n[screen] {hits}/{len(rows)} D7 rows contain >=1 MBPP-test 8-gram "
          f"({100*hits/max(1,len(rows)):.2f}%)", flush=True)
    print(f"[screen] worst {min(A.top, len(worst))} rows by overlap count:")
    for c, i, ex in worst[: A.top]:
        print(f"   row {i:5d}  {c:4d} grams  e.g. {ex}")
    if not worst:
        print("   none -- pool is clean against MBPP")


if __name__ == "__main__":
    main()
