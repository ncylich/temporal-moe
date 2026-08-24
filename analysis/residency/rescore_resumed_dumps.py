#!/usr/bin/env python3
"""Repair resumed dumps whose per-item `pass` is stale.

`resume_truncated.py` wrote its merged dump BEFORE scoring it for a while, so
every `*_cap8k_*` dump written in that window carried the pass value inherited
from the truncated original while its CSV row carried the correct re-scored one.
Per-item `pass` feeds every flip and wrongness analysis, so the dumps were
silently wrong where it mattered most.

Scoring is pure CPU over the saved text, so this repairs them in place with no
regeneration: re-extract, re-run the sandboxed scorer, rewrite `pass` and
`unfinished`, and report any cell whose pass@1 moves (all of them should, by
exactly the amount the resume recovered).

    rescore_resumed_dumps.py [--write]      (dry by default)
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
from resume_truncated import score_items                             # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
KIND = {"humaneval_gemma_fixed": "humaneval_gemma",
        "humaneval_gptoss": "humaneval_gptoss",
        "humaneval_think": "humaneval_think"}


def main():
    write = "--write" in sys.argv
    print(f"{'dump':52s} {'was':>7} {'now':>7} {'unfin':>6}")
    for p in sorted(glob.glob(f"{SAMP}/*_cap8k_*.json")):
        base = os.path.basename(p)[:-5]
        m = re.match(r"(.+)_(free|R\d+)_(.+)$", base)
        if not m or m.group(3) not in KIND:
            continue          # mmlu_dual cells are scored by their own harness
        blob = json.load(open(p))
        items = blob["items"]
        was = sum(1 for i in items if i.get("pass")) / len(items)
        cap = max(i["gen_toks"] for i in items)
        cap = 8192 if cap > 8192 - 8 else 8192      # resumed cells all target 8192
        p1, unf = score_items(items, KIND[m.group(3)], cap)
        print(f"{base:52s} {was:7.4f} {p1:7.4f} {unf:6d}"
              + ("  (stale)" if abs(p1 - was) > 1e-9 else ""))
        if write:
            json.dump(blob, open(p, "w"))
    print("\nrewrote in place" if write else "\nDRY: pass --write to apply")


if __name__ == "__main__":
    main()
