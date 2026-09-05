#!/usr/bin/env python3
"""Render an MBPP dump (genbench_samples/<tag>_<arm>_<task>.json) as readable markdown for review:
decoded prompt, raw generation, extracted code, pass/fail, thinking tokens, cap/unfinished flags,
plus a header of summary counts. Meant for a reviewer to check coherence and parsing by eye.

    mbpp_review_dump.py <dump.json> --tokenizer <model dir> [--n 12] [--maxchars 2500] [--out review.md]
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--maxchars", type=int, default=2500)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
    d = json.load(open(A.dump))
    items = d["items"] if isinstance(d, dict) else d
    n = len(items)
    passed = sum(1 for it in items if str(it.get("pass")) == "True")
    unf = sum(1 for it in items if str(it.get("unfinished")) == "True")
    cap = sum(1 for it in items if str(it.get("hit_cap")) == "True")
    gt = [int(it.get("gen_toks", 0)) for it in items]
    th = [int(it.get("think_toks", 0)) for it in items]
    lines = [f"# {os.path.basename(A.dump)}", "",
             f"items {n}, pass {passed} ({passed / max(n, 1):.3f}), unfinished {unf}, at cap {cap}, "
             f"gen tokens median {sorted(gt)[n // 2] if n else 0} max {max(gt) if gt else 0}, "
             f"thinking tokens median {sorted(th)[n // 2] if n else 0} max {max(th) if th else 0}", ""]
    for it in items[: A.n]:
        pid = it.get("prompt_ids")
        if isinstance(pid, str):
            pid = json.loads(pid)
        prompt = tk.decode(pid, skip_special_tokens=False) if pid else "(no prompt ids)"
        raw = it.get("raw", "")
        lines += [f"## {it.get('doc')}  pass={it.get('pass')} gen_toks={it.get('gen_toks')} "
                  f"think_toks={it.get('think_toks')} unfinished={it.get('unfinished')} hit_cap={it.get('hit_cap')}",
                  "", "### prompt (decoded, special tokens kept)", "```", prompt[-A.maxchars:], "```", "",
                  "### raw generation", "```", raw[: A.maxchars] + ("\n... [truncated]" if len(raw) > A.maxchars else ""),
                  "```", "", "### extracted code (what was executed)", "```python",
                  str(it.get("extracted", "(producer did not store extracted code)"))[: A.maxchars], "```", ""]
    text = "\n".join(lines)
    if A.out:
        open(A.out, "w").write(text); print(f"wrote {A.out} ({n} items, {min(n, A.n)} shown)")
    else:
        print(text)


if __name__ == "__main__":
    main()
