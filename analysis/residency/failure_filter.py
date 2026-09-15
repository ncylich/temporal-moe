#!/usr/bin/env python3
"""Filter GSM8K generations by outcome across arms and compute mechanical failure stats.

Why. The qwen adapter recovers almost nothing at R8 (+2.1 on GSM8K, ~0 mean) where gemma
recovers +2.3/+3.1. Before changing the method, understand WHAT the constrained qwen model
does wrong on the problems the adapter fails to fix -- truncation, lost final answer,
degenerate repetition, arithmetic slips, or losing the thread -- and whether those are the
same failures the adapter DOES fix elsewhere.

Reads the committed n=1319 sample dumps (they carry the full generated text), the GSM8K
test set for question + gold, and writes:
  <out>/<model>_categories.json      per-doc category + per-arm correctness/tokens
  <out>/<model>_<category>.jsonl     readable subsets: question, gold, every arm's response
  stdout                             category counts + mechanical stats per category/arm

Categories (flexible-extract correctness; bF/b8 = base free/R8, aF/a8 = adapted free/R8):
  damage_unfixed   bF & !b8 & !a8   residency broke it and the adapter did not repair it
  damage_fixed     bF & !b8 &  a8   residency broke it and the adapter repaired it
  adapter_broke     b8 & !a8         the adapter made a working problem fail
  always_right     bF &  b8 &  a8
  always_wrong    !bF & !b8 & !a8
  other            everything else (e.g. base free wrong but R8 right)

    failure_filter.py --model qwen35   # or gemma4
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
SPEC = {
    "qwen35": {"base": "qwen35_think_off_n1319", "adapted": "qwen35_ce_rebuild_n1319",
               "tight": "R8", "loose": "R32", "cap": 2048},
    "gemma4": {"base": "gemma4_instruct_n1319", "adapted": "gemma4_ce_rebuild_n1319",
               "tight": "R8", "loose": "R16", "cap": 2048},
}
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_ANS = re.compile(r"(?:answer is|####|\\boxed\{)\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)


def load_arm(rec, arm):
    """doc_id -> {correct, raw, gen_toks}. Each doc has two rows (strict, flexible);
    keep the flexible verdict (max) and the shared generation."""
    p = f"{SAMP}/{rec}_{arm}_gsm8k_cot_zeroshot.json"
    out = {}
    for i in json.load(open(p))["items"]:
        d = i["doc_id"]
        prev = out.get(d)
        c = bool(i["exact_match"])
        if prev is None or (c and not prev["correct"]):
            out[d] = {"correct": c, "raw": i.get("raw", ""), "gen_toks": i.get("gen_toks", 0)}
    return out


def gold_of(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def num_norm(s):
    return s.replace(",", "").rstrip(".").strip()


def stats_of(raw, gold, cap):
    """Mechanical features of one generation."""
    toks = raw.split()
    n = len(toks)
    grams = [" ".join(toks[i:i + 8]) for i in range(max(0, n - 7))]
    rep = 1 - len(set(grams)) / len(grams) if grams else 0.0
    nums = [num_norm(x) for x in _NUM.findall(raw)]
    m = _ANS.findall(raw)
    stated = num_norm(m[-1]) if m else None
    return {
        "words": n,
        "rep8": round(rep, 3),                        # fraction of repeated 8-grams
        "n_numbers": len(nums),
        "gold_in_text": gold in nums,                  # computed the right number somewhere
        "stated_answer": stated,                       # what an extractor would take
        "stated_is_gold": stated == gold,
        "has_answer_marker": stated is not None,
        "ends_clean": raw.rstrip().endswith((".", "!", "$", ")", "*")) if raw else False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(SPEC), required=True)
    ap.add_argument("--out", default=os.path.join(ABLATIONS, "failure_analysis"))
    ap.add_argument("--adapted", default=None, help="override the adapted record, e.g. qwen35_ce_digit10_n1319")
    A = ap.parse_args()
    S = dict(SPEC[A.model])
    if A.adapted: S["adapted"] = A.adapted
    os.makedirs(A.out, exist_ok=True)

    from datasets import load_dataset
    gsm = list(load_dataset("openai/gsm8k", "main", split="test"))

    arms = {}
    for who in ("base", "adapted"):
        for arm in ("free", S["tight"], S["loose"]):
            arms[(who, arm)] = load_arm(S[who], arm)
    T = S["tight"]
    cats = defaultdict(list)
    rows = {}
    for d in range(len(gsm)):
        bF = arms[("base", "free")][d]["correct"]; b8 = arms[("base", T)][d]["correct"]
        aF = arms[("adapted", "free")][d]["correct"]; a8 = arms[("adapted", T)][d]["correct"]
        if bF and not b8 and not a8: c = "damage_unfixed"
        elif bF and not b8 and a8: c = "damage_fixed"
        elif b8 and not a8: c = "adapter_broke"
        elif bF and b8 and a8: c = "always_right"
        elif not bF and not b8 and not a8: c = "always_wrong"
        else: c = "other"
        cats[c].append(d)
        gold = gold_of(gsm[d]["answer"])
        rows[d] = {"doc_id": d, "category": c, "question": gsm[d]["question"], "gold": gold,
                   "arms": {f"{who}_{arm}": {**arms[(who, arm)][d],
                                             **stats_of(arms[(who, arm)][d]["raw"], gold, S["cap"])}
                            for (who, arm) in arms}}

    print(f"{A.model}: tight arm {T}, n={len(gsm)}\n")
    print(f"{'category':<16}{'n':>6}{'%':>7}")
    for c in ("damage_unfixed", "damage_fixed", "adapter_broke", "always_right", "always_wrong", "other"):
        print(f"{c:<16}{len(cats[c]):>6}{100*len(cats[c])/len(gsm):>7.1f}")
    print()

    # mechanical stats: for the two categories that matter, on the constrained arms
    def agg(docs, key):
        vals = [rows[d]["arms"][key] for d in docs]
        if not vals: return None
        toks = [v["gen_toks"] for v in vals]
        return {"n": len(vals),
                "med_toks": statistics.median(toks),
                "at_cap%": round(100 * sum(t >= S["cap"] - 8 for t in toks) / len(vals), 1),
                "rep8_med": round(statistics.median(v["rep8"] for v in vals), 3),
                "rep8>0.3%": round(100 * sum(v["rep8"] > 0.3 for v in vals) / len(vals), 1),
                "no_marker%": round(100 * sum(not v["has_answer_marker"] for v in vals) / len(vals), 1),
                "gold_in_text%": round(100 * sum(v["gold_in_text"] for v in vals) / len(vals), 1),
                "stated=gold%": round(100 * sum(v["stated_is_gold"] for v in vals) / len(vals), 1)}
    print("MECHANICAL STATS (constrained arms). Read across a row to see how the failing")
    print("generation differs from the same problem's working generation.\n")
    hdr = f"{'category / arm':<34}{'n':>5}{'medtok':>7}{'cap%':>6}{'rep8':>6}{'loop%':>6}{'nomark%':>8}{'goldin%':>8}{'stated=gold%':>13}"
    print(hdr)
    for c in ("damage_unfixed", "damage_fixed", "adapter_broke", "always_right"):
        for key in ("base_free", f"base_{T}", f"adapted_{T}"):
            a = agg(cats[c], key)
            if not a: continue
            print(f"{c+' / '+key:<34}{a['n']:>5}{a['med_toks']:>7.0f}{a['at_cap%']:>6}{a['rep8_med']:>6}"
                  f"{a['rep8>0.3%']:>6}{a['no_marker%']:>8}{a['gold_in_text%']:>8}{a['stated=gold%']:>13}")
        print()

    json.dump({"model": A.model, "tight": T, "categories": {c: cats[c] for c in cats},
               "rows": rows}, open(f"{A.out}/{A.model}_categories.json", "w"))
    for c, docs in cats.items():
        with open(f"{A.out}/{A.model}_{c}.jsonl", "w") as f:
            for d in docs:
                r = rows[d]
                f.write(json.dumps({"doc_id": d, "question": r["question"], "gold": r["gold"],
                                    **{k: {"correct": v["correct"], "gen_toks": v["gen_toks"],
                                           "raw": v["raw"]} for k, v in r["arms"].items()}},
                                   ensure_ascii=False) + "\n")
    print(f"wrote {A.out}/{A.model}_categories.json and per-category jsonl")


if __name__ == "__main__":
    main()
