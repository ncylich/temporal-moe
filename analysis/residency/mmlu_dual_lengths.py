#!/usr/bin/env python3
"""Recover per-item generation lengths from the committed mmlu_dual dumps.

The 228-item genbench_samples/*_mmlu_dual.json files (think-off MMLU diagnosis
era) carry the raw generated text per item but were written before gen_toks
capture. Re-tokenize each record's text with its own model's tokenizer to get
per-item lengths. Items carry no doc ids, so before pairing across arms the
fixed ordering is verified via the gold sequences: arms of a record pair ONLY
if their 228-long gold sequences match exactly; mismatching records are
reported and left unpaired (ordering_ok column).

Model identity per record comes from screening_genbench.csv / the CSV E,k
columns (E=128 gemma4-26B-IT; E=256 qwen3.5-35B; E=512 = qwen35 halfgrain,
same tokenizer). Output: results/ablations/mmlu_dual_lengths.csv, one row per
(record, arm, idx).
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

GEMMA_TOK = "google/gemma-4-26B-A4B-it"          # tokenizer-only cache hit
QWEN_TOK = "/workspace/instruct-models/qwen35-35b-a3b-instruct"
MIN_ITEMS = 100          # early 4-item files are the pre-fix overwrite bug: skip


def family_of(record):
    if record.startswith(("qwen35_",)):
        return "qwen35"
    if record.startswith(("gemma4_", "pair_", "scr_", "dual_")):
        return "gemma4"          # E=128,k=8 in screening_genbench.csv
    raise SystemExit(f"unknown record family: {record}")


def main():
    files = sorted(glob.glob(os.path.join(ABLATIONS, "genbench_samples",
                                          "*_mmlu_dual.json")))
    recs = {}
    for f in files:
        m = re.match(r"(.+)_(free|R\d+)_mmlu_dual\.json$", os.path.basename(f))
        assert m, f
        items = json.load(open(f))["items"]
        if len(items) < MIN_ITEMS:
            print(f"[skip] {os.path.basename(f)}: {len(items)} items (early bug era)")
            continue
        recs.setdefault(m.group(1), {})[m.group(2)] = items

    from transformers import AutoTokenizer
    toks = {}

    def tok(fam):
        if fam not in toks:
            toks[fam] = AutoTokenizer.from_pretrained(
                GEMMA_TOK if fam == "gemma4" else QWEN_TOK)
        return toks[fam]

    out = os.path.join(ABLATIONS, "mmlu_dual_lengths.csv")
    with open(out, "w", newline="") as fh:
        fh.write('"# Per-item generation lengths re-tokenized from the committed '
                 'mmlu_dual dumps (think-off MMLU, 228 items = 4/subject x 57). '
                 'gen_toks: model-own-tokenizer count of the dumped text (post-strip '
                 '= full generation for think-off). ordering_ok: 1 iff this '
                 "record's arms share an identical 228-long gold sequence, the "
                 'precondition for pairing rows across arms by idx. correct_*: '
                 'pred==gold under the relaxed / strict extractor. Producer: '
                 'analysis/residency/mmlu_dual_lengths.py"\n')
        w = csv.writer(fh)
        w.writerow(["record", "model_family", "arm", "idx", "gold",
                    "correct_relaxed", "correct_strict", "gen_toks",
                    "ordering_ok"])
        ref_gold = {}
        for rec in sorted(recs):
            arms = recs[rec]
            fam = family_of(rec)
            golds = {a: [x["gold"] for x in items] for a, items in arms.items()}
            ok = len({tuple(g) for g in golds.values()}) == 1
            if not ok:
                print(f"[UNPAIRABLE] {rec}: gold sequences differ across arms "
                      f"{sorted(arms)} -- rows written with ordering_ok=0")
            else:
                ref_gold.setdefault(fam, tuple(golds[sorted(arms)[0]]))
            for arm, items in sorted(arms.items()):
                t = tok(fam)
                for i, x in enumerate(items):
                    n = len(t(x["text"], add_special_tokens=False).input_ids)
                    w.writerow([rec, fam, arm, i, x["gold"],
                                int(x["pred_relaxed"] == x["gold"]),
                                int((x["pred_strict"] or "") == x["gold"]),
                                n, int(ok)])
                lens = [len(t(x["text"], add_special_tokens=False).input_ids)
                        for x in items]
                print(f"[ok] {rec} {arm} ({fam}): n={len(items)} "
                      f"mean={sum(lens)/len(lens):.0f} max={max(lens)}")
        # cross-record check (informational): same doc set => same gold sequence
        for fam, g in ref_gold.items():
            same = [r for r in recs if family_of(r) == fam and
                    all(tuple(x["gold"] for x in it) == g
                        for it in recs[r].values())]
            print(f"[info] {fam}: {len(same)}/{sum(family_of(r) == fam for r in recs)} "
                  f"records share the reference gold sequence")
    print("wrote", out)


if __name__ == "__main__":
    main()
