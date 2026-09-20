#!/usr/bin/env python3
"""Per-item token lengths for the committed WritingBench responses.

The rescued response files predate gen_toks capture in wb_generate.py, so
lengths are recovered by re-tokenizing each record's response text with its
own model's tokenizer. Writes results/ablations/writingbench/response_lengths.csv
with one row per (record, arm, subset, index).
"""
import csv
import glob
import json
import os
import re

RESP = "/workspace/temporal-moe/results/ablations/writingbench/responses"
OUT = "/workspace/temporal-moe/results/ablations/writingbench/response_lengths.csv"

TOK = {  # record prefix -> tokenizer source (model's own)
    "gemma4": "google/gemma-4-26B-A4B-it",
    "qwen35": "/root/models/qwen35-35b-a3b",
    "smoke_qwen": "/root/models/qwen35-35b-a3b",
    "oss20": "openai/gpt-oss-20b",
    "oss120": "openai/gpt-oss-120b",
    "lfm25": "LiquidAI/LFM2.5-8B-A1B",
}
NAME = re.compile(r"(.+)_(free|R\d+)(_s[BC])?\.jsonl$")


def main():
    from transformers import AutoTokenizer
    toks = {}

    def tok(rec):
        pre = next((p for p in TOK if rec.startswith(p)), None)
        assert pre, f"no tokenizer mapping for record {rec}"
        if pre not in toks:
            toks[pre] = AutoTokenizer.from_pretrained(TOK[pre])
        return toks[pre]

    with open(OUT, "w", newline="") as fh:
        fh.write('"# Per-item WritingBench response lengths, re-tokenized with each '
                 "model's own tokenizer (rescued responses predate native gen_toks "
                 'capture). subset: A = base 50 queries, B/C = the _sB/_sC subset '
                 'files (offsets 50/100). index = WritingBench query index, the join '
                 'key to scores/*.jsonl. Producer: analysis/writingbench/wb_lengths.py"\n')
        w = csv.writer(fh)
        w.writerow(["record", "arm", "subset", "index", "gen_toks"])
        for f in sorted(glob.glob(os.path.join(RESP, "*.jsonl"))):
            m = NAME.match(os.path.basename(f))
            assert m, f
            rec, arm = m.group(1), m.group(2)
            subset = (m.group(3) or "_sA")[2:]
            t = tok(rec)
            n = 0
            lens = []
            for line in open(f):
                d = json.loads(line)
                gl = d.get("gen_toks") or len(
                    t(d["response"], add_special_tokens=False).input_ids)
                w.writerow([rec, arm, subset, d["index"], gl])
                lens.append(gl)
                n += 1
            print(f"[wb-len] {rec} {arm} {subset}: n={n} "
                  f"mean={sum(lens)/len(lens):.0f} max={max(lens)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
