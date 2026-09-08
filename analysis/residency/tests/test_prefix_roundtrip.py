#!/usr/bin/env python3
"""Test T1 (TRUNCATION_RERUN_PLAN §3, Blocker A): can a saved transcript be turned
back into the exact token sequence the model produced?

Continuing a truncated generation means resuming from its prefix. The dumps store
`raw` text and a token count, not token IDs, so a continuation must re-tokenize
`raw` -- and that only reproduces the original sequence if the text round-trips
exactly. Where it does not, continuation resumes from a sequence the model never
generated: the output looks entirely plausible and is silently wrong.

Two assertions per item, because equal length does not prove equal segmentation:
  1. decode(encode(raw)) is byte-identical to raw
  2. len(encode(raw)) == the recorded gen_toks

Emits results/ablations/prefix_roundtrip_allowlist.json: the (record, task) pairs
that pass. The continuation path must read that allowlist and refuse anything
absent from it. THIS TEST IS GREEN WHEN IT REPORTS ACCURATELY, not when
everything passes -- transcripts carrying channel/think markers are expected to
fail, because those markers were written into `raw` as literal text and do not
re-tokenize back to their special-token IDs.

Run: python analysis/residency/tests/test_prefix_roundtrip.py
"""
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # analysis/residency
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # analysis/
from paths import ABLATIONS                                          # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
PER_DUMP = 60            # items sampled per dump (runtime); stated in the output
TOK = {"qwen35": "/workspace/instruct-models/qwen35-35b-a3b-instruct",
       "gemma4": "google/gemma-4-26B-A4B-it",
       "gptoss_20b": "openai/gpt-oss-20b",
       "gptoss_120b": "openai/gpt-oss-120b",
       "lfm25": "LiquidAI/LFM2.5-8B-A1B",
       "olmoe": "allenai/OLMoE-1B-7B-0125-Instruct"}


def tok_for(rec):
    for pre in ("gptoss_120b", "gptoss_20b", "qwen35", "gemma4", "lfm25", "olmoe"):
        if rec.startswith(pre):
            return pre
    return None


def main():
    from transformers import AutoTokenizer
    cache, allow, rows = {}, [], []
    for p in sorted(glob.glob(f"{SAMP}/*.json")):
        b = os.path.basename(p)[:-5]
        m = re.match(r"(.+)_(free|R\d+)_(.+)$", b)
        if not m:
            continue
        rec, arm, task = m.group(1), m.group(2), m.group(3)
        fam = tok_for(rec)
        if fam is None:
            continue
        items = json.load(open(p))["items"]
        items = [i for i in items if "raw" in i and "gen_toks" in i]
        if not items:
            continue
        if fam not in cache:
            try:
                cache[fam] = AutoTokenizer.from_pretrained(TOK[fam])
            except Exception as e:
                print(f"  skip {fam}: tokenizer unavailable ({type(e).__name__})")
                cache[fam] = False
        tk = cache[fam]
        if tk is False:
            continue
        n = ok_txt = ok_len = 0
        drift = []
        for it in items[:PER_DUMP]:
            ids = tk(it["raw"], add_special_tokens=False).input_ids
            n += 1
            ok_txt += tk.decode(ids) == it["raw"]
            ok_len += len(ids) == it["gen_toks"]
            drift.append(len(ids) - it["gen_toks"])
        # PROVENANCE GATE: a dump can only certify a prefix if its gen_toks came
        # from the ENGINE (len(output.token_ids)). The mmlu harness computed
        # gen_toks by tokenizing `raw` with this same tokenizer, so comparing a
        # re-tokenization against it is circular and proves nothing. Those dumps
        # are "unverifiable", not "pass" -- the fix is engine token IDs in the
        # dump (Blocker A item 1), not a greener-looking table.
        engine_truth = not task.startswith("mmlu")
        passed = (ok_txt == n and ok_len == n and engine_truth)
        rows.append((rec, arm, task, n, ok_txt, ok_len,
                     sorted(drift)[len(drift) // 2], min(drift, key=abs)
                     if drift else 0, max(drift, key=abs) if drift else 0, passed))
        if passed:
            allow.append({"record": rec, "task": task})
    hdr = f"{'record':30s} {'task':22s} {'n':>4} {'text-ok':>8} {'len-ok':>7} {'med':>5} {'worst':>6}"
    print(hdr)
    for rec, arm, task, n, ot, ol, med, _, worst, passed in rows:
        mark = ("PASS" if passed else
                ("unverifiable (gen_toks re-tokenized, not engine)"
                 if task.startswith("mmlu") else "fail"))
        print(f"{rec:30s} {task[:22]:22s} {n:4d} {ot:8d} {ol:7d} {med:5d} {worst:6d}  {mark}")
    uniq = sorted({(a["record"], a["task"]) for a in allow})
    out = os.path.join(ABLATIONS, "prefix_roundtrip_allowlist.json")
    json.dump({"note": "(record, task) pairs whose saved transcripts re-tokenize "
                       "to the exact original sequence. ONLY these may use a "
                       "continuation/resume path; everything else must regenerate "
                       "from scratch. Producer: "
                       "analysis/residency/tests/test_prefix_roundtrip.py",
               "items_sampled_per_dump": PER_DUMP,
               "allow": [{"record": r, "task": t} for r, t in uniq]},
              open(out, "w"), indent=1)
    fams = {}
    for rec, _, _, _, _, _, _, _, _, passed in rows:
        f = tok_for(rec)
        fams.setdefault(f, [0, 0])
        fams[f][0] += 1
        fams[f][1] += passed
    print("\nper family (dumps passing / total):")
    for f, (t, p) in sorted(fams.items()):
        print(f"  {f:12s} {p:3d}/{t:3d}")
    print(f"\nallowlist: {len(uniq)} (record, task) pairs -> {out}")
    if not uniq:
        print("EMPTY BY DESIGN: no committed dump carries engine token IDs, so no\n"
              "prefix can be certified today. Continuation stays unavailable until\n"
              "drivers persist gen_ids (Blocker A item 1). Regeneration is unaffected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
