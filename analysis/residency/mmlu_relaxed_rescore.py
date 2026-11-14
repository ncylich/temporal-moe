#!/usr/bin/env python3
"""Add the reported relaxed-extraction MMLU metric to a run that recorded only the
stock flan filter, offline from its committed raw text.

Why this exists. The Group A re-run of gemma4 think-on MMLU at 8192 went through
the `mmlu_flan_cot_fewshot` task, so it wrote `exact_match,get-answer` and nothing
else. The paper reports relaxed answer extraction (Appendix D), which is what the
`mmlu_gptoss_relaxed` task emits. Wiring the new rows in without this step does not
change the budget, it changes the METRIC: gemma4 think-on R8 reads -4.4 relaxed and
-9.2 strict at the same 4096 budget, so a resolver preferring the new record would
report -12.3 against -4.4 as though a budget had moved it.

Everything here is re-analysis of committed dumps. No GPU, no regeneration, and no
resampling, so the rescored rows describe the same generations the strict rows do.
That matters more than the cost: a fresh boot draws a fresh sample, and one has
already been measured at 3.5 points on a Qwen3.5 IFEval cell.

Method, and it is the harness's own path rather than a reimplementation:
  * `extract` and `STRICT` are imported from mmlu_gptoss, not copied;
  * the think segment is stripped exactly as genprotocol.install does it,
    `raw.split(marker)[-1]`, with the per-model marker mmlu_gptoss chooses;
  * gold labels come from the parent record's mmlu_dual dump, joined on `doc`.

Validated before use, and the validation is worth keeping. Extracting from `raw`
instead of the post-strip text disagrees with the harness on 3 to 5 of 228 items
per arm and moves accuracy inconsistently (free down, R8 up), because the
extractor then reads the model's own deliberation. Reconstructing the strip first
reproduces the stored text byte-for-byte on 228/228 and the recorded accuracy
exactly on every arm. --validate re-runs that check.

    mmlu_relaxed_rescore.py --validate
    mmlu_relaxed_rescore.py --record gemma4_think_on_cap8k --gold-from gemma4_think_on
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genprotocol                                                   # noqa: E402
from mmlu_gptoss import STRICT, extract                              # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")
GRID = os.path.join(ABLATIONS, "instruct_genbench_vllm.csv")

# the marker mmlu_gptoss.THINK_MARK picks per architecture, keyed by record prefix
MARK = {"gemma4": "<channel|>", "qwen35": "</think>", "lfm25": "</think>",
        "gptoss": "<|channel|>final<|message|>"}

# cells whose recorded relaxed accuracy is known, used by --validate
KNOWN = {("gemma4_think_on", "free"): 0.938596,
         ("gemma4_think_on", "R8"): 0.894737,
         ("gemma4_think_on", "R16"): 0.929825}


def marker(record):
    for pre, mk in MARK.items():
        if record.startswith(pre):
            return mk
    raise SystemExit(f"no think marker known for {record}")


def strip(raw, mk):
    """The scoring text, as genprotocol.install builds it."""
    return raw.split(mk)[-1] if mk in raw else raw


def load(record, arm, task):
    p = os.path.join(SAMP, f"{record}_{arm}_{task}.json")
    return json.load(open(p))["items"] if os.path.exists(p) else None


def score(items, gold, mk):
    """(relaxed acc, strict acc, per-item dump rows). gold maps doc -> letter."""
    rows, hit_r, hit_s = [], 0, 0
    for i in items:
        text = strip(i["raw"], mk)
        pr = extract(text)
        sm = STRICT.search(text)
        ps = sm.group(1) if sm else None
        g = gold[i["doc"]]
        hit_r += (pr == g)
        hit_s += (ps == g)
        # carry the length fields through. Dropping raw_toks silently demotes the
        # dump to the raw-text route in length_figs.lengths(), where total becomes
        # the ANSWER and the cell reads as thinking-off.
        row = {"doc": i["doc"], "raw": i["raw"], "text": text, "gold": g,
               "pred_relaxed": pr, "pred_strict": ps,
               "gen_toks": i.get("gen_toks")}
        for f in ("raw_toks", "doc_id", "gen_ids"):
            if f in i:
                row[f] = i[f]
        rows.append(row)
    n = max(1, len(items))
    return hit_r / n, hit_s / n, rows


def validate():
    ok = True
    for (rec, arm), ref in KNOWN.items():
        items = load(rec, arm, "mmlu_dual")
        if not items:
            print(f"  {rec} {arm}: no mmlu_dual dump, skipped")
            continue
        mk = marker(rec)
        gold = {i["doc"]: i["gold"] for i in items}
        acc, _, rows = score(items, gold, mk)
        same_text = sum(1 for i, r in zip(items, rows) if r["text"] == i["text"])
        same_pred = sum(1 for i, r in zip(items, rows)
                        if r["pred_relaxed"] == i.get("pred_relaxed"))
        good = abs(acc - ref) < 5e-6 and same_text == len(items)
        ok &= good
        print(f"  {rec} {arm:5} recorded {ref:.6f}  offline {acc:.6f}  "
              f"text {same_text}/{len(items)}  pred {same_pred}/{len(items)}  "
              f"{'OK' if good else 'MISMATCH'}")
    print("VALIDATION", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", help="record to rescore, e.g. gemma4_think_on_cap8k")
    ap.add_argument("--gold-from", help="record whose mmlu_dual dump carries gold")
    ap.add_argument("--arms", default="free,R8,R16")
    ap.add_argument("--task", default="mmlu_flan_cot_fewshot",
                    help="the task the run recorded under")
    ap.add_argument("--validate", action="store_true",
                    help="only re-check that this path reproduces the harness")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rows without writing anything")
    A = ap.parse_args()

    if A.validate or not A.record:
        raise SystemExit(0 if validate() else 1)
    if not validate():
        raise SystemExit("refusing to rescore: the offline path no longer "
                         "reproduces the harness")

    src = load(A.gold_from, "free", "mmlu_dual")
    if not src:
        raise SystemExit(f"no gold source at {A.gold_from} free mmlu_dual")
    gold = {i["doc"]: i["gold"] for i in src}
    mk = marker(A.record)

    E = k = None
    for r in csv.reader(open(GRID)):
        if len(r) > 9 and r[0] == A.record:
            E, k = r[1], r[2]
            break

    out = []
    for arm in A.arms.split(","):
        items = load(A.record, arm, A.task)
        if not items:
            print(f"  {arm}: no dump, skipped")
            continue
        missing = [i["doc"] for i in items if i["doc"] not in gold]
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} docs absent from gold "
                             f"({missing[:3]}) -- wrong --gold-from?")
        acc, acc_s, rows = score(items, gold, mk)
        print(f"  {arm:5} n={len(items)}  relaxed={acc:.6f}  strict-flan={acc_s:.6f}")
        R = arm[1:] if arm.startswith("R") else ""
        cap = 0
        for r in csv.reader(open(GRID)):
            if len(r) > 9 and r[0] == A.record and r[3] == arm and r[5] == A.task:
                cap = r[9]
                break
        for met, val in (("acc,relaxed-extract", acc), ("acc,strict-flan", acc_s)):
            out.append([A.record, E, k, arm, R, "mmlu_gptoss_relaxed", met,
                        f"{val:.6f}", 4, cap, "0"])
        if not A.dry_run:
            genprotocol.DUMP_DIR = SAMP
            genprotocol.write_dump(A.record, arm, "mmlu_dual", rows, len(items))

    if A.dry_run:
        print("\n--dry-run, rows not written:")
        for r in out:
            print("   ", ",".join(map(str, r)))
        return
    with open(GRID, "a", newline="") as fh:
        csv.writer(fh).writerows(out)
    print(f"appended {len(out)} rows to {GRID}")


if __name__ == "__main__":
    main()
