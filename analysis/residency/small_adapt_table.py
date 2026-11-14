#!/usr/bin/env python3
"""On-policy adaptation of the two small released models (OLMoE-1B-7B, LFM2.5-8B-A1B): base vs adapted,
both arms, every surface task, from instruct_genbench_vllm.csv; GSM8K deltas paired by question (McNemar,
arm_power.load). Base rows are the full-size records (n=1319 GSM8K, 541 IFEval) where they exist.

  $PY analysis/residency/small_adapt_table.py olmoe:olmoe_ce_online_scratch_e16 lfm25:lfm25_ce_online_scratch_e16
"""
import csv, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_power import load  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/ablations/instruct_genbench_vllm.csv")
# task -> (csv task name, metric prefix, base-record suffix, adapted-record suffix)
TASKS = {"gsm8k": ("gsm8k_cot_zeroshot", "exact_match,flexible-extract", "_n1319", "_n1319"),
         "ifeval": ("ifeval", "prompt_level_strict_acc", "_full", "_full"),
         "mmlu": ("mmlu_gptoss_relaxed", "acc,relaxed-extract", "", "_n_dual"),
         "humaneval": ("humaneval_instruct", "pass@1", "", "_code"),
         "mbpp": ("mbpp_chat", "pass@1", "_mbpp", "_mbpp")}
FAM = {"olmoe": ("olmoe_instruct", "R8"), "lfm25": ("lfm25_instruct", "R4")}


def rows():
    out = {}
    for r in csv.reader(open(CSV)):
        if len(r) < 10 or r[0].startswith("#") or r[0] == "model":
            continue
        out.setdefault((r[0], r[3], r[5]), {})[r[6]] = (float(r[7]), r[8])
    return out


def get(tab, rec, arm, task, metric):
    d = tab.get((rec, arm, task))
    if not d:
        return None
    for k, v in d.items():
        if k.startswith(metric):
            return v
    return None


def main():
    tab = rows()
    for spec in sys.argv[1:]:
        fam, adapted = spec.split(":")
        base, arm = FAM[fam]
        print(f"\n{adapted} vs {base} (arm {arm}); values in %, n from the record")
        print(f"{'task':10s} {'n':>5s} | {'free base':>9s} {'free adapt':>10s} {'delta':>6s} | {arm+' base':>9s} {arm+' adapt':>9s} {'delta':>6s}")
        for name, (task, metric, bs, as_) in TASKS.items():
            if fam == "lfm25" and name == "humaneval":
                print(f"{name:10s}   not run for LFM (its HumanEval cell uses humaneval_think, which has no adapter path)")
                continue
            vals = [get(tab, rec, a, task, metric) for rec, a in
                    ((base + bs, "free"), (adapted + as_, "free"), (base + bs, arm), (adapted + as_, arm))]
            if any(v is None for v in vals):
                print(f"{name:10s}   missing rows: " + ", ".join(f"{rec+s}/{a}" for (rec, s, a), v in
                      zip(((base, bs, "free"), (adapted, as_, "free"), (base, bs, arm), (adapted, as_, arm)), vals) if v is None))
                continue
            (fb, n), (fa, _), (cb, _), (ca, _) = vals
            line = (f"{name:10s} {n:>5s} | {100*fb:9.1f} {100*fa:10.1f} {100*(fa-fb):+6.1f} | "
                    f"{100*cb:9.1f} {100*ca:9.1f} {100*(ca-cb):+6.1f}")
            if name == "gsm8k":       # paired by question
                for a in ("free", arm):
                    b_, a_ = load(base + bs, a, task), load(adapted + as_, a, task)
                    if b_ and a_:
                        k = sorted(set(b_) & set(a_))
                        w = sum(1 for d in k if a_[d] and not b_[d]); l = sum(1 for d in k if b_[d] and not a_[d])
                        se = math.sqrt(w + l) / len(k)
                        line += f"   paired {a}: {100*(w-l)/len(k):+.1f} +/- {100*se:.1f} z={(w-l)/len(k)/se if se else 0:+.2f}"
            print(line)


if __name__ == "__main__":
    main()
