#!/usr/bin/env python3
"""Three-way read: published vs real-corpus rebuild vs self-generated rebuild.

The two rebuild arms differ in exactly two lanes -- math and code -- so the difference
between them isolates what those lanes carry. Everything else (pool size, screen, length
gate, trajectory cap, cut, ranks, KL, budget, grid) is identical.

  arm A (real-corpus)  every lane mined from WildChat/oasst2
  arm B (self-gen)     math and code AUTHORED by the model, as the original pool did

Published damages are vs the unadapted base FREE arm. gemma MMLU is reported against its
own free arm instead, because the published -1.8 is a multi-run mean over screening-protocol
bases that DATA_CONTRACT says are not comparable to full runs.

    compare_arms.py --family gemma4
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FAM = {
 "gemma4": dict(
   base="gemma4_instruct", A="gemma4_ce_rebuild", B="gemma4_ce_selfgen",
   rows=[("GSM8K","gsm8k_cot_zeroshot","exact_match,flexible-extract",0.0,-6.0,False),
         ("IFEval","ifeval","prompt_level_strict_acc,none",-1.0,0.0,False),
         ("HumanEval","humaneval_gemma_fixed","pass@1,channel-aware",-1.2,-6.1,False),
         ("MMLU","mmlu_gptoss_relaxed","acc,relaxed-extract",-1.8,-0.2,True)]),
 "qwen35": dict(
   base=None, baseabs={"GSM8K":0.845,"IFEval":0.890,"HumanEval":0.921,"MMLU":0.930},
   A="qwen35_ce_rebuild", B="qwen35_ce_selfgen",
   rows=[("GSM8K","gsm8k_cot_zeroshot","exact_match,flexible-extract",-3.5,None,False),
         ("IFEval","ifeval","prompt_level_strict_acc,none",-6.0,None,False),
         ("HumanEval","humaneval_instruct","pass@1,create_test",-1.2,None,False),
         ("MMLU","mmlu_gptoss_relaxed","acc,relaxed-extract",-0.4,None,True)]),
}


def load():
    rows = [l for l in open(os.path.join(ABLATIONS, "instruct_genbench_vllm.csv"))
            if not l.startswith('"#')]
    return {(x["model"], x["arm"], x["task"], x["metric"]): float(x["value"])
            for x in csv.DictReader(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="gemma4", choices=sorted(FAM))
    ap.add_argument("--arm", default="R8")
    A = ap.parse_args()
    G, f = load(), FAM[A.family]

    print(f"\n{A.family}  arm {A.arm}   damage in points (negative = worse than base free)")
    print(f"{'':11s} {'published':>10} {'real-corpus':>12} {'self-gen':>10} {'B-A':>7}  lane changed?")
    for lbl, task, met, pub, bpub, dual in f["rows"]:
        def dmg(rec):
            r = rec + ("_dual" if dual else "")
            got = G.get((r, A.arm, task, met))
            if got is None:
                return None
            if f.get("base"):
                b = f["base"] + ("_dual" if dual else "")
                ref = G.get((b, "free", task, met))
                if ref is None:                       # fall back to own free arm
                    ref = G.get((r, "free", task, met))
            else:
                ref = f["baseabs"][lbl]
            return None if ref is None else 100.0 * (got - ref)
        a, b = dmg(f["A"]), dmg(f["B"])
        changed = "YES (math/code)" if lbl in ("GSM8K", "HumanEval") else "no"
        fa = f"{a:+12.1f}" if a is not None else f"{'--':>12}"
        fb = f"{b:+10.1f}" if b is not None else f"{'pending':>10}"
        fd = f"{b-a:+7.1f}" if (a is not None and b is not None) else f"{'':>7}"
        print(f"{lbl:11s} {pub:+10.1f} {fa} {fb} {fd}  {changed}")
    print("\nB-A is the effect of swapping math+code to self-generated. The two 'no' rows are")
    print("the control: their lanes are identical across arms, so they should not move.")


if __name__ == "__main__":
    main()
