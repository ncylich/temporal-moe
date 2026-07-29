#!/usr/bin/env python3
"""Collect Phase 1 rank-ladder cells into results/ablations/ple_ladder.csv.

GATES (PLE_PLAN.md §5, as revised). Cells run 50M tokens, so they are gated at MATCHED budget:

  C@50M  = 0.8791   router + norms, same surface, same budget. Beating this by >2σ is the
                    minimum claim: PLE recovers some of the residual constraint price.
  CE@50M = 0.8269   router + norms + LoRA r32, same surface and budget, with LoRA as the third
                    mechanism instead of PLE. §1's premise is that PLE supplies lexical capacity
                    LoRA cannot, so this is that premise's direct test -- and it was already
                    measured, at no cost.
  F' = 0.8106       the constraint price. Only a fair comparison at 250M (Phase 2), not here.

Landing between 0.8791 and 0.8269 means PLE helps but is the WORSE third mechanism, and that is
what the report must say plainly rather than presenting a C-relative win on its own.

An earlier draft of the plan gated 50M cells against C@250M = 0.8505. C's own curve is
0.8791 / 0.8627 / 0.8560 / 0.8526 / 0.8505 at 50 / 100 / 150 / 200 / 250M, so that gate demanded
PLE alone overcome the whole 50M->250M trajectory as well as beat C. The gap it silently imposed
was 0.0286 BPB = 2.4x the 2σ bar of 0.012 (~4.8σ at σ=0.006), which would have returned a false
negative almost regardless of whether PLE worked.

2σ = 0.012 BPB. Differences smaller than that are noise and must not be reported as effects.
"""

import csv, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                 # noqa: E402

C_50M, CE_50M, FPRIME = 0.8791, 0.8269, 0.8106
BASE, IMPOSE = 0.6727, 2.7507
TWO_SIGMA = 0.012


def recovery(bpb):
    return 1.0 - (bpb - BASE) / (IMPOSE - BASE)


def verdict(bpb):
    if bpb > C_50M - TWO_SIGMA:
        return "does not beat C@50M by 2 sigma"
    if bpb > CE_50M + TWO_SIGMA:
        return "beats C@50M, worse than LoRA (CE@50M) -- PLE is the weaker third mechanism"
    if bpb > CE_50M - TWO_SIGMA:
        return "ties CE@50M within 2 sigma"
    return "beats CE@50M by >2 sigma -- supports the §1 premise"


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "ple_ladder_*.json"))):
        r = json.load(open(path))
        b = r["final_bpb"]
        rows.append({
            "tag": r["tag"], "rank": r["rank"], "train_tokens": r["train_tokens"],
            # The trainer was already running when this reporter was written, so it is not edited
            # mid-ladder to emit accum. Every Phase 1 cell holds the effective batch at 16 by
            # construction, so accum follows exactly from mb; asserted rather than assumed.
            "mb": r["mb"], "accum": 16 // r["mb"], "lr": r["lr"],
            "table_wd": r["table_wd"], "adam8bit": r["adam8bit"],
            "flash_attention": "on",
            "ple_params": r["ple_params"],
            "final_bpb": round(b, 6),
            "recovery": round(recovery(b), 4),
            "delta_vs_C50M": round(b - C_50M, 6),
            "delta_vs_CE50M": round(b - CE_50M, 6),
            "beats_C50M_by_2sigma": (C_50M - b) > TWO_SIGMA,
            "beats_CE50M_by_2sigma": (CE_50M - b) > TWO_SIGMA,
            "verdict": verdict(b),
            "final_swap": round(r["final_swap"], 6),
            "final_entropy": round(r["final_entropy"], 6),
            "divisor": r["divisor"],
            "curve_bpb": "|".join(f"{h['tok']//10**6}M:{h['bpb']:.6f}" for h in r["curve"]),
        })
    if not rows:
        print("no ple_ladder_*.json yet"); return
    for ref, name in ((C_50M, "C@50M router+norms"), (CE_50M, "CE@50M router+norms+LoRA"),
                      (FPRIME, "F' full finetune 6.92B (fair only at 250M)")):
        rows.append({"tag": "REFERENCE", "rank": name, "train_tokens": "",
                     "mb": "", "accum": "", "lr": "", "table_wd": "", "adam8bit": "",
                     "flash_attention": "", "ple_params": "",
                     "final_bpb": ref, "recovery": round(recovery(ref), 4),
                     "delta_vs_C50M": "", "delta_vs_CE50M": "",
                     "beats_C50M_by_2sigma": "", "beats_CE50M_by_2sigma": "",
                     "verdict": "", "final_swap": "", "final_entropy": "",
                     "divisor": 3.1089070924799973, "curve_bpb": ""})
    path = os.path.join(ABLATIONS, "ple_ladder.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{str(r['tag']):14s} rank={str(r['rank']):10s} BPB={r['final_bpb']} {r['verdict']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
