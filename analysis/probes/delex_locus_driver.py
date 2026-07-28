#!/usr/bin/env python3
"""Driver for delex-1e19 (b) locus probes. Runs delex_locus.analyze on the 3 cells, writes
results/ablations/mechinterp_locus_1e19.csv, and enforces the null-control gate: median null AUC
(iid permutation AND circular shift) must be 0.500 +- 0.002 for every model, else STOP and report.
"""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delex_locus

ROOT = os.environ.get("TEMPORAL_MOE_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/mechinterp_locus_1e19.csv")
CELLS = [("moe_coarse_1e19", "moe_coarse_1e19"),
         ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
         ("temporal_fine_1e19", "temporal_fine_g3_1e19")]


def main():
    verify = "--verify" in sys.argv
    all_rows, summary, gate_ok = [], [], True
    for label, run in CELLS:
        cap = os.path.join(RUNS, run, "delex_capture.pt")
        if not os.path.exists(cap):
            print(f"[skip] {label}: no capture"); continue
        rows, pm = delex_locus.analyze(cap, label, run, verify=verify)
        all_rows += rows
        tok = np.median(pm["tok"]); ctx = np.median(pm["ctx"])
        dom = pm["ndom"] / max(1, pm["ntot"])
        ni = np.nanmedian(pm["null_iid"]); ns = np.nanmedian(pm["null_shift"])
        ok = abs(ni - 0.5) <= 0.002 and abs(ns - 0.5) <= 0.002
        gate_ok = gate_ok and ok
        summary.append((label, tok, ctx, dom, ni, ns, ok))
        print(f"[ok] {label}: n={pm['ntot']} med_tokAUC={tok:.3f} med_ctxAUC={ctx:.3f} "
              f"ctx_dom={dom*100:.0f}% null_iid={ni:.4f} null_shift={ns:.4f} gate={'PASS' if ok else 'FAIL'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "run", "layer", "expert", "usage_count",
                    "token_AUC", "context_AUC", "context_minus_token", "variant"])
        w.writerows(all_rows)
    print(f"[write] {OUT}: {len(all_rows)} rows")
    print("\nNULL-CONTROL GATE:", "PASS (all models 0.500+-0.002)" if gate_ok
          else "FAIL — nulls off 0.500, results suspect, STOP and report")
    print("\nHEADLINE  model | med token_AUC | med context_AUC | % context-dominated | null(iid/shift)")
    for lab, tk, cx, dm, ni, ns, ok in summary:
        print(f"  {lab:22} {tk:.3f}          {cx:.3f}          {dm*100:5.0f}%        {ni:.3f}/{ns:.3f}")
    return gate_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 2)
