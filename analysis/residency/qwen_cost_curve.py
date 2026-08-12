#!/usr/bin/env python3
"""What does residency actually cost on Qwen3.5-35B, as a function of budget and of which layers?

Adaptation on this model is not reachable on one GPU: HF's Qwen MoE forward loops in Python over hit
experts (40 layers x up to 256 experts), the per-expert LoRA branch doubles that loop's work, and
gradient checkpointing recomputes all of it -- measured at 176 s/step, or 93 tok/s against 4700 tok/s
(that figure is specific to Qwen3.5 with 461M expert LoRA at mb=1; it is NOT a general stock
baseline and was wrongly used as one -- see results/ablations/crossmodel_RESULTS.md S9)
for the same model in eval. Three matched training arms were not reachable before the deadline, so
the budget goes where this hardware is 50x more productive: test-time characterisation, which is what
the scaling question asked for in the first place and which needs no base-vs-instruct distinction.

Three things in one model load, all replaying identical cached batches:

  A. cost curve      all 40 layers constrained at R in {4,8,16,32,64,128}. Gives the price of
                     residency as a function of resident budget rather than at a single operating
                     point, so the OLMoE comparison can be made at matched FRACTION or matched R.
  B. free-set        at R=8 and R=32: which layers are actually worth exempting. Includes the
                     OLMoE-inherited {0,1,L-2,L-1} and tail-only sets, because the per-layer profile
                     here is back-heavy and the inherited recipe may be mis-weighted. Solo damage is
                     known not to predict joint value, so these are measured jointly, not summed.
  C. shared expert   damage with Qwen's always-resident shared expert live, and with its contribution
                     zeroed. Qwen pays ~70x less than OLMoE for the same constraint; if that is the
                     shared expert giving every token an intact path, the result is architecture-
                     specific, and if it is expert redundancy, it should generalise. Measured as a
                     difference of differences so it does not depend on the shared expert's absolute
                     worth.

    qwen_cost_curve.py --n-seq 24
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402
from qwen_sweep import batches, score, preflight                   # noqa: E402


DATA = "/workspace/qwen35-adapt/data"
OUT = "/workspace/qwen35-adapt/results"
SHARED = {"on": True}


def _block(self, hidden_states):
    """Stock block, with the shared-expert contribution switchable. Zeroed rather than removed, so
    the routed path sees exactly the tokens it would otherwise see."""
    b, s, h = hidden_states.shape
    self.gate._resid_shape = (b, s)
    x = hidden_states.view(-1, h)
    _, w, idx = self.gate(x)
    out = self.experts(x, idx, w)
    if SHARED["on"]:
        out = out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    return out.reshape(b, s, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--family", default="qwen3_5", choices=("qwen3_5", "qwen3"))
    ap.add_argument("--model", default="/workspace/qwen35-adapt/model")
    ap.add_argument("--data", default="/workspace/qwen35-adapt/data")
    ap.add_argument("--slice-name", default="qwen")
    ap.add_argument("--out", default="/workspace/qwen35-adapt/results")
    ap.add_argument("--tag", default="qwen35")
    A = ap.parse_args()
    global DATA, OUT
    DATA, OUT = A.data, A.out
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(f"{DATA}/bpb_slice_meta_{A.slice_name}.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model(path=A.model, family=A.family)
    cfg = model.config
    E, L = cfg.num_experts, cfg.num_hidden_layers
    import qwen_sweep as QS
    QS.DATA = A.data                     # batches() reads the slice from here
    bl = [torch.load(f"{A.data}/bpb_slice_ids_{A.slice_name}.pt", weights_only=False)[i:i+1].long()
          for i in range(A.n_seq)]
    ALL = list(range(L))
    preflight(model, bl, {"E": E, "n_layers": L})
    has_shared = hasattr(cfg, "shared_expert_intermediate_size")
    if has_shared:
        RQ.FAMILIES[A.family][0].forward = _block   # after preflight, which checks the stock path
    else:
        print("  [note] no shared expert on this family; part C is not applicable", flush=True)

    rows = []
    def cell(part, name, free_set, R, shared=True):
        SHARED["on"] = shared
        t0 = time.time()
        bpb = score(model, bl, D, free_set, R)[0]
        rows.append({"part": part, "cell": name, "R": R,
                     "resident_routed_pct": f"{100*R/E:.2f}", "shared_expert": "on" if shared else "zeroed",
                     "bpb": f"{bpb:.6f}", "n_seq": A.n_seq, "secs": f"{time.time()-t0:.1f}"})
        print(f"  [{part}] {name:28} R={R:<4} shared={'on ' if shared else 'off'} "
              f"BPB={bpb:.6f} ({time.time()-t0:.0f}s)", flush=True)
        return bpb

    k = cfg.num_experts_per_tok
    free = cell("A", "free_baseline", ALL, k)
    print(f"\n  === A. cost of residency vs resident budget (all {L} layers constrained) ===", flush=True)
    # R must be >= k: below that the router cannot fill its k slots from the resident set,
    # so it measures degraded top-R routing rather than residency. R=k is the tightest valid point.
    for R in [r for r in (4, 8, 16, 32, 64, 128) if r >= k]:
        b = cell("A", f"all_constrained_R{R}", None, R)
        print(f"      R={R:<4} {100*R/E:5.2f}% resident   damage {b-free:+.6f}", flush=True)

    print(f"\n  === B. which layers are worth freeing (measured jointly) ===", flush=True)
    SETS = {"free_none": [], "free_last1": [L-1], "free_last2": [L-2, L-1],
            "free_last4": list(range(L-4, L)), "free_last8": list(range(L-8, L)),
            "free_first2": [0, 1], "free_first2_last2": [0, 1, L-2, L-1],
            "free_first2_last4": [0, 1] + list(range(L-4, L))}
    for R in (8, 32):
        for name, fs in SETS.items():
            b = cell("B", f"{name}_R{R}", fs if fs else None, R)
            print(f"      {name:20} R={R:<3} {len(fs):>2} freed  damage {b-free:+.6f}", flush=True)

    if not has_shared:
        print("\n  === C. skipped: this family has no shared expert ===", flush=True)
        _write(rows, free, OUT, A, E, L, D); return
    print(f"\n  === C. is the shared expert what makes this cheap? ===", flush=True)
    res = {}
    for sh in (True, False):
        f_ = cell("C", f"free_shared_{'on' if sh else 'off'}", ALL, 8, shared=sh)
        c_ = cell("C", f"constrained_shared_{'on' if sh else 'off'}", None, 8, shared=sh)
        res[sh] = (f_, c_, c_ - f_)
        print(f"      shared={'on ' if sh else 'off'}  free {f_:.6f}  constrained {c_:.6f}  "
              f"damage {c_-f_:+.6f}", flush=True)
    dw, dwo = res[True][2], res[False][2]
    print(f"\n      damage with shared {dw:+.6f} | zeroed {dwo:+.6f} | ratio "
          f"{dwo/dw if dw else float('nan'):.2f}x", flush=True)
    print("      ratio >> 1 => architecture-specific; ratio ~ 1 => redundancy, should generalise",
          flush=True)

    _write(rows, free, OUT, A, E, L, D)


def _write(rows, free, OUT, A, E, L, D):
    path = os.path.join(OUT, f"{A.tag}_cost_curve.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# {A.tag} ({A.family}) test-time residency cost, one model load, all cells on "
                    f"identical cached batches. E={E} layers={L}, BPB divisor {D:.7f} on the audited "
                    f"slice re-tokenized to byte-identical text. Part A: price vs resident budget. "
                    f"Part B: joint value of free sets (solo damage is known not to predict this). "
                    f"Part C: shared-expert ablation, damage measured with it live and zeroed. "
                    f"free_baseline BPB {free:.6f}. Producer: analysis/residency/qwen_cost_curve.py"])
        w.writerow(list(rows[0].keys()) + ["damage_vs_free"])
        for r in rows:
            w.writerow(list(r.values()) + [f"{float(r['bpb'])-free:+.6f}"])
    print(f"\n[write] {path}: {len(rows)} cells", flush=True)
    print("=== COST CURVE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
