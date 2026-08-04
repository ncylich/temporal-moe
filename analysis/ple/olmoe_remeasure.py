#!/usr/bin/env python3
"""Re-measure OLMoE residency with gate mass preserved, and quantify what the artifact cost us.

Every OLMoE residency number in this repository was produced with `gate_mass="renorm"`: masking to
-inf then taking the softmax over the R residents, which on a `norm_topk_prob=False` model raises the
gate mass from ~0.40 to 1.0 and multiplies every MoE block output by ~2.5x across 16 layers. That is
an activation-scale change riding on top of the routing constraint, and it is ~91% of the measured
damage. Qwen is structurally immune (norm_topk_prob=True makes the two identical), so every
cross-model ratio this program has published is inflated by roughly an order of magnitude.

Three things, in one model load:

    anchors      free, and constrained at R=8, under BOTH gate-mass policies. The "renorm" arm should
                 reproduce the published +2.078 -- if it does not, the diagnosis is wrong and nothing
                 below should be trusted.
    per-layer    section 2's U-shaped profile, one layer constrained at a time, re-run under
                 "preserve". The published profile was produced under the artifact, so the shape
                 itself may be an artifact: gate-mass inflation compounds with the number of
                 constrained layers, which is exactly the axis that profile varies.
    curve        R in {8,16,32,64} under "preserve", giving OLMoE a cost curve comparable to the
                 Qwen ones rather than a single point.

Not covered, and it is the expensive gap: every *trained* OLMoE cell adapted to the artifact, so the
recovery percentages describe adaptation to an intervention we did not intend. Re-running those is a
training job, not a measurement.

    olmoe_remeasure.py --n-seq 16
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
from olmoe_paths import DATA_DIR                                   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                        # noqa: E402


@torch.no_grad()
def bpb(model, ids, divisor, n_seq):
    tot = ntok = 0
    for i in range(n_seq):
        b = ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel()
        del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--skip-profile", action="store_true")
    A = ap.parse_args()

    meta = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))
    D = meta["divisor_D"]
    model, tok = RES.load_model()
    L = model.config.num_hidden_layers
    ids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt", weights_only=False)[: A.n_seq]
    ALL = list(range(L))
    print(f"  OLMoE E={model.config.num_experts} k={model.config.num_experts_per_tok} layers={L} "
          f"norm_topk_prob={model.config.norm_topk_prob} divisor={D:.7f}", flush=True)

    rows = []
    def cell(name, free_set, R, gm):
        RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=False, gate_mass=gm)
        RES.set_free_layers(free_set)
        t0 = time.time()
        v = bpb(model, ids, D, A.n_seq)
        rows.append({"cell": name, "R": R, "gate_mass": gm, "bpb": f"{v:.6f}",
                     "n_seq": A.n_seq, "secs": f"{time.time()-t0:.1f}"})
        print(f"  {name:26} R={R:<4} gate_mass={gm:8} BPB={v:.6f} ({time.time()-t0:.0f}s)", flush=True)
        return v

    free = cell("free_baseline", ALL, 8, "preserve")
    d_ren = cell("constrained_R8", None, 8, "renorm") - free
    d_pre = cell("constrained_R8", None, 8, "preserve") - free
    print(f"\n  === the artifact ===", flush=True)
    print(f"  damage as published (renorm)   {d_ren:+.6f}   [published value: +2.078]", flush=True)
    print(f"  damage with gate mass preserved {d_pre:+.6f}", flush=True)
    print(f"  artifact share of published damage: {100*(d_ren-d_pre)/d_ren:.1f}%", flush=True)

    print(f"\n  === cost curve, gate mass preserved ===", flush=True)
    for R in (8, 16, 32, 64):
        v = cell(f"all_constrained_R{R}", None, R, "preserve")
        print(f"      R={R:<3} {100*R/model.config.num_experts:5.1f}% resident   damage {v-free:+.6f}",
              flush=True)

    if not A.skip_profile:
        print(f"\n  === per-layer profile, gate mass preserved (section 2 re-run) ===", flush=True)
        for li in range(L):
            cell(f"solo_L{li:02d}", [x for x in ALL if x != li], 8, "preserve")

    path = os.path.join(ABLATIONS, "olmoe_gatemass_remeasure.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# OLMoE-1B-7B residency re-measured with gate mass preserved. The published "
                    f"numbers used gate_mass=renorm, which on a norm_topk_prob=False model raises the "
                    f"top-k gate mass from ~0.40 to 1.0 and scales every MoE block output by ~2.5x "
                    f"over 16 layers. 'renorm' rows reproduce the published intervention; 'preserve' "
                    f"rows select from the masked distribution but weight from the unmasked one. "
                    f"free_baseline {free:.6f}. Producer: analysis/ple/olmoe_remeasure.py"])
        w.writerow(list(rows[0].keys()) + ["damage_vs_free"])
        for r in rows:
            w.writerow(list(r.values()) + [f"{float(r['bpb'])-free:+.6f}"])
    print(f"\n[write] {path}: {len(rows)} cells", flush=True)
    print("=== OLMOE REMEASURE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
