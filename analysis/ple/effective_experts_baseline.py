#!/usr/bin/env python3
"""Per-layer effective expert count of the untrained OLMoE, as the reference every later cell moves from.

Two senses, both per layer (definitions in `residency.effective_experts`):

    eff_load   1 / sum_e p_e^2 over the dispatch distribution -- how many experts carry the corpus.
               1 is total collapse, E is perfect balance. This is what the aux loss holds up.
    eff_tok    exp(mean token-wise routing entropy) -- how many experts one token spreads over.

**Two baselines, not one.** The stock model has never seen residency, so there are two different
reference points and they answer different questions:

    free      no residency at all, the model as released. What OLMoE's routing looks like natively.
    imposed   residency R=k applied at evaluation only, untrained. What the constraint does to
              routing BEFORE any adaptation repairs it.

Reporting one of these as "the baseline" is how a comparison quietly becomes unmatched: an adapted
cell is trained under the constraint, so `imposed` is its like-for-like ancestor, while `free` is the
ceiling the constraint moved away from. Both are written, labelled by `regime`.

No training, no gradients. One forward per regime.

    effective_experts_baseline.py
    effective_experts_baseline.py --seq 1024 --batch 4
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                        # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                    # noqa: E402

OUT = os.path.join(ABLATIONS, "effective_experts_baseline.csv")
HEADER = ["regime", "layer", "E", "k", "eff_load", "eff_load_frac_of_E", "eff_tok", "n_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()

    model, tok = RES.load_model()
    E = model.config.num_experts
    k = model.config.num_experts_per_tok
    os.environ["OLMOE_TOPK"] = str(k)
    # The SAME audited held-out slice the training-time log scores on. This first used random token
    # ids, which made the baseline incomparable to every number it exists to be compared against: a
    # router fed uniform-random tokens routes nothing like one fed text, and the effective expert
    # count is a property of the routing, not of the architecture.
    from olmoe_paths import DATA_DIR
    bpb_ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
    ids = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, A.batch).long()].to("cuda").long()
    if ids.shape[1] > A.seq:
        ids = ids[:, :A.seq]
    n_tok = ids.shape[0] * ids.shape[1]
    print(f"  scoring the audited slice: {ids.shape[0]} x {ids.shape[1]} = {n_tok} tokens", flush=True)

    rows = []
    for regime in ("free", "imposed"):
        if regime == "free":
            # Every layer treated as unconstrained, so the mask is never applied: the released model.
            RES.enable_residency(R=k)
            RES.set_free_layers(list(range(model.config.num_hidden_layers)))
        else:
            RES.enable_residency(R=k)
            RES.set_free_layers(None)
        out = model(ids, output_router_logits=True)
        for r in RES.effective_experts(out.router_logits, ids.shape[0], ids.shape[1], k):
            rows.append([regime, r["layer"], E, k, f"{r['eff_load']:.3f}",
                         f"{r['eff_load'] / E:.4f}", f"{r['eff_tok']:.3f}", n_tok])
        got = [r["eff_load"] for r in RES.effective_experts(out.router_logits, ids.shape[0], ids.shape[1], k)]
        print(f"  {regime:8} eff_load per layer: min {min(got):.1f}  median "
              f"{sorted(got)[len(got) // 2]:.1f}  max {max(got):.1f}   (E={E}, k={k})", flush=True)

    with open(A.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["# Per-layer effective expert count of the UNTRAINED OLMoE, on the audited held-out slice (the same tokens the training-time log scores). regime=free is the "
                    "released model with no residency; regime=imposed is residency R=k applied at "
                    "eval only, untrained. eff_load = 1/sum(p^2) over dispatch, range 1..E, higher "
                    "is more balanced. eff_tok = exp(mean token routing entropy). Producer: "
                    "analysis/ple/effective_experts_baseline.py"])
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[write] {A.out}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
