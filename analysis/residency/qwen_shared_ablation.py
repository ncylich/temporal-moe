#!/usr/bin/env python3
"""Is Qwen's always-resident shared expert what makes residency cheap here?

Imposing R=k residency costs Qwen3.5 ~0.055 BPB and costs OLMoE 2.078 -- a ~40x difference for what
is nominally the same rule. Two candidate explanations, with opposite implications for scaling:

    shared expert   Qwen runs one expert on every token unconditionally, outside the router. No
                    matter how badly the resident set is chosen, that path is intact. OLMoE has no
                    such path. If this is the mechanism, the result is about Qwen's architecture and
                    transfers only to models with a shared expert.
    redundancy      256 experts at top-8 leaves far more substitutable capacity than 64 at top-8.
                    If this is the mechanism, the result is about expert count and should improve
                    with scale generally.

The ablation separates them. Damage is measured twice -- with the shared expert live, and with its
contribution zeroed -- so the comparison is a difference of differences and does not depend on how
much the shared expert is worth in absolute terms:

    damage_with    = BPB(constrained, shared on)  - BPB(free, shared on)
    damage_without = BPB(constrained, shared off) - BPB(free, shared off)

If damage_without >> damage_with, the shared expert is absorbing the constraint and the headline is
architecture-specific. If they are similar, redundancy is doing the work and the result generalises.

Zeroing rather than deleting: the block still runs the shared path and discards it, so the routed
path sees exactly the tokens it would otherwise see.
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
import residency_qwen as RQ                                        # noqa: E402
from qwen_sweep import batches, score                              # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeSparseMoeBlock,
)

DATA = "/workspace/qwen35-adapt/data"
OUT = "/workspace/qwen35-adapt/results"
SHARED = {"on": True}
_orig = None


def _block_no_shared(self, hidden_states):
    b, s, h = hidden_states.shape
    self.gate._resid_shape = (b, s)
    x = hidden_states.view(-1, h)
    _, w, idx = self.gate(x)
    out = self.experts(x, idx, w)
    if SHARED["on"]:
        sh = self.shared_expert(x)
        out = out + torch.sigmoid(self.shared_expert_gate(x)) * sh
    return out.reshape(b, s, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=16)
    A = ap.parse_args()
    meta = json.load(open(f"{DATA}/bpb_slice_meta_qwen.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model()
    L = model.config.num_hidden_layers
    global _orig
    _orig = Qwen3_5MoeSparseMoeBlock.forward
    Qwen3_5MoeSparseMoeBlock.forward = _block_no_shared
    bl = batches(A.n_seq, 1)
    ALL = list(range(L))

    rows = []
    for shared in (True, False):
        SHARED["on"] = shared
        free_bpb = score(model, bl, D, ALL, 8)[0]
        con_bpb = score(model, bl, D, None, 8)[0]
        rows.append({"shared_expert": "on" if shared else "zeroed",
                     "bpb_free": f"{free_bpb:.6f}", "bpb_constrained": f"{con_bpb:.6f}",
                     "damage": f"{con_bpb - free_bpb:.6f}", "n_seq": A.n_seq})
        print(f"  shared={'on ' if shared else 'off'}  free={free_bpb:.6f}  "
              f"constrained={con_bpb:.6f}  damage={con_bpb-free_bpb:+.6f}", flush=True)

    dw, dwo = float(rows[0]["damage"]), float(rows[1]["damage"])
    print(f"\n  damage with shared expert   {dw:+.6f}")
    print(f"  damage with it zeroed       {dwo:+.6f}   ratio {dwo/dw if dw else float('nan'):.2f}x")
    print("  ratio >> 1 => the shared expert absorbs the constraint (architecture-specific)")
    print("  ratio ~ 1  => expert redundancy is doing the work (should generalise with scale)")

    path = os.path.join(OUT, "qwen35_shared_expert_ablation.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["# Does Qwen's always-resident shared expert explain why residency costs it "
                    "~0.055 BPB where OLMoE pays 2.078? Damage = constrained minus free, measured "
                    "twice: shared expert live, and its contribution zeroed. A difference of "
                    "differences, so it does not depend on the shared expert's absolute value. "
                    "R=8 of 256 routed experts. Producer: analysis/residency/qwen_shared_ablation.py"])
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow(list(r.values()))
    print(f"\n[write] {path}", flush=True)


if __name__ == "__main__":
    main()
