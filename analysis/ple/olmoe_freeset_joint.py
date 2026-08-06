#!/usr/bin/env python3
"""Which four OLMoE layers should be freed, measured jointly and after the gate-mass fix?

The inherited free set {0,1,14,15} was justified by a per-layer profile in which the first two layers
cost 2.66x a middle layer. That profile was produced with gate_mass=renorm. Re-measured with the
artifact removed (results/ablations/olmoe_gatemass_remeasure.csv), the first two layers cost 0.97x a
middle layer -- no premium at all -- while the last two cost 2.64x and layer 15 alone costs 3.51x.
The evidence for {0,1,14,15} is therefore gone.

It does not follow that the tail-heavy set wins. Solo damage has failed to predict joint free-set
value four times in this programme; Qwen3-30B has the most extreme tail measured (28x the middle) and
still preferred {first 2, last 2} over {last 4}. Freed layers interact, so the sets are compared
jointly here: residency imposed on every layer EXCEPT the set, which is the configuration a training
run would actually use.

All candidates free exactly 4 of 16 layers, so the comparison is matched on resident-memory budget.

Scored on disjoint blocks of the held-out slice, and the quantity reported for the contested pair is
the PER-BLOCK PAIRED difference: block-to-block difficulty is common to both arms and cancels, which
is far more sensitive than differencing two means. §5B of the Qwen write-up turned on a 0.0025 BPB
gap, so an error bar is not optional here.

    olmoe_freeset_joint.py --block 16 --blocks 3
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                            # noqa: E402
from olmoe_paths import DATA_DIR                                   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                        # noqa: E402

# Grouped by how many layers are freed, because freeing a layer is not free: a constrained layer
# holds R=8 of 64 experts and a freed layer holds all 64, so resident slots = 8*(16-f) + 64*f. That
# is 128 slots at f=0, 240 at f=2, 352 at f=4, 464 at f=6 -- the f=4 recipe costs 2.75x the memory of
# f=0. Comparing sets only within a fixed f answers "which layers", but the choice of f is the actual
# memory/quality trade, so all three budgets are measured on the same blocks here.
SETS = {
    # f=0
    "f0_all_constrained":    [],
    # f=2
    "f2_last2_14_15":        [14, 15],         # the tail peak alone
    "f2_first2_0_1":         [0, 1],           # head-only, now that the head premium is gone
    # f=4
    "f4_inherited_0_1_14_15": [0, 1, 14, 15],  # the recipe the programme has been using
    "f4_tail4_12_13_14_15":  [12, 13, 14, 15], # what the corrected solo profile points at
    "f4_top4solo_10_12_14_15": [10, 12, 14, 15],  # four highest-solo-damage layers, non-contiguous
    "f4_head4_0_1_2_3":      [0, 1, 2, 3],     # control: the discredited hypothesis, taken further
    # f=6
    "f6_10_11_12_13_14_15":  [10, 11, 12, 13, 14, 15],  # is there anything left after f=4?
}


def resident_slots(free, L=16, R=8, E=64):
    """Expert slots held resident across the model; the currency the memory saving is denominated in."""
    return R * (L - len(free)) + E * len(free)


@torch.no_grad()
def bpb(model, ids, divisor):
    tot = ntok = 0
    for i in range(ids.shape[0]):
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
    ap.add_argument("--block", type=int, default=16, help="sequences per block")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--R", type=int, default=8)
    A = ap.parse_args()

    D = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))["divisor_D"]
    allids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt", weights_only=False)
    need = A.block * A.blocks
    if allids.shape[0] < need:
        sys.exit(f"slice has {allids.shape[0]} sequences, need {need} for "
                 f"{A.blocks} disjoint blocks of {A.block}")
    blocks = [allids[j * A.block:(j + 1) * A.block] for j in range(A.blocks)]

    model, _ = RES.load_model()
    L = model.config.num_hidden_layers
    print(f"  OLMoE E={model.config.num_experts} k={model.config.num_experts_per_tok} layers={L} "
          f"R={A.R} gate_mass=preserve  {A.blocks} disjoint blocks x {A.block} seq", flush=True)

    # free baseline per block: residency off entirely, so damage is measured against the same data
    base = []
    RES._CFG.update(on=True, R=A.R, evict="min_logit", collect_telem=False, gate_mass="preserve")
    RES.set_free_layers(list(range(L)))
    for j, blk in enumerate(blocks):
        base.append(bpb(model, blk, D))
    print(f"  free baseline per block: " + "  ".join(f"{b:.6f}" for b in base), flush=True)

    dmg = {}
    for tag, fs in SETS.items():
        RES._CFG.update(on=True, R=A.R, evict="min_logit", collect_telem=False, gate_mass="preserve")
        RES.set_free_layers(fs if fs else [])
        per = [bpb(model, blk, D) - base[j] for j, blk in enumerate(blocks)]
        dmg[tag] = per
        m = sum(per) / len(per)
        spread = max(per) - min(per)
        sl = resident_slots(fs, L, A.R, model.config.num_experts)
        print(f"  {tag:26} slots {sl:4} ({sl/resident_slots([], L, A.R, model.config.num_experts):.2f}x)"
              f"  damage {m:+.6f}   per-block " + " ".join(f"{p:+.6f}" for p in per)
              + f"   spread {spread:.6f}", flush=True)

    a, b = "f4_inherited_0_1_14_15", "f4_tail4_12_13_14_15"
    paired = [dmg[b][j] - dmg[a][j] for j in range(A.blocks)]
    pm = sum(paired) / len(paired)
    print(f"\n  === contested pair, paired per block ({b} minus {a}) ===")
    print(f"  per-block " + "  ".join(f"{p:+.6f}" for p in paired))
    print(f"  mean {pm:+.6f}   (negative => the tail set is better; |mean| vs spread "
          f"{max(paired)-min(paired):.6f} says whether it is an effect)")

    path = os.path.join(ABLATIONS, "olmoe_freeset_joint.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# OLMoE-1B-7B: joint free-set comparison at R={A.R} of 64, gate_mass=preserve "
                    f"(after the artifact fix). Residency imposed on every layer EXCEPT the named "
                    f"set; all sets free exactly 4 of 16 layers, so the comparison is matched on "
                    f"resident-memory budget. Damage is BPB above a residency-off baseline scored on "
                    f"the same block. {A.blocks} disjoint blocks of {A.block} sequences give the "
                    f"spread. Producer: analysis/ple/olmoe_freeset_joint.py"])
        w.writerow(["set", "n_free", "free_layers", "resident_slots", "slots_vs_f0"]
                   + [f"damage_block{j}" for j in range(A.blocks)] + ["damage_mean", "spread"])
        base_slots = resident_slots([], L, A.R, model.config.num_experts)
        for tag, fs in SETS.items():
            per = dmg[tag]
            sl = resident_slots(fs, L, A.R, model.config.num_experts)
            w.writerow([tag, len(fs), "|".join(map(str, fs)), sl, f"{sl/base_slots:.3f}"]
                       + [f"{p:+.6f}" for p in per]
                       + [f"{sum(per)/len(per):+.6f}", f"{max(per)-min(per):.6f}"])
        w.writerow([])
        w.writerow(["free_baseline_bpb", ""] + [f"{x:.6f}" for x in base])
    print(f"\n[write] {path}", flush=True)
    print("=== FREESET JOINT COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
