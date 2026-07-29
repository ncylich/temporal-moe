#!/usr/bin/env python3
"""Per-layer residency damage: constrain exactly ONE MoE layer at a time, measure the BPB cost.

No training anywhere. The base model is evaluated 18 times:
    all free        16 layers unconstrained -> should reproduce the 0.6727 reference
    layer i only    layer i under residency R=8, other 15 free, for i = 0..15
    all constrained 16 layers constrained   -> should reproduce the 2.7507 impose point

Deliverable: damage_bpb = BPB(layer i only) - BPB(all free), the per-layer share of residency cost.
This measures what the free-the-early-layers runs could only assume: whether the damage is
concentrated in particular layers or spread across depth.

It also settles two things:
  SUPERADDITIVITY  sum_i damage_i vs the full-residency damage. Far below means the layers interact
    and single-layer numbers cannot be summed to predict a subset.
  WHICH TO FREE    freeing a layer costs the same memory wherever it sits (64 experts vs 8), so the
    right layers to free are simply the most damaging ones, at whatever depth.

SUBSET MODE (--sets): instead of constraining one layer at a time, leave a SUBSET unconstrained and
compare the measured damage against the additive prediction from the single-layer profile. The gap
is the interaction term. This is what showed that {0,1,2} and {0,1,15} have near-identical additive
predictions (0.1408 solo damage each for layers 2 and 15) but recover 0.573 and 0.409 respectively:
layer 15's damage is largely redundant with what freeing layers 0-1 already fixes.
"""
import argparse, csv, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES
from olmoe_paths import DATA_DIR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

BASE_REF, IMPOSE_REF = 0.6727, 2.7507

ap = argparse.ArgumentParser()
ap.add_argument("--eval-n", type=int, default=256)
ap.add_argument("--R", type=int, default=8)
A = ap.parse_args()

D = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))["divisor_D"]
model, _ = RES.load_model()
model.eval()
L = model.config.num_hidden_layers
ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
sub = ids[torch.linspace(0, ids.shape[0] - 1, A.eval_n).long()].long()


def ev():
    tot = n = 0
    with torch.no_grad():
        for i in range(sub.shape[0]):
            x = sub[i:i + 1].to("cuda")
            lg = model(x).logits.float()
            tot += float(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="sum"))
            n += x[:, 1:].numel()
    return (tot / n) / D


RES.enable_residency(R=A.R)
RES.set_free_layers(range(L))
b_free = ev()
print(f"[abl] all free        BPB={b_free:.6f}   (published base {BASE_REF})", flush=True)
RES.set_free_layers([])
b_all = ev()
print(f"[abl] all constrained BPB={b_all:.6f}   (published impose {IMPOSE_REF})", flush=True)

rows = [{"layer": "none (all free)", "bpb": round(b_free, 6), "damage_bpb": 0.0},
        {"layer": "all constrained", "bpb": round(b_all, 6),
         "damage_bpb": round(b_all - b_free, 6)}]
dmg = []
for i in range(L):
    RES.set_free_layers([j for j in range(L) if j != i])
    b = ev()
    d = b - b_free
    dmg.append(d)
    rows.append({"layer": i, "bpb": round(b, 6), "damage_bpb": round(d, 6)})
    print(f"[abl] layer {i:2d} only    BPB={b:.6f}   damage={d:+.6f}", flush=True)
RES.set_free_layers(None)

tot_d, full_d = sum(dmg), b_all - b_free
print(f"\n[abl] sum {tot_d:.6f} vs full-residency {full_d:.6f}  ratio {tot_d/full_d:.3f}", flush=True)
order = sorted(range(L), key=lambda i: -dmg[i])
print(f"[abl] most damaging: {order[:4]}   least: {order[-4:]}", flush=True)

path = os.path.join(ABLATIONS, "ple_layer_damage.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer", "bpb", "damage_bpb"])
    w.writeheader()
    w.writerows(rows)
print("wrote", path)

# The figure is drawn by `report.py figure` from the consolidated CSV, because matplotlib is not
# installed in the venv that runs the model.
