#!/usr/bin/env python3
"""Joint damage of freeing a SUBSET of layers, training-free, vs the additive prediction.

The single-layer ablation gives damage(i) for each layer alone. It does not say what freeing a set
S costs, because the constraint turned out to be mildly super-additive (singles sum to 1.861 against
a full-residency damage of 2.078, ratio 0.896). This measures the sets directly.

  damage(constrain all but S) = BPB(free S) - BPB(free everything)

and compares it to the additive prediction full_damage - sum_{i in S} damage(i). The gap is the
interaction term, and it is what decides whether a third freed layer is worth +131% resident memory
before a 50-minute training cell is spent finding out.
"""
import csv, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES
from olmoe_paths import DATA_DIR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

D = json.load(open(os.path.join(DATA_DIR, "bpb_slice_meta.json")))["divisor_D"]
solo = {}
for r in csv.DictReader(open(os.path.join(ABLATIONS, "layer_damage.csv"))):
    if r["layer"].isdigit():
        solo[int(r["layer"])] = float(r["damage_bpb"])
    elif r["layer"] == "all constrained":
        FULL = float(r["damage_bpb"])

model, _ = RES.load_model()
model.eval()
L = model.config.num_hidden_layers
ids = torch.load(os.path.join(DATA_DIR, "bpb_slice_ids.pt"))
sub = ids[torch.linspace(0, ids.shape[0] - 1, 256).long()].long()


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


RES.enable_residency(R=8)
RES.set_free_layers(range(L))
b_free = ev()
print(f"[joint] all free  BPB={b_free:.6f}", flush=True)

SETS = [([], "none (full residency)"), ([0], "{0}"), ([1], "{1}"), ([0, 1], "{0,1}"),
        ([0, 1, 2], "{0,1,2}"), ([0, 1, 15], "{0,1,15}"), ([0, 1, 14, 15], "{0,1,14,15}")]
rows = []
for S, name in SETS:
    RES.set_free_layers(S)
    b = ev()
    dmg = b - b_free
    pred = FULL - sum(solo[i] for i in S)
    slots = (L - len(S)) * 8 + len(S) * 64
    rows.append({"free_set": name, "n_free": len(S), "bpb": round(b, 6),
                 "damage_bpb": round(dmg, 6), "additive_prediction": round(pred, 6),
                 "interaction": round(dmg - pred, 6),
                 "resident_slots": slots, "memory_vs_full_residency_pct": round(slots / 128 * 100 - 100, 1)})
    print(f"[joint] free {name:12s} BPB={b:.6f} damage={dmg:.6f} "
          f"additive_pred={pred:.6f} interaction={dmg-pred:+.6f} mem +{slots/128*100-100:.1f}%",
          flush=True)
RES.set_free_layers(None)

path = os.path.join(ABLATIONS, "joint_free.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote", path)
