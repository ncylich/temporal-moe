#!/usr/bin/env python3
"""Bake-off pre-step: calibrate eval-noise sigma. Re-eval the BASE model (no training) on 3 DISJOINT
256-pack subsamples of the audited BPB slice, at BOTH free-routing (base regime, BPB~0.673) and the
R=8 residency regime (impose regime, the condition every arm is scored under). Report per-subsample
BPB + mean/std so the bake-off CSV header can state whether an inter-arm gap is noise or real.
Writes data/eval_sigma.json."""
import sys, json, torch
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES

OUT = "/workspace/olmoe-adapt/data"
D = json.load(open(f"{OUT}/bpb_slice_meta.json"))["divisor_D"]
model, tok = RES.load_model()
ids = torch.load(f"{OUT}/bpb_slice_ids.pt")
N = ids.shape[0]

# 3 disjoint thirds; within each take a 256-pack linspace -> representative, non-overlapping
subs = []
for t in range(3):
    lo, hi = t * (N // 3), (t + 1) * (N // 3) - 1
    idx = torch.linspace(lo, hi, 256).long()
    subs.append(ids[idx].to("cuda").long())


def bpb(sub):
    tot = n = 0
    with torch.no_grad():
        for i in range(sub.shape[0]):
            x = sub[i:i + 1]; out = model(x).logits.float()
            l = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                  x[:, 1:].reshape(-1), reduction="sum")
            tot += l.item(); n += x[:, 1:].numel()
    return (tot / n) / D


res = {}
for regime, R in [("free", None), ("R8", 8)]:
    if R is None:
        RES.disable_residency()
    else:
        RES.enable_residency(R=R)
    vals = [bpb(s) for s in subs]
    m = sum(vals) / 3
    sd = (sum((v - m) ** 2 for v in vals) / 3) ** 0.5
    res[regime] = {"per_subsample": vals, "mean": m, "std": sd}
    print(f"[sigma] {regime}: {[round(v,4) for v in vals]} mean={m:.4f} std={sd:.4f}", flush=True)

json.dump({"divisor_D": D, "subsample_packs": 256, "n_subsamples": 3,
           "note": "disjoint thirds of audited slice; std = eval-noise sigma for inter-arm gaps",
           **res}, open(f"{OUT}/eval_sigma.json", "w"), indent=1)
print("[sigma] wrote data/eval_sigma.json", flush=True)
