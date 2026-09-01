#!/usr/bin/env python3
"""Fair-usage metrics for the CoSMoEs sweep (user note 2026-09-01): BlES can shrink switching
by OVER-USING a few experts and neglecting the rest, so switching alone is not the story.
Per run and layer, from router_log.pt (one fixed batch, raw logits pre-mask):
  swaps_tl   consecutive-token top-k set replacements / (k) per token-layer (their operating cost)
  eff_exp    exp(entropy(marginal top-k load)) -- effective number of experts in use
  max_share  largest single expert's share of top-k selections (over-use)
  neglected  experts with share < 0.1/E (neglect), out of E
  union_seq  mean distinct experts a sequence touches
Appends one row per (run, layer) + a run-mean row to results/phase0/cosmoes_metrics.csv.
"""
import os, sys, csv, numpy as np, torch
run = sys.argv[1]
d = torch.load(f"results/phase0/runs/{run}/router_log.pt", map_location="cpu")
rows = []
for ln, r in d["layers"].items():
    lg = r["logits"].float().numpy(); k = r["k"]
    if lg.ndim == 2: lg = lg[:, None, :]          # [T,E] -> [T,1,E]
    if lg.ndim == 3 and lg.shape[0] < lg.shape[1]: lg = lg.transpose(1,0,2)  # ensure [T,B,E]
    T, B, E = lg.shape
    idx = np.argpartition(-lg, k-1, axis=-1)[..., :k]
    sel = np.zeros((T,B,E), bool); np.put_along_axis(sel, idx, True, axis=-1)
    swaps = float(np.abs(sel[1:].astype(int)-sel[:-1].astype(int)).sum()) / 2 / (B*(T-1)) / k
    load = sel.sum((0,1)).astype(float); p = load/load.sum()
    nz = p[p>0]; eff = float(np.exp(-(nz*np.log(nz)).sum()))
    rows.append(dict(run=run, layer=ln, swaps_tl=round(swaps,4), eff_exp=round(eff,1),
                     max_share=round(float(p.max()),4), neglected=int((p < 0.1/E).sum()), E=E,
                     union_seq=round(float(sel.any(0).sum(-1).mean()),1)))
mean = {k2: round(float(np.mean([r[k2] for r in rows])),4) for k2 in ("swaps_tl","eff_exp","max_share","neglected","union_seq")}
rows.append(dict(run=run, layer="MEAN", E=rows[0]["E"], **mean))
out = "results/phase0/cosmoes_metrics.csv"; new = not os.path.exists(out)
with open(out,"a",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run","layer","E","swaps_tl","eff_exp","max_share","neglected","union_seq"])
    if new: w.writeheader()
    for r in rows: w.writerow(r)
print(f"[metrics] {run}: " + " ".join(f"{k}={v}" for k,v in mean.items()))
