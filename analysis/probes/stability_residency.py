#!/usr/bin/env python3
"""PART B of the stability probe: residency-frequency join (temporal cells only).

From each temporal cell's PROBE=1 router log (logits + resident mask on the fixed eval batch),
per MoE layer, per expert:
  resident_frac  = fraction of tokens the expert is in the resident (cached/served) set  [from mask]
  selected_frac  = fraction of tokens the expert is in the token's UNCONSTRAINED top-k demand
                   (raw router argmax-k, ignoring residency)  [what a plain-MoE router would pick]
The resident-vs-selected gap is the temporal mechanism's signature: experts kept resident beyond
their instantaneous demand (sticky) vs demanded-but-evicted.

Layer index is emitted as (log layer_number - 1) to match Part A's 0-indexed decoder.layers.N,
so router row norms (Part A, component=router_row) join to resident_frac on (run, layer, expert).

Output: results/ablations/stability_residency.csv  columns: run,layer,expert,resident_frac,selected_frac
"""
import os, csv, torch

ROOT = "/workspace/FLAME-MoE"
RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/stability_residency.csv")

# Part-A label -> router_log.pt path
CELLS = [
    ("g1_temporal",          "flame38m_g1_temporal"),
    ("g3_temporal",          "flame38m_g3_temporal"),
    ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
    ("temporal_fine_1e19",   "temporal_fine_g3_1e19"),
]


def run_rows(label, run):
    log = os.path.join(RUNS, run, "router_log.pt")
    d = torch.load(log, map_location="cpu", weights_only=False)
    assert d["temporal"], f"{run}: expected temporal log"
    rows = []
    for ln in sorted(d["layers"]):
        rec = d["layers"][ln]
        lg = rec["logits"].float()          # [S, B, E]
        mask = rec["mask"]                   # [S, B, E] bool  (resident set)
        k = rec["k"]
        S, B, E = lg.shape
        resident_frac = mask.float().mean((0, 1))           # [E]
        # unconstrained demand: per-token top-k of raw logits
        sel = torch.zeros_like(mask)
        sel.scatter_(-1, lg.topk(k, dim=-1).indices, True)
        selected_frac = sel.float().mean((0, 1))            # [E]
        layer = ln - 1                                       # -> Part A 0-indexed decoder.layers.N
        for e in range(E):
            rows.append([label, layer, e, f"{resident_frac[e]:.6f}", f"{selected_frac[e]:.6f}"])
    return rows, S, B


def main():
    all_rows = []
    for label, run in CELLS:
        log = os.path.join(RUNS, run, "router_log.pt")
        if not os.path.exists(log):
            print(f"[skip] {label}: no router_log.pt at {run}")
            continue
        rows, S, B = run_rows(label, run)
        all_rows += rows
        nl = len({r[1] for r in rows})
        print(f"[ok] {label}: {len(rows)} rows, {nl} layers, batch {S}x{B}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "layer", "expert", "resident_frac", "selected_frac"])
        w.writerows(all_rows)
    print(f"[write] {OUT}: {len(all_rows)} rows")


if __name__ == "__main__":
    main()
