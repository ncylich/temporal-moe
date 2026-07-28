#!/usr/bin/env python3
"""PART A of the stability probe: expert weight statics from checkpoint reads only (no forward pass).

For every MoE layer / routed expert / expert weight matrix: Frobenius norm, max abs entry, excess
kurtosis of the flattened entries. Same for the shared expert, the dense model's FFN (1-expert
reference), and the per-expert row norm of the router weight.

Output: results/ablations/stability_weights.csv
  columns: run,layer,component,expert,matrix,l2,maxabs,kurtosis
  component in {routed, shared, dense_ffn, router_row};  expert empty for shared/dense_ffn.

Param-name mapping (Megatron TEGroupedMLP), verified against the checkpoints:
  routed  : decoder.layers.L.mlp.experts.experts.linear_fc{1,2}.weight   [E, out, in]  (slice per expert)
  shared  : decoder.layers.L.mlp.shared_experts.linear_fc{1,2}.weight    [out, in]
  router  : decoder.layers.L.mlp.router.weight                           [E, H]  (row = per-expert)
  dense   : decoder.layers.L.mlp.linear_fc{1,2}.weight                   [out, in]  (non-MoE layers)
"""
import os, sys, csv, re, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_read

ROOT = "/workspace/FLAME-MoE"
RUNS_DIR = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/stability_weights.csv")

# label -> run dir (seed-1 primary for 1e18; the four 1e19 cells)
RUNS = [
    ("dense_local",     "flame38m_dense_local"),
    ("g1_moe",          "flame38m_g1_moe"),
    ("g1_temporal",     "flame38m_g1_temporal"),
    ("g3_moe",          "flame38m_g3_moe"),
    ("g3_temporal",     "flame38m_g3_temporal"),
    ("dense_1e19",      "dense_1e19"),
    ("moe_coarse_1e19", "moe_coarse_1e19"),
    ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
    ("temporal_fine_1e19",   "temporal_fine_g3_1e19"),
]


def stats(t):
    """L2 (Frobenius) norm, max abs entry, excess (Fisher) kurtosis of a flattened tensor."""
    x = t.detach().reshape(-1).to(torch.float64)
    l2 = torch.linalg.vector_norm(x).item()
    maxabs = x.abs().max().item()
    m = x.mean()
    d = x - m
    var = (d * d).mean()
    m4 = (d ** 4).mean()
    kurt = (m4 / (var * var) - 3.0).item() if var > 0 else float("nan")
    return l2, maxabs, kurt


def layer_of(key):
    m = re.search(r"decoder\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else -1


def run_rows(label, run):
    ckpt = os.path.join(RUNS_DIR, run, "ckpt")
    ipath = ckpt_read.iter_dir(ckpt)
    meta = ckpt_read.weight_keys(ckpt_read.FileSystemReader(ipath))

    want = [k for k in meta if re.search(
        r"mlp\.(experts\.experts\.linear_fc[12]\.weight|shared_experts\.linear_fc[12]\.weight"
        r"|router\.weight|linear_fc[12]\.weight)$", k)]
    sd = ckpt_read.load(ipath, want)

    rows = []
    for k in sorted(want, key=lambda k: (layer_of(k), k)):
        L = layer_of(k)
        t = sd[k]
        fc = "fc1" if "linear_fc1" in k else ("fc2" if "linear_fc2" in k else "router")
        if "experts.experts" in k:                       # routed: [E, out, in]
            for e in range(t.shape[0]):
                l2, mx, ku = stats(t[e])
                rows.append([label, L, "routed", e, fc, l2, mx, ku])
        elif "shared_experts" in k:                      # shared: [out, in]
            l2, mx, ku = stats(t)
            rows.append([label, L, "shared", "", fc, l2, mx, ku])
        elif "router.weight" in k:                       # router rows: [E, H]
            for e in range(t.shape[0]):
                l2, mx, ku = stats(t[e])
                rows.append([label, L, "router_row", e, "router", l2, mx, ku])
        elif t.dim() == 3:                               # dense FFN stacked across layers: [L, out, in]
            for li in range(t.shape[0]):
                l2, mx, ku = stats(t[li])
                rows.append([label, li, "dense_ffn", "", fc, l2, mx, ku])
        else:                                            # dense FFN per-layer (e.g. MoE layer 0): [out, in]
            l2, mx, ku = stats(t)
            rows.append([label, L, "dense_ffn", "", fc, l2, mx, ku])
    return rows


def main():
    all_rows = []
    for label, run in RUNS:
        if not os.path.isdir(os.path.join(RUNS_DIR, run, "ckpt")):
            print(f"[skip] {label}: no checkpoint at {run}", file=sys.stderr)
            continue
        r = run_rows(label, run)
        all_rows += r
        nr = sum(1 for x in r if x[2] == "routed")
        print(f"[ok] {label}: {len(r)} rows ({nr} routed)", file=sys.stderr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "layer", "component", "expert", "matrix", "l2", "maxabs", "kurtosis"])
        for row in all_rows:
            w.writerow(row[:5] + [f"{row[5]:.6g}", f"{row[6]:.6g}", f"{row[7]:.6g}"])
    print(f"[write] {OUT}: {len(all_rows)} rows", file=sys.stderr)

    # ---- sanity checks ----
    torch.manual_seed(0)
    _, _, ku0 = stats(torch.randn(4096, 256))
    print(f"SANITY1 randn excess kurtosis = {ku0:+.4f}  (|.|<0.05 -> {abs(ku0) < 0.05})", file=sys.stderr)
    comps = {}
    for r in all_rows:
        comps.setdefault(r[2], 0)
        comps[r[2]] += 1
    print(f"SANITY2 component row counts: {comps}", file=sys.stderr)
    # per-run routed = layers x experts x matrices
    for label, run in RUNS:
        rr = [r for r in all_rows if r[0] == label and r[2] == "routed"]
        if not rr:
            continue
        layers = len({r[1] for r in rr}); experts = len({r[3] for r in rr}); mats = len({r[4] for r in rr})
        ok = layers * experts * mats == len(rr)
        print(f"SANITY3 {label}: routed {len(rr)} == {layers}L x {experts}E x {mats}M -> {ok}", file=sys.stderr)


if __name__ == "__main__":
    main()
