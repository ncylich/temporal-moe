#!/usr/bin/env python3
"""PART C converter: activation_probe.py dumps (act_log.pt per cell) -> two aggregate CSVs.

  stability_activations.csv  (layer/expert stats), columns:
    run,layer,expert,out_l2_mean,sel_count,interm_max,interm_kurt,
    rlogit_mean,rlogit_std,rlogit_mean_res,rlogit_mean_nonres,gate_mean,gate_std
  stability_trunk.csv  (layer/head + residual stats), columns:
    run,layer,head_or_dim,attn_max_logit,attn_out_in_ratio,mlp_out_in_ratio,resid_absmean
    - attention rows: layer 0..L-1, head_or_dim=head, attn_max_logit + ratios filled
    - residual rows : layer=-1, head_or_dim=dim rank (top-64 by |activation|), resid_absmean filled;
                      head_or_dim=-1 -> overall mean, -2 -> overall max (final-layer residual stream)

Layer index is 0-indexed decoder.layers.N (matches Parts A/B). Router stats use Megatron
layer_number (1-indexed) internally and are realigned here by -1.
"""
import os, sys, csv, torch

ROOT = "/workspace/FLAME-MoE"
RUNS = os.path.join(ROOT, "results/phase0/runs")
ACT_OUT = os.path.join(ROOT, "results/ablations/stability_activations.csv")
TRUNK_OUT = os.path.join(ROOT, "results/ablations/stability_trunk.csv")

CELLS = [
    ("dense_local", "flame38m_dense_local"), ("g1_moe", "flame38m_g1_moe"),
    ("g1_temporal", "flame38m_g1_temporal"), ("g3_moe", "flame38m_g3_moe"),
    ("g3_temporal", "flame38m_g3_temporal"), ("dense_1e19", "dense_1e19"),
    ("moe_coarse_1e19", "moe_coarse_1e19"), ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
    ("temporal_fine_1e19", "temporal_fine_g3_1e19"),
]


def g(t, i):
    return f"{float(t[i]):.6g}"


def main():
    act_rows, trunk_rows = [], []
    for label, run in CELLS:
        f = os.path.join(RUNS, run, "act_log.pt")
        if not os.path.exists(f):
            print(f"[skip] {label}: no act_log.pt")
            continue
        S = torch.load(f, map_location="cpu", weights_only=False)["stats"]
        experts = S.get("experts", {})
        router = {ln - 1: r for ln, r in S.get("router", {}).items()}   # realign to 0-indexed
        for L in sorted(experts):
            e = experts[L]
            r = router.get(L, {})
            E = int(e["sel_count"].numel())
            for x in range(E):
                row = [label, L, x, g(e["out_l2_mean"], x), int(e["sel_count"][x]),
                       g(e["interm_max"], x), g(e["interm_kurt"], x)]
                if r:
                    row += [g(r["rlogit_mean"], x), g(r["rlogit_std"], x),
                            g(r["rlogit_mean_res"], x) if "rlogit_mean_res" in r else "",
                            g(r["rlogit_mean_nonres"], x) if "rlogit_mean_nonres" in r else "",
                            g(r["gate_mean"], x), g(r["gate_std"], x)]
                else:
                    row += ["", "", "", "", "", ""]
                act_rows.append(row)
        # trunk: per (layer, head) attn max logit + block ratios
        attn, mlp, core = S.get("attn", {}), S.get("mlp", {}), S.get("core", {})
        for L in sorted(core):
            a = attn.get(L, {}); m = mlp.get(L, {})
            ar = f"{a['out'] / a['in']:.6g}" if a.get("in") else ""
            mr = f"{m['out'] / m['in']:.6g}" if m.get("in") else ""
            ml = core[L].get("maxlogit")
            if ml is not None:
                for h in range(ml.numel()):
                    trunk_rows.append([label, L, h, f"{float(ml[h]):.6g}", ar, mr])
        # residual stream (final layer): top-64 |activation| dims + overall mean/max
        resid = S.get("resid", {}).get("x_absmean_perdim")
        if resid is not None:
            vals, idx = resid.sort(descending=True)
            for rank in range(min(64, vals.numel())):
                trunk_rows.append([label, -1, int(idx[rank]), "", "", "", f"{float(vals[rank]):.6g}"])
            trunk_rows.append([label, -1, -1, "", "", "", f"{float(resid.mean()):.6g}"])   # overall mean
            trunk_rows.append([label, -1, -2, "", "", "", f"{float(resid.max()):.6g}"])    # overall max
        print(f"[ok] {label}: {sum(1 for r in act_rows if r[0]==label)} act rows, "
              f"{sum(1 for r in trunk_rows if r[0]==label)} trunk rows")

    with open(ACT_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "layer", "expert", "out_l2_mean", "sel_count", "interm_max", "interm_kurt",
                    "rlogit_mean", "rlogit_std", "rlogit_mean_res", "rlogit_mean_nonres",
                    "gate_mean", "gate_std"])
        w.writerows(act_rows)
    with open(TRUNK_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "layer", "head_or_dim", "attn_max_logit",
                    "attn_out_in_ratio", "mlp_out_in_ratio", "resid_absmean"])
        w.writerows(trunk_rows)
    print(f"[write] {ACT_OUT}: {len(act_rows)} rows ({os.path.getsize(ACT_OUT)} B)")
    print(f"[write] {TRUNK_OUT}: {len(trunk_rows)} rows ({os.path.getsize(TRUNK_OUT)} B)")


if __name__ == "__main__":
    main()
