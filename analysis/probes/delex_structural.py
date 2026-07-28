#!/usr/bin/env python3
"""delex-1e19 Part 1 (a): structural stats per model -> mechinterp_structural_1e19.csv.

Per model (experts pooled over all MoE layers of the capture):
  PR_median        median expert selectivity = normalized inverse Simpson of renormalized gate mass
                   q_e(t)=g_e(t)/sum_t g_e(t);  PR_e = 1/(N sum_t q_e(t)^2) in (0,1]
  generalist_frac  |{e: PR_e>0.5}|/E
  router_entropy   mean_t[-sum_e g_e(t) ln g_e(t)] / ln E   (per-token routing flatness in [0,1])
  dist2centroid_mean  mean_e (1 - cos(w_e, wbar)) of flattened FFN weights vs centroid
  pairwise_cos_med / _p99  median / p99 of pairwise cos(w_e, w_e')
  eff_rank         participation ratio of the eigenvalues of the expert gate-mass correlation matrix
  strong_corr_pairs  # expert pairs with Pearson corr of per-token gate series > 0.5
Schema mirrors mechinterp_structural.csv + a `budget` column (1e19).
"""
import os, sys, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_read

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, RUNS
OUT = os.path.join(ROOT, "results/ablations/mechinterp_structural_1e19.csv")
CELLS = [("moe_coarse_1e19", "moe_coarse_1e19", "full"),
         ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19", "temporal"),
         ("temporal_fine_1e19", "temporal_fine_g3_1e19", "temporal")]


def gate_stats(cap):
    """Pool per-token gates over all MoE layers -> per-expert PR, router entropy, gate-series corr."""
    d = torch.load(cap, map_location="cpu", weights_only=False)
    PR, Hs, corr_counts, eff_ranks, k = [], [], 0, [], None
    for L in sorted(d["layers"]):
        Ld = d["layers"][L]; lg = Ld["logits"].float(); k = Ld["k"]; E = lg.shape[-1]
        N = lg.shape[0] * lg.shape[1]
        g = torch.softmax(lg, dim=-1).reshape(N, E).double()          # [N,E]
        # router flatness (per token)
        H = (-(g * (g.clamp(min=1e-12)).log()).sum(-1)).mean() / np.log(E)
        Hs.append(float(H))
        # selectivity per expert
        mass = g.sum(0)                                                # [E]
        q = g / mass.clamp(min=1e-12)                                  # renormalized over tokens
        pr = 1.0 / (N * (q * q).sum(0)).clamp(min=1e-12)
        PR.extend(pr.tolist())
        # gate-series correlation across experts
        gc = g - g.mean(0)
        cov = (gc.T @ gc) / N
        sd = cov.diag().clamp(min=1e-12).sqrt()
        cc = cov / (sd[:, None] * sd[None, :])
        iu = torch.triu_indices(E, E, 1)
        corr_counts += int((cc[iu[0], iu[1]].abs() > 0.5).sum())
        ev = torch.linalg.eigvalsh(cov).clamp(min=0)
        eff_ranks.append(float((ev.sum() ** 2) / (ev * ev).sum().clamp(min=1e-12)))
    return np.array(PR), float(np.mean(Hs)), corr_counts, float(np.mean(eff_ranks)), k, E


def weight_identity(run):
    """Flatten each routed expert's (fc1|fc2) weights over MoE layers -> per-expert vector; centroid."""
    ip = ckpt_read.iter_dir(os.path.join(RUNS, run, "ckpt"))
    meta = ckpt_read.weight_keys(ckpt_read.FileSystemReader(ip))
    import re
    keys = sorted([kk for kk in meta if re.search(r"experts\.experts\.linear_fc[12]\.weight$", kk)])
    sd = ckpt_read.load(ip, keys)
    # group by layer -> concat fc1,fc2 per expert; then average identity across layers
    vecs_by_layer = {}
    for kk in keys:
        L = int(re.search(r"layers\.(\d+)\.", kk).group(1))
        t = sd[kk].float()                                            # [E,out,in]
        E = t.shape[0]
        v = t.reshape(E, -1)
        vecs_by_layer.setdefault(L, []).append(v)
    d2c, pcos = [], []
    for L, parts in vecs_by_layer.items():
        W = torch.cat(parts, dim=1)                                   # [E, feat]
        Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cbar = Wn.mean(0); cbar = cbar / cbar.norm().clamp(min=1e-12)
        d2c.extend((1 - (Wn @ cbar)).tolist())
        C = Wn @ Wn.T
        iu = torch.triu_indices(W.shape[0], W.shape[0], 1)
        pcos.extend(C[iu[0], iu[1]].tolist())
    return float(np.mean(d2c)), float(np.median(pcos)), float(np.percentile(pcos, 99))


def main():
    rows = []
    for label, run, kind in CELLS:
        cap = os.path.join(RUNS, run, "delex_capture.pt")
        if not os.path.exists(cap):
            print(f"[skip] {label}: no capture"); continue
        PR, Hbar, corrpairs, effrank, k, E = gate_stats(cap)
        d2c, pcos_med, pcos_p99 = weight_identity(run)
        row = [label, run, kind, E, k, round(float(np.median(PR)), 4),
               round(float((PR > 0.5).mean()), 4), round(Hbar, 4), round(effrank, 2),
               corrpairs, round(d2c, 4), round(pcos_med, 4), round(pcos_p99, 4), "1e19"]
        rows.append(row)
        print(f"[ok] {label}: PR_med={row[5]} generalist={row[6]} Hbar={row[7]} "
              f"d2c={row[10]} pcos_med={row[11]}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "run", "kind", "E", "k", "PR_median", "generalist_frac", "router_entropy",
                    "eff_rank", "strong_corr_pairs", "dist2centroid_mean", "pairwise_cos_med",
                    "pairwise_cos_p99", "budget"])
        w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
