#!/usr/bin/env python3
"""Mechanistic graphs from the router-probe logs (no training). Generates:
  A  per-token expert raster: full-MoE top-k / temporal resident / temporal unconstrained-preference
  B  rolling-policy hit-rate (coverage) vs resident budget K
  C  expert lifetime vs K
plus the A3 scalar: does the temporal model, unconstrained, already want its previous resident set?

Reads results/phase0/runs/<run>/router_log.pt. Lower/■ = better where noted.
"""
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/workspace/FLAME-MoE/results/phase0/runs"
def load(run):
    d = torch.load(f"{R}/{run}/router_log.pt", map_location="cpu")
    out = {}
    for ln, rec in d["layers"].items():
        lg = rec["logits"].float().numpy()                 # [seq, batch, E]
        mk = rec["mask"].numpy() if rec["mask"] is not None else None
        out[ln] = {"logits": lg, "mask": mk, "k": rec["k"]}
    return {"temporal": d["temporal"], "layers": out}

# matched pairs (temporal, full-MoE) + G3
S2T, S2M = "tmoe_minlogit_sh1_s2_1e17", "v16k_sweep_s2_1e17"      # 64 experts, k=6, 5 MoE layers
S0T, S0M = "tmoe_minlogit_sh1_s0_1e16", "v16k_d_s0_1e16"          # 64 experts, k=6, 3 MoE layers
G3T      = "g3_tmoe_s1_1e17"                                       # 192 experts, k=18, 4 MoE layers

def topk_ids(logits, k):                                            # [.., E] -> set-mask [.., E] of top-k
    idx = np.argpartition(-logits, k-1, axis=-1)[..., :k]
    m = np.zeros_like(logits, bool); np.put_along_axis(m, idx, True, axis=-1); return m

# ---------------- A: per-token expert raster (s2@1e17, deepest MoE layer, one sequence) ----------------
def graph_A():
    t = load(S2T); m = load(S2M)
    L = sorted(t["layers"])[-1]                                    # deepest MoE layer (strongest locality)
    k = t["layers"][L]["k"]; b, W = 0, 220                          # sequence 0, first W tokens
    moe_tk  = topk_ids(m["layers"][L]["logits"][:W, b, :], k)      # full MoE top-k
    res     = t["layers"][L]["mask"][:W, b, :]                     # temporal resident set
    unc_tk  = topk_ids(t["layers"][L]["logits"][:W, b, :], k)      # temporal unconstrained top-k
    panels = [("full MoE  (top-k)", moe_tk, "C0"),
              ("temporal  (resident set used)", res, "C2"),
              ("temporal  (unconstrained preference)", unc_tk, "C2")]
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    for ax, (title, M, c) in zip(axes, panels):
        ys, xs = np.where(M.T)                                      # expert (y) vs token (x)
        ax.scatter(xs, ys, s=6, c=c, marker="s"); ax.set_ylabel("expert"); ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(-1, M.shape[1]); ax.grid(True, ls=":", alpha=0.3)
    axes[-1].set_xlabel(f"token position (sequence 0, MoE layer {L})")
    fig.suptitle("A — experts chosen per token: full MoE vs temporal (s2 @ 1e17, k=6 of 64)\n"
                 "horizontal streaks = an expert stays active across consecutive tokens (temporal locality)")
    fig.tight_layout(); fig.savefig(f"{R}/../probe_A_raster.png", dpi=140); plt.close(fig)
    print("wrote probe_A_raster.png")

# ---------------- A3 scalar: unconstrained preference vs previous resident set --------------------------
def a3_overlap():
    t = load(S2T); m = load(S2M); rows = []
    for tag, run in [("temporal (s2)", t), ("MoE (s2)", m)]:
        ov_prev_res, ov_prev_top = [], []
        for L, rec in run["layers"].items():
            k = rec["k"]; lg = rec["logits"]; T = lg.shape[0]
            tk = topk_ids(lg, k)                                    # [T,B,E] this-token top-k
            # previous-token reference set: resident(t-1) for temporal, top-k(t-1) for MoE
            prev = (rec["mask"] if rec["mask"] is not None else tk)
            inter = (tk[1:] & prev[:-1]).sum(-1) / k                # overlap of top-k(t) with prev set(t-1)
            ov_prev_res.append(inter.mean())
        rows.append((tag, float(np.mean(ov_prev_res))))
    print("A3  overlap( unconstrained top-k(t) , previous resident/top-k(t-1) ):")
    for tag, v in rows: print(f"      {tag:16s} {v*100:5.1f}%")
    return rows

# ---------------- rolling residency simulation (our policy: K budget, <=1 swap/token, min_logit) --------
def rolling(logits, k, K, evict="min_logit"):
    T, E = logits.shape
    order = np.argsort(-logits, axis=1)
    res = np.zeros(E, bool); res[order[0, :K]] = True
    enter = np.where(res, 0, -1).astype(float); cov = np.empty(T); lifes = []
    for tkn in range(T):
        D = order[tkn, :k]; dm = np.zeros(E, bool); dm[D] = True
        cov[tkn] = res[D].mean()
        for e in D[~res[D]][:1]:                                    # <=1 swap/token: highest-logit miss
            cand = np.where(res & ~dm)[0]
            if len(cand) == 0: cand = np.where(res)[0]
            e_out = cand[np.argmin(logits[tkn, cand])]              # min_logit eviction
            lifes.append(tkn - enter[e_out]); res[e_out] = False; res[e] = True; enter[e] = tkn
    lifes += list(T - enter[res])
    return cov.mean(), float(np.mean(lifes))

def sweep(run, nseq=8):
    r = load(run); k = next(iter(r["layers"].values()))["k"]
    E = next(iter(r["layers"].values()))["logits"].shape[-1]
    Ks = list(range(k, E + 1, max(1, (E - k)//16)))
    covs, lifes = [], []
    for K in Ks:
        c_acc, l_acc, n = 0, 0, 0
        for L, rec in r["layers"].items():
            lg = rec["logits"]
            for b in range(min(nseq, lg.shape[1])):
                c, l = rolling(lg[:, b, :], k, K); c_acc += c; l_acc += l; n += 1
        covs.append(c_acc/n); lifes.append(l_acc/n)
    return np.array(Ks), np.array(covs), np.array(lifes), k, E

# ---------------- B & C ----------------
def graphs_BC():
    series = [("temporal-pref (s2, k=6/64)", S2T, "C2", "-"),
              ("MoE demand  (s2, k=6/64)",   S2M, "C0", "-"),
              ("temporal-pref (G3, k=18/192)", G3T, "C3", "--")]
    data = {name: sweep(run) for name, run, _, _ in series}
    # B: coverage vs K/k
    figB, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, run, c, ls in series:
        Ks, cov, _, k, E = data[name]; ax.plot(Ks/k, cov, ls, color=c, marker="o", ms=4, label=name)
    ax.set_xlabel("resident budget  K / k   (1 = our current setting; →  = larger resident cache)")
    ax.set_ylabel("rolling-policy hit-rate  (mean top-k coverage)")
    ax.set_title("B — how much a bigger resident cache closes the routing gap (our ≤1-swap/token policy)")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(); ax.axhline(1.0, color="gray", lw=.8, ls=":")
    figB.tight_layout(); figB.savefig(f"{R}/../probe_B_coverage_vs_k.png", dpi=140); plt.close(figB)
    # C: lifetime vs K/k
    figC, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, run, c, ls in series:
        Ks, _, life, k, E = data[name]; ax.plot(Ks/k, life, ls, color=c, marker="o", ms=4, label=name)
    ax.set_xlabel("resident budget  K / k"); ax.set_ylabel("mean expert lifetime  (consecutive tokens resident)")
    ax.set_title("C — expert lifetime vs resident budget (our ≤1-swap/token policy)")
    ax.grid(True, ls=":", alpha=0.4); ax.legend()
    figC.tight_layout(); figC.savefig(f"{R}/../probe_C_lifetime_vs_k.png", dpi=140); plt.close(figC)
    print("wrote probe_B_coverage_vs_k.png, probe_C_lifetime_vs_k.png")
    for name in data:
        Ks, cov, life, k, E = data[name]
        print(f"   {name:30s} coverage@K=k {cov[0]*100:4.1f}%  @K=2k {cov[min(len(cov)-1, np.argmin(np.abs(Ks-2*k)))]*100:4.1f}%")

if __name__ == "__main__":
    graph_A(); a3_overlap(); graphs_BC()
