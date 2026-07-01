#!/usr/bin/env python3
"""Mechanistic graphs from the router-probe logs (no training):
  A   per-token expert raster (full-MoE / temporal-resident / temporal-unconstrained-preference)
  A3  learned-locality overlap vs model scale (does the router learn to want its resident set?)
  B   rolling-policy hit-rate (coverage) vs resident budget K
  C   expert lifetime vs K
Reads results/phase0/runs/<run>/router_log.pt. See docs/research/mechanistic-probe-results.md.
"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RUNS = "/workspace/FLAME-MoE/results/phase0/runs"; OUT = "/workspace/FLAME-MoE/results/phase0"
def load(run):
    d = torch.load(f"{RUNS}/{run}/router_log.pt", map_location="cpu")
    return {"temporal": d["temporal"],
            "layers": {ln: {"logits": r["logits"].float().numpy(),
                            "mask": (r["mask"].numpy() if r["mask"] is not None else None),
                            "k": r["k"]} for ln, r in d["layers"].items()}}

def topk_ids(logits, k):
    idx = np.argpartition(-logits, k-1, axis=-1)[..., :k]
    m = np.zeros_like(logits, bool); np.put_along_axis(m, idx, True, axis=-1); return m

# tag, active-params(M), temporal run, full-MoE run (or None)
PAIRS = [("s0 @1e16", 1.36, "tmoe_minlogit_sh1_s0_1e16", "v16k_d_s0_1e16"),
         ("s2 @1e17", 8.12, "tmoe_minlogit_sh1_s2_1e17", "v16k_sweep_s2_1e17"),
         ("s3 @1e17", 14.77, "tmoe_minlogit_sh1_s3_1e17", "v16k_sweep_s3_1e17"),
         ("38M @1e18", 38.0, "flame38m_temporal_minlogit", None)]
G3 = ("G3 s1@1e17", 3.91, "g3_tmoe_s1_1e17", None)

# ---------------- A: per-token raster (2 or 3 panels) ----------------
def raster(temporal_run, moe_run, tag, outfile, W=220):
    t = load(temporal_run); L = sorted(t["layers"])[-1]; k = t["layers"][L]["k"]
    E = t["layers"][L]["logits"].shape[-1]; b = 0
    panels = []
    if moe_run:
        m = load(moe_run); panels.append(("full MoE  (top-k)", topk_ids(m["layers"][L]["logits"][:W, b], k), "C0"))
    panels.append(("temporal  (resident set used)", t["layers"][L]["mask"][:W, b], "C2"))
    panels.append(("temporal  (unconstrained preference)", topk_ids(t["layers"][L]["logits"][:W, b], k), "C2"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.5*len(panels)+0.6), sharex=True)
    if len(panels) == 1: axes = [axes]
    for ax, (title, M, c) in zip(axes, panels):
        ys, xs = np.where(M.T); ax.scatter(xs, ys, s=6, c=c, marker="s")
        ax.set_ylabel("expert"); ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(-1, E); ax.grid(True, ls=":", alpha=0.3)
    axes[-1].set_xlabel(f"token position (sequence 0, MoE layer {L})")
    fig.suptitle(f"A — experts chosen per token: {tag} (k={k} of {E})\n"
                 "horizontal streaks = expert stays active across consecutive tokens (temporal locality)")
    fig.tight_layout(); fig.savefig(f"{OUT}/{outfile}", dpi=140); plt.close(fig); print("wrote", outfile)

# ---------------- A3: learned-locality overlap vs scale ----------------
def overlap(run):
    r = load(run); vals = []
    for L, rec in r["layers"].items():
        k = rec["k"]; tk = topk_ids(rec["logits"], k)
        prev = rec["mask"] if rec["mask"] is not None else tk           # resident(t-1) / top-k(t-1)
        vals.append(((tk[1:] & prev[:-1]).sum(-1) / k).mean())
    return float(np.mean(vals))

def a3_scale():
    print("A3  overlap( top-k(t), previous active set(t-1) )  vs scale:")
    xs, temp, moe, rnd = [], [], [], []
    for tag, N, tr, mr in PAIRS:
        rt = load(tr); E = next(iter(rt["layers"].values()))["logits"].shape[-1]
        k = next(iter(rt["layers"].values()))["k"]
        ot = overlap(tr); om = overlap(mr) if mr else np.nan
        xs.append(N); temp.append(ot); moe.append(om); rnd.append(k/E)
        print(f"   {tag:10s} N={N:5.1f}M   temporal {ot*100:4.1f}%   MoE {('%4.1f%%'%(om*100)) if mr else '  n/a'}"
              f"   random {k/E*100:4.1f}%")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(xs, np.array(temp)*100, "o-", color="C2", lw=2, label="temporal (learned preference)")
    mm = [(x, v*100) for x, v in zip(xs, moe) if not np.isnan(v)]
    ax.plot([p[0] for p in mm], [p[1] for p in mm], "o-", color="C0", label="vanilla MoE (natural)")
    ax.plot(xs, np.array(rnd)*100, ":", color="gray", label="random baseline (k/E)")
    ax.set_xscale("log"); ax.set_xlabel("active non-embedding params (millions)")
    ax.set_ylabel("overlap with previous active set  (%)  — higher = more temporally coherent")
    ax.set_title("A3 — the temporal router learns to want its resident set (and it holds with scale)")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/probe_A3_vs_scale.png", dpi=140); plt.close(fig)
    print("wrote probe_A3_vs_scale.png")

# ---------------- rolling residency sim (our policy: K, <=1 swap/token, min_logit) ----------------
def rolling(logits, k, K):
    T, E = logits.shape; order = np.argsort(-logits, axis=1)
    res = np.zeros(E, bool); res[order[0, :K]] = True; enter = np.where(res, 0, -1).astype(float)
    cov = np.empty(T); lifes = []
    for tk in range(T):
        D = order[tk, :k]; dm = np.zeros(E, bool); dm[D] = True; cov[tk] = res[D].mean()
        for e in D[~res[D]][:1]:
            cand = np.where(res & ~dm)[0]
            if len(cand) == 0: cand = np.where(res)[0]
            eo = cand[np.argmin(logits[tk, cand])]; lifes.append(tk-enter[eo])
            res[eo] = False; res[e] = True; enter[e] = tk
    lifes += list(T - enter[res]); return cov.mean(), float(np.mean(lifes))

def sweep(run, nseq=8):
    r = load(run); k = next(iter(r["layers"].values()))["k"]; E = next(iter(r["layers"].values()))["logits"].shape[-1]
    Ks = list(range(k, E+1, max(1, (E-k)//16))); covs, lifes = [], []
    for K in Ks:
        c = l = n = 0
        for rec in r["layers"].values():
            for b in range(min(nseq, rec["logits"].shape[1])):
                cc, ll = rolling(rec["logits"][:, b], k, K); c += cc; l += ll; n += 1
        covs.append(c/n); lifes.append(l/n)
    return np.array(Ks), np.array(covs), np.array(lifes), k

def graphs_BC():
    series = [("temporal s2 (8M, k6/64)",  "tmoe_minlogit_sh1_s2_1e17", "C2", "-"),
              ("temporal s3 (15M, k6/64)",  "tmoe_minlogit_sh1_s3_1e17", "C2", "--"),
              ("temporal 38M@1e18 (k6/64)", "flame38m_temporal_minlogit", "C4", "-"),
              ("MoE s2 (8M, k6/64)",        "v16k_sweep_s2_1e17", "C0", "-"),
              ("temporal G3 (k18/192)",     "g3_tmoe_s1_1e17", "C3", ":")]
    data = {n: sweep(r) for n, r, _, _ in series}
    for fname, idx, ylab, title in [
        ("probe_B_coverage_vs_k.png", 1, "rolling-policy hit-rate (mean top-k coverage)",
         "B — bigger resident cache closes the routing gap (≤1-swap/token); temporal ~2x more cacheable than MoE at every scale"),
        ("probe_C_lifetime_vs_k.png", 2, "mean expert lifetime (consecutive tokens resident)",
         "C — expert lifetime vs resident budget (≤1-swap/token policy)")]:
        fig, ax = plt.subplots(figsize=(9, 5.6))
        for n, r, c, ls in series:
            Ks, cov, life, k = data[n]; ax.plot(Ks/k, (cov if idx == 1 else life), ls, color=c, marker="o", ms=3.5, label=n)
        ax.set_xlabel("resident budget  K / k   (1 = current; →  larger resident cache)")
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
        if idx == 1: ax.axhline(1.0, color="gray", lw=.8, ls=":")
        fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=140); plt.close(fig); print("wrote", fname)

if __name__ == "__main__":
    raster("tmoe_minlogit_sh1_s2_1e17", "v16k_sweep_s2_1e17", "full MoE vs temporal (s2 @ 1e17)", "probe_A_raster.png")
    raster("tmoe_minlogit_sh1_s3_1e17", "v16k_sweep_s3_1e17", "full MoE vs temporal (s3 @ 1e17, 15M)", "probe_A_raster_s3.png")
    raster("flame38m_temporal_minlogit", None, "temporal 38M @ 1e18 (real budget)", "probe_A_raster_38M.png")
    a3_scale(); graphs_BC()
