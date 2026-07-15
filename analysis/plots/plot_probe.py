#!/usr/bin/env python3
"""Mechanistic graphs from the router-probe logs (no training):
  A   per-token expert raster (full-MoE / temporal-resident / temporal-unconstrained-preference)
  A3  learned-locality overlap vs model scale (does the router learn to want its resident set?)
  B   rolling-policy hit-rate (coverage) vs resident budget K
  C   expert lifetime vs K
Reads results/phase0/runs/<run>/router_log.pt. See docs/research/mechanism/probe-results.md.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --no-caption = paper mode: drop the baked-in captions (detail goes in the LaTeX caption), use
# compact figsizes + short titles/labels + large fonts so figures stay legible after downscaling
# into a paper column. Outputs get a _nocaption suffix. Default mode is unchanged.
NO_CAPTION = "--no-caption" in sys.argv
PAPER = NO_CAPTION
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = f"{REPO}/results/phase0/runs"; OUT = f"{REPO}/results/phase0/figures"
if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
def outname(f): return f.replace(".png", "_nocaption.png") if NO_CAPTION else f
def load(run):
    import torch  # lazy: paper-mode learned-locality reads a CSV, so torch is only needed with raw logs
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
def _raster_csv(outfile, panels, k, E, L):
    """Condensed data behind a raster: the active (token, expert) cells per panel — a few thousand
    rows (~k per token), the tiny stand-in for the raw router_log.pt this raster was drawn from."""
    import csv
    figdata = OUT.replace("figures", "figure_data"); os.makedirs(figdata, exist_ok=True)
    name = outfile.replace(".png", ".csv")            # caption-independent name
    with open(f"{figdata}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["panel", "token", "expert", "num_experts_E", "topk_k", "moe_layer"])
        for title, M, _ in panels:                    # M is [W tokens, E experts] bool
            tok, exp = np.where(M)
            for tt, ee in zip(tok.tolist(), exp.tolist()):
                w.writerow([title.strip(), tt, ee, E, k, L])
    print("wrote", f"{figdata}/{name}")


def _draw_raster(panels, k, E, L, tag, outfile):
    # paper mode: taller panels + small markers so 64 expert rows don't blur into blobs when the
    # figure is downscaled into a paper column; edgeless squares keep small dots crisp.
    fig_w = 7.0 if PAPER else 13
    per_panel = 2.0 if PAPER else 2.5
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_w, per_panel*len(panels)+0.6), sharex=True)
    if len(panels) == 1: axes = [axes]
    for ax, (title, M, c) in zip(axes, panels):
        ys, xs = np.where(M.T); ax.scatter(xs, ys, s=(2.4 if PAPER else 6), c=c, marker="o", linewidths=0)
        ax.set_ylabel("expert idx"); ax.set_title(title, loc="left", fontsize=(15 if PAPER else 10))
        ax.set_ylim(-1, E); ax.set_yticks([0, (E-1)//2, E-1]); ax.grid(True, ls=":", alpha=0.3)
    axes[-1].set_xlabel("token position" if PAPER else f"token position (sequence 0, MoE layer {L})")
    if PAPER:
        fig.suptitle(f"Experts active per token ({k} of {E})", fontsize=16)
    else:
        fig.suptitle(f"Which experts are active at each token: {tag} (top-k = {k} of {E} experts)\n"
                     "horizontal streaks = an expert stays active across consecutive tokens (temporal locality)")
        fig.text(0.5, 0.01,
                 "Each dot marks an expert (y-axis) that is active at a given token position (x-axis) in one "
                 "sequence, at the last Mixture-of-Experts layer. 'temporal' = rolling residency: keep the "
                 "top-k experts resident and swap at most one per token. Long horizontal streaks mean an "
                 "expert stays selected across many consecutive tokens (temporal locality); this is descriptive, "
                 "not better/worse.", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 1] if PAPER else [0, 0.06, 1, 1])
    fig.savefig(f"{OUT}/{outname(outfile)}", dpi=200 if PAPER else 140); plt.close(fig); print("wrote", f"{OUT}/{outname(outfile)}")

def raster(temporal_run, moe_run, tag, outfile, W=220):
    t = load(temporal_run); L = sorted(t["layers"])[-1]; k = t["layers"][L]["k"]
    E = t["layers"][L]["logits"].shape[-1]; b = 0
    panels = []
    if moe_run:
        m = load(moe_run); panels.append(("full MoE  (top-k)", topk_ids(m["layers"][L]["logits"][:W, b], k), "C0"))
    panels.append(("temporal (resident set used)", t["layers"][L]["mask"][:W, b], "C2"))
    panels.append(("temporal (unconstrained preference)", topk_ids(t["layers"][L]["logits"][:W, b], k), "C2"))
    _raster_csv(outfile, panels, k, E, L)             # dump the condensed CSV alongside the figure
    _draw_raster(panels, k, E, L, tag, outfile)

def raster_from_csv(csv_name, outfile):
    """Redraw a raster from its condensed CSV (no raw logs needed) — same draw path as raster()."""
    import csv
    figdata = OUT.replace("figures", "figure_data")
    by_panel = {}; E = k = L = None
    with open(f"{figdata}/{csv_name}") as f:
        for r in csv.DictReader(f):
            E, k, L = int(r["num_experts_E"]), int(r["topk_k"]), r["moe_layer"]
            by_panel.setdefault(r["panel"], []).append((int(r["token"]), int(r["expert"])))
    W = 1 + max(tok for cells in by_panel.values() for tok, _ in cells)
    panels = []
    for title, cells in by_panel.items():              # dict preserves CSV (panel) order
        M = np.zeros((W, E), bool)
        for tok, exp in cells: M[tok, exp] = True
        panels.append((title, M, "C0" if title.startswith("full MoE") else "C2"))
    _draw_raster(panels, k, E, L, "", outfile)

# ---------------- A3: learned-locality overlap vs scale ----------------
def overlap(run):
    r = load(run); vals = []
    for L, rec in r["layers"].items():
        k = rec["k"]; tk = topk_ids(rec["logits"], k)
        prev = rec["mask"] if rec["mask"] is not None else tk           # resident(t-1) / top-k(t-1)
        vals.append(((tk[1:] & prev[:-1]).sum(-1) / k).mean())
    return float(np.mean(vals))

def a3_scale():
    if PAPER:   # read the committed exact series (no raw logs needed); coarse models = the plotted line
        import csv
        rows = []
        with open(f"{REPO}/results/phase0/figure_data/learned_locality_vs_scale.csv") as f:
            for r in csv.DictReader(f):
                if "coarse" not in r["model"]:
                    continue
                rows.append((float(r["active_params_M"]), float(r["temporal_overlap_pct"]),
                             float(r["full_moe_overlap_pct"]) if r["full_moe_overlap_pct"] else None,
                             float(r["random_pct"])))
        rows.sort()
        xs = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(4.7, 3.9))
        ax.plot(xs, [r[1] for r in rows], "o-", color="C2", lw=2, label="temporal")
        mm = [(r[0], r[2]) for r in rows if r[2] is not None]
        ax.plot([p[0] for p in mm], [p[1] for p in mm], "o-", color="C0", lw=2, label="full MoE")
        ax.plot(xs, [rows[0][3]] * len(xs), ":", color="gray", label="random")
        from matplotlib.ticker import FixedLocator, FixedFormatter
        ax.set_xscale("log"); ax.set_xlabel("active params (M)")
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 40]))
        ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10", "20", "40"]))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel("same-set overlap (%)")
        ax.set_title("Learned temporal locality")
        ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
        fig.tight_layout()
        out = f"{OUT}/{outname('learned_temporal_locality_vs_model_size.png')}"
        fig.savefig(out, dpi=200); plt.close(fig); print("wrote", out); return
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
    ax.plot([p[0] for p in mm], [p[1] for p in mm], "o-", color="C0", label="full MoE (natural)")
    ax.plot(xs, np.array(rnd)*100, ":", color="gray", label="random baseline (k / E)")
    ax.set_xscale("log"); ax.set_xlabel("active non-embedding params N (millions)")
    ax.set_ylabel("overlap with previous active set  (%)  — higher = more temporally coherent")
    ax.set_title("The temporal router learns to prefer its resident set of experts (and it holds as models grow)")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
    if not NO_CAPTION:
        fig.text(0.5, 0.01,
                 "Overlap between a token's freely chosen top-k experts and the previous token's active expert "
                 "set, averaged over layers and tokens; higher (%) = more temporally coherent routing. "
                 "'temporal' = rolling residency (keep top-k experts resident, swap 1 per token). x-axis is "
                 "active non-embedding parameters N (millions, log scale). Random baseline = k / E (top-k over "
                 "E experts).", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0 if NO_CAPTION else 0.08, 1, 1])
    out = f"{OUT}/{outname('learned_temporal_locality_vs_model_size.png')}"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)

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
    series = [("temporal, 8.1M active",             "tmoe_minlogit_sh1_s2_1e17", "C2", "-"),
              ("temporal, 15M active",              "tmoe_minlogit_sh1_s3_1e17", "C2", "--"),
              ("temporal, 38M active",              "flame38m_temporal_minlogit", "C4", "-"),
              ("full MoE, 8.1M active",             "v16k_sweep_s2_1e17", "C0", "-"),
              ("temporal, fine-grained (18 of 192)", "g3_tmoe_s1_1e17", "C3", ":")]
    data = {n: sweep(r) for n, r, _, _ in series}
    for fname, idx, ylab, title, cap in [
        ("routing_coverage_vs_resident_cache_size.png", 1, "routing hit-rate (mean fraction of top-k already resident)",
         "A bigger resident cache closes the routing gap; the temporal router is ~2x more cacheable than full MoE at every model size",
         "Rolling-residency simulation: keep K experts resident and evict at most one per token (min-logit "
         "eviction). x-axis K / k = resident cache size relative to the active top-k (k), where 1 = the "
         "current setting. y-axis = fraction of each token's top-k experts that are already resident "
         "(hit-rate, 0-1); higher is better. 'temporal' = rolling-residency model; 'full MoE' = standard "
         "top-k routing. E = total experts."),
        ("expert_lifetime_vs_resident_cache_size.png", 2, "mean expert lifetime (consecutive tokens resident)",
         "Experts stay resident longer as the resident cache grows",
         "Same rolling-residency simulation (keep K experts resident, evict at most one per token). "
         "x-axis K / k = resident cache size relative to the active top-k (k), 1 = current setting. y-axis = "
         "mean number of consecutive tokens an expert stays resident before eviction (lifetime); higher = "
         "more stable residency. 'temporal' = rolling-residency model; 'full MoE' = standard top-k routing.")]:
        fig, ax = plt.subplots(figsize=(9, 5.9))
        for n, r, c, ls in series:
            Ks, cov, life, k = data[n]; ax.plot(Ks/k, (cov if idx == 1 else life), ls, color=c, marker="o", ms=3.5, label=n)
        ax.set_xlabel("resident cache size  K / k   (1 = current; →  larger resident cache)")
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
        if idx == 1: ax.axhline(1.0, color="gray", lw=.8, ls=":")
        if not NO_CAPTION:
            fig.text(0.5, 0.01, cap, ha="center", fontsize=8, wrap=True)
        fig.tight_layout(rect=[0, 0 if NO_CAPTION else 0.08, 1, 1])
        fig.savefig(f"{OUT}/{outname(fname)}", dpi=140); plt.close(fig); print("wrote", f"{OUT}/{outname(fname)}")

if __name__ == "__main__":
    have_logs = os.path.exists(f"{RUNS}/tmoe_minlogit_sh1_s2_1e17/router_log.pt")
    if PAPER and not have_logs:
        # local run without raw logs: redraw rasters + learned-locality from the committed CSVs
        for csvf, out in [("expert_selection_per_token_8M_model.csv",  "expert_selection_per_token_8M_model.png"),
                          ("expert_selection_per_token_15M_model.csv", "expert_selection_per_token_15M_model.png"),
                          ("expert_selection_per_token_38M_model.csv", "expert_selection_per_token_38M_model.png")]:
            raster_from_csv(csvf, out)
        a3_scale()
    else:
        raster("tmoe_minlogit_sh1_s2_1e17", "v16k_sweep_s2_1e17", "full MoE vs temporal, 8.1M active @ 10^17 FLOPs", "expert_selection_per_token_8M_model.png")
        raster("tmoe_minlogit_sh1_s3_1e17", "v16k_sweep_s3_1e17", "full MoE vs temporal, 15M active @ 10^17 FLOPs", "expert_selection_per_token_15M_model.png")
        raster("flame38m_temporal_minlogit", None, "temporal, 38M active @ 10^18 FLOPs", "expert_selection_per_token_38M_model.png")
        a3_scale(); graphs_BC()
