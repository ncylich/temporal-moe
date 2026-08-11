#!/usr/bin/env python3
"""Response length against benchmark damage. Lengths are reconstructed as final logged
vLLM output throughput x elapsed / items (estimate, ~10%); damages from the audited
genbench CSVs. Finding: pooled Spearman +0.72 (p=0.01) with damage NEGATIVE, i.e.
SHORT-response tasks take the damage; no evidence of error accumulation over length.
Data table inline (log-derived lengths pinned at analysis time; logs are ephemeral)."""
DATA = {  # (model, task): (mean_free_len_tokens, dmg_at_k, dmg_at_12p5)
 ("olmoe", "gsm8k"): (91, -28.5, None), ("olmoe", "humaneval"): (101, -8.5, None),
 ("olmoe", "mmlu"): (125, -15.8, None), ("olmoe", "ifeval"): (370, -9.0, None),
 ("gemma4", "gsm8k"): (229, -9.5, -1.0), ("gemma4", "mmlu"): (334, -5.7, -2.6),
 ("gemma4", "ifeval"): (336, -1.0, -2.5),
 ("qwen35", "humaneval"): (145, -4.9, -3.7), ("qwen35", "gsm8k"): (877, -7.5, -4.0),
 ("qwen35", "mmlu"): (965, 3.9, 2.2), ("qwen35", "ifeval"): (1264, 0.5, 0.5),
}
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import ABLATIONS
    cols = {"olmoe": "tab:blue", "gemma4": "tab:green", "qwen35": "tab:red"}
    names = {"olmoe": "OLMoE-Instruct", "gemma4": "gemma4-26B-IT", "qwen35": "Qwen3.5-35B"}
    marks = {"gsm8k": "o", "ifeval": "s", "humaneval": "^", "mmlu": "D"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for (model, task), v in DATA.items():
        ax.scatter(v[0], v[1], s=90, color=cols[model], marker=marks[task],
                   edgecolor="black", linewidth=0.6, zorder=5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", ls="", ms=9, label=names[m])
               for m, c in cols.items()]
    handles += [Line2D([], [], color="grey", marker=mk, ls="", ms=8, label=t.upper())
                for t, mk in marks.items()]
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 400, 800, 1200])
    ax.set_xticklabels(["100", "200", "400", "800", "1200"])
    import matplotlib.ticker as mt
    ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlabel("mean response length, tokens (reconstructed, ±10%)")
    ax.set_ylabel("accuracy change at R = k, points")
    ax.set_title("Constraint damage against response length\n"
                 "(short-answer tasks take the damage; Spearman +0.72, p = 0.01)",
                 fontsize=10)
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(ABLATIONS, "figures", "length_vs_damage.png"), dpi=150)
    print("wrote length_vs_damage.png")


if __name__ == "__main__":
    from scipy.stats import spearmanr
    pts = [(v[0], v[1]) for v in DATA.values()]
    rho, p = spearmanr(*zip(*pts))
    print(f"pooled spearman(length, damage@R=k) = {rho:+.2f} (n={len(pts)}, p={p:.3f})")
    figure()
