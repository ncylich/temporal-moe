#!/usr/bin/env python3
"""Talk-only figures for the advising pitch (not used by the paper).

Writes three self-contained slide graphics to paper/talk_figures/:
  slide02_total_vs_active.png  - total vs active params, two production MoEs, log-y bars
  slideA3_olmoe_ladder.png     - OLMoE adaptation recovery ladder (% of cold-impose gap)

Colors follow the repo isoFLOP standard where it applies (MoE blue, temporal/active green).
Run: python3 analysis/plots/plot_talk_extras.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = f"{REPO}/paper/talk_figures"
os.makedirs(OUT, exist_ok=True)

BLUE = "#5aa0dd"      # total params
GREEN = "#2e8b57"     # active params
DK_GREEN = "#145a14"  # emphasis

plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15})


# --- Slide 2: total vs active parameters (log-scale grouped bars) --------------
def total_vs_active():
    models = ["Qwen3-30B-A3B", "Kimi K3"]
    total = [30.0, 2800.0]      # billions
    active = [3.0, 50.0]        # billions
    pct = ["~10% active", "~1.8% active"]
    tlab = ["30B", "2.8T"]
    alab = ["3B", "50B"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = [0, 1]
    w = 0.36
    bt = ax.bar([xi - w / 2 for xi in x], total, w, color=BLUE, label="total params",
                edgecolor="k", linewidth=0.6)
    ba = ax.bar([xi + w / 2 for xi in x], active, w, color=GREEN, label="active / token",
                edgecolor="k", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1, 9000)
    for b, s in zip(bt, tlab):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12, s, ha="center",
                fontsize=14, fontweight="bold")
    for b, s in zip(ba, alab):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12, s, ha="center",
                fontsize=14, fontweight="bold")
    for b, p in zip(ba, pct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 2.6, p, ha="center",
                fontsize=13, color=DK_GREEN, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylabel("parameters (B, log)")
    ax.set_title("Total vs active parameters")
    ax.legend(fontsize=13, loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", which="major", ls=":", alpha=0.4)
    fig.tight_layout()
    p = f"{OUT}/slide02_total_vs_active.png"
    fig.savefig(p, dpi=200); print("wrote", p); plt.close(fig)


# --- Slide A3: OLMoE adaptation recovery ladder -------------------------------
def olmoe_ladder():
    labels = ["router\nonly", "+ RMSNorm\ngains", "+ LoRA r32", "full\nfine-tune"]
    vals = [70.7, 91.4, 93.2, 93.4]
    # emphasize the deployable recipe (+LoRA r32, index 2)
    cols = [GREEN, GREEN, DK_GREEN, GREEN]

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    bars = ax.bar(labels, vals, color=cols, width=0.68, edgecolor="k", linewidth=0.6)
    bars[2].set_linewidth(1.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                fontsize=14, fontweight="bold")
    ax.text(2, 111, "deployable recipe", ha="center", fontsize=13,
            color=DK_GREEN, fontweight="bold")
    ax.set_ylim(0, 122)
    ax.set_ylabel("% of cold-impose gap recovered")
    ax.set_title("OLMoE adaptation  (cold impose = +2.08 BPB)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.tick_params(axis="x", labelsize=13)
    fig.tight_layout()
    p = f"{OUT}/slideA3_olmoe_ladder.png"
    fig.savefig(p, dpi=200); print("wrote", p); plt.close(fig)


if __name__ == "__main__":
    total_vs_active()
    olmoe_ladder()
