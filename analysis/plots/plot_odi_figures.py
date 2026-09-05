#!/usr/bin/env python3
"""ODI workshop variants of the four main-text figures.

The workshop paper displays these figures at 0.38 to 0.58 of a 5.5in text column,
smaller than the main paper does, so the shared files print with tiny text there.
These variants redraw the same content on canvases sized near the workshop display
widths, with fonts chosen to print at 6.5 to 8pt and with padding checked by eye,
and write to paper/odi/figures/ under *_odi names so the shared files are untouched.

Data is the same as the shared producers (plot_isoflop_panels.py,
plot_serving_context.py, plot_bandwidth_timeline.py, swap_diagram.py); the isoFLOP
values and serving sweep points are copied from those scripts, which document their
result-file sources.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import ScaledTranslation

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "paper", "odi", "figures")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- swap diagram
def swap_diagram():
    BLUE, GREEN = "#0B5394", "#1f9d55"
    GREEN_FILL, GREY_FILL, GREY_EDGE = "#e6f6ec", "#eef2f6", "#d3dbe4"
    POOL_FILL, POOL_EDGE, INK = "#f6f9fc", "#e1e9f1", "#4a4f57"
    N, STAYS, EVICTED, ADMITTED = 5, 0, 3, 4
    SLOT_W, SLOT_H, GAP, TOP = 52.0, 11.5, 2.6, 18.0
    PITCH = SLOT_H + GAP
    BOTTOM = TOP + (N - 1) * PITCH + SLOT_H
    LX, RX = 10.0, 196.0

    def sy(i):
        return TOP + i * PITCH

    def smid(i):
        return sy(i) + SLOT_H / 2

    W, H = 258.0, 110.0
    fig, ax = plt.subplots(figsize=(W / 95, H / 95))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def box(x, y, w, h, fill, edge, lw=0.8, dashed=False, r=2.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor=fill, edgecolor=edge, linewidth=lw,
                                    linestyle=(0, (2.4, 1.7)) if dashed else "solid",
                                    zorder=2))

    def arrow(a, b, dashed=False, lw=1.1):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                     color=BLUE, linewidth=lw, zorder=3,
                                     linestyle=(0, (2.6, 2.0)) if dashed else "solid",
                                     shrinkA=0, shrinkB=0))

    for x, label in ((LX, "TOKEN $t$"), (RX, "TOKEN $t{+}1$")):
        box(x, 17, SLOT_W + 4, BOTTOM - 17 + 5, POOL_FILL, POOL_EDGE, 0.7, r=3)
        ax.text(x + (SLOT_W + 4) / 2, 9, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=BLUE)

    for i in range(N):
        for x, side in ((LX + 2, "L"), (RX + 2, "R")):
            if i == STAYS:
                box(x, sy(i), SLOT_W, SLOT_H, GREEN_FILL, GREEN, 1.1)
            elif side == "L" and i == EVICTED:
                box(x, sy(i), SLOT_W, SLOT_H, GREEN_FILL, GREEN, 1.1, dashed=True)
            elif side == "R" and i == ADMITTED:
                box(x, sy(i), SLOT_W, SLOT_H, GREEN, GREEN, 1.1)
            else:
                box(x, sy(i), SLOT_W, SLOT_H, GREY_FILL, GREY_EDGE, 0.7)

    y = smid(STAYS)
    ax.plot([LX + SLOT_W + 4, RX], [y, y], color="#8fbf9f", lw=0.7,
            dashes=(2.0, 2.4), zorder=1)

    rx, ry, rw, rh = 104.0, 40.0, 80.0, 24.0
    box(rx, ry, rw, rh, "#ffffff", BLUE, 1.3, r=3)
    ax.text(rx + rw / 2, ry + rh / 2, "ROUTER", ha="center", va="center",
            fontsize=10, fontweight="bold", color=BLUE)
    arrow((rx + rw / 2, ry - 13.5), (rx + rw / 2, ry - 0.8), lw=0.9)

    arrow((LX + SLOT_W + 5, smid(EVICTED)), (rx - 1.5, ry + rh - 6), dashed=True)
    ax.text(82, smid(EVICTED) + 9, "evict", ha="center", va="center",
            fontsize=8, color=BLUE)
    arrow((rx + rw + 1.5, ry + rh - 4), (RX + 1, smid(ADMITTED)))
    ax.text(170, smid(ADMITTED) + 7, "admit", ha="center", va="center",
            fontsize=8, color=BLUE)

    # key: two swatches per row so the larger labels keep clear air between entries
    kw, kh = 9.0, 6.4
    for x, ky, label, fill, edge, dashed in (
            (28, 99.0, "resident", GREEN_FILL, GREEN, False),
            (128, 99.0, "evicted", GREEN_FILL, GREEN, True),
            (28, 109.0, "admitted", GREEN, GREEN, False),
            (128, 109.0, "not resident", GREY_FILL, GREY_EDGE, False)):
        box(x, ky, kw, kh, fill, edge, 0.9 if fill != GREY_FILL else 0.7,
            dashed=dashed, r=1.2)
        ax.text(x + kw + 3.5, ky + kh / 2, label, ha="left", va="center",
                fontsize=7.5, color=INK)
    ax.set_ylim(120, 0)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ("pdf", "png"):
        p = f"{OUT}/swap_diagram_odi.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.02, transparent=False)
        print("wrote", p)
    plt.close(fig)


# ----------------------------------------------------------- bandwidth timeline
def bandwidth_timeline():
    FS = 15
    LIGHT, EDGE, DARK, GREY = "#cfe3f5", "#4a7fb5", "#0d3b66", "#666666"
    fig, ax = plt.subplots(figsize=(6.2, 1.62))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    n, x0, w, gap, y, h = 5, 0.15, 1.30, 0.06, 2.05, 0.68
    for i in range(n):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=LIGHT, edgecolor=EDGE,
                               lw=1.2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, r"$t_{\mathrm{fast}}$",
                ha="center", va="center", fontsize=FS - 2, color=DARK)
    top_end = x0 + n * (w + gap) - gap
    ax.text(x0, y + h + 0.16, r"$k-1$ resident experts read from fast memory",
            fontsize=FS - 1, color=DARK, weight="bold", va="bottom")

    ax.add_patch(FancyArrow(x0, 1.62, top_end - x0 + 0.55, 0, width=0.012,
                            head_width=0.17, head_length=0.18, color=DARK))
    ax.text(top_end + 0.72, 1.62, "time", fontsize=FS - 3, color=GREY, va="center")
    ax.text(9.9, 2.35, r"$t_{\mathrm{fast}} = b/\mathrm{BW}_{\mathrm{fast}}$",
            fontsize=FS - 4, color=DARK, ha="right", va="center")
    ax.text(9.9, 0.94, r"$t_{\mathrm{slow}} = b/\mathrm{BW}_{\mathrm{slow}}$",
            fontsize=FS - 4, color=DARK, ha="right", va="center")

    sw = top_end - x0 - 0.55
    ax.add_patch(Rectangle((x0, 0.60), sw, h, facecolor=DARK, edgecolor=DARK))
    ax.text(x0 + sw / 2, 0.60 + h / 2,
            r"incoming expert streams for $t_{\mathrm{slow}}$",
            ha="center", va="center", fontsize=FS - 2.5, color="white")
    ax.text(x0 + sw + 0.14, 0.60 + h / 2, "hidden", fontsize=FS - 4, color=GREY,
            va="center", style="italic")

    ax.text(x0, 0.06,
            r"free when $(k-1)\,t_{\mathrm{fast}} \geq t_{\mathrm{slow}}$,  i.e. "
            r"$k-1 \geq \mathrm{BW}_{\mathrm{fast}}/\mathrm{BW}_{\mathrm{slow}}$",
            fontsize=FS - 3, color=DARK, ha="left", va="bottom")

    fig.tight_layout(pad=0.15)
    p = f"{OUT}/bandwidth_timeline_odi.png"
    fig.savefig(p, dpi=300)
    print("wrote", p)
    plt.close(fig)


# --------------------------------------------------------------- isoFLOP panels
DENSE_C = "#7f7f7f"
MOE_COARSE, MOE_FINE = "#f4756b", "#9e0f14"
TMP_COARSE, TMP_FINE = "#7ecb7e", "#0b5c1c"
P16 = {
    "dense": {0.770: 1.534, 1.361: 1.519, 3.812: 1.591},
    "moe_c": {0.770: 1.4766, 1.361: 1.447, 3.812: 1.540},
    "moe_f": {0.81: 1.4786, 1.42: 1.4585, 3.91: 1.5352},
    "tmp_c": {0.770: 1.4872, 1.361: 1.4599, 3.812: 1.5473},
    "tmp_f": {0.81: 1.4976, 1.42: 1.4753, 3.91: 1.5861},
}
P17 = {
    "dense": {3.812: 1.361, 8.115: 1.341, 14.774: 1.408},
    "moe_c": {3.812: 1.2803, 8.115: 1.269, 14.774: 1.289},
    "moe_f": {3.91: 1.2846, 8.23: 1.2708, 15.09: 1.2815},
    "tmp_c": {3.812: 1.3027, 8.115: 1.2821, 14.774: 1.3061},
    "tmp_f": {3.91: 1.3065, 8.23: 1.2873, 15.09: 1.3129},
}
P18 = {
    "dense": {6.88: 1.4072, 12.19: 1.3911, 48.50: 1.4256},
    "moe_c": {6.88: 1.3272, 12.19: 1.3175, 48.50: 1.3766},
    "moe_f": {6.88: 1.3570, 12.19: 1.3478, 48.50: 1.4174},
    "tmp_c": {6.88: 1.3198, 12.19: 1.3122, 48.50: 1.3762},
    "tmp_f": {6.88: 1.3379, 12.19: 1.3339, 48.50: 1.4047},
}
P19 = [("dense", 1.1260, DENSE_C), ("temporal\ncoarse", 1.0680, TMP_COARSE),
       ("temporal\nfine", 1.0655, TMP_FINE), ("full MoE\ncoarse", 1.0514, MOE_COARSE),
       ("full MoE\nfine", 1.0604, MOE_FINE)]
STYLE = [("dense", DENSE_C, 1.4), ("moe_c", MOE_COARSE, 1.8), ("moe_f", MOE_FINE, 1.8),
         ("tmp_c", TMP_COARSE, 1.8), ("tmp_f", TMP_FINE, 2.9)]
LEG = ["dense", "MoE · coarse", "MoE · fine", "temporal · coarse", "temporal · fine (ours)"]
# sub-point vertical dodge so coincident series render side by side rather than
# stacked; a pure drawing offset (max 0.9pt), the plotted values are untouched
DODGE_PT = {"dense": 0.0, "moe_c": -0.9, "moe_f": -0.3, "tmp_c": 0.3, "tmp_f": 0.9}
TICKS = {"1e16": ([0.77, 1.4, 3.85], ["0.8", "1.4", "3.9"]),
         "1e17": ([3.85, 8.17, 14.9], ["3.9", "8.1", "15"]),
         "1e18": ([6.88, 12.19, 48.50], ["6.9", "12", "49"])}


def isoflop_panels():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})

    def curve_panel(ax, data, title, ticks=None, xlabel=True, ylabel=True):
        for z, (key, color, lw) in enumerate(STYLE):
            d = data[key]
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], alpha=0)     # register data limits
            tr = ax.transData + ScaledTranslation(0, DODGE_PT[key] / 72,
                                                  ax.figure.dpi_scale_trans)
            ax.plot(xs, [d[x] for x in xs], "-o", color=color, mfc=color,
                    mec="white", mew=0.9, ms=6 if key == "tmp_f" else 5, lw=lw,
                    zorder=2 + z, transform=tr)
        ax.set_xscale("log")
        if ticks:
            ax.set_xticks(ticks[0])
            ax.set_xticklabels(ticks[1])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.grid(True, which="major", ls=":", alpha=0.4)
        ax.set_title(title, pad=4)
        if xlabel:
            ax.set_xlabel("active params (M)", labelpad=1.5)
        if ylabel:
            ax.set_ylabel("test BPB", labelpad=2)

    def bar_panel(ax, title, ylabel=True):
        labels = [p[0] for p in P19]
        vals = [p[1] for p in P19]
        cols = [p[2] for p in P19]
        bars = ax.bar(labels, vals, color=cols, width=0.66, edgecolor="k",
                      linewidth=[1.6 if l == "temporal\nfine" else 0.6 for l in labels])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(1.0, 1.17)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.set_title(title, pad=4)
        if ylabel:
            ax.set_ylabel("test BPB", labelpad=2)
        # colours carry the series identity through the shared legend; the five
        # two-line tick labels collide at this panel width, so no tick labels
        ax.set_xticks([])

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.0))
    curve_panel(axes[0][0], P16, "$10^{16}$ FLOPs · 16k vocab",
                ticks=TICKS["1e16"], xlabel=False)
    curve_panel(axes[0][1], P17, "$10^{17}$ FLOPs · 16k vocab",
                ticks=TICKS["1e17"], xlabel=False, ylabel=False)
    curve_panel(axes[1][0], P18, "$10^{18}$ FLOPs · 50k vocab", ticks=TICKS["1e18"])
    bar_panel(axes[1][1], "$10^{19}$ FLOPs · 50k vocab", ylabel=False)
    handles = [Line2D([0], [0], color=c, lw=3.2 if k == "tmp_f" else 2.2)
               for k, c, _ in STYLE]
    leg = fig.legend(handles, LEG, ncol=5, loc="upper center", fontsize=9.3,
                     frameon=False, bbox_to_anchor=(0.5, 1.002),
                     columnspacing=0.8, handlelength=1.3)
    leg.get_texts()[-1].set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(wspace=0.18, hspace=0.52)
    p = f"{OUT}/isoflop_panels_odi.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("wrote", p)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    matplotlib.use("Agg")


# --------------------------------------------------------- serving context sweep
def serving_context():
    ctx = [1024, 2048, 4096, 8192, 16384]
    A_pp = [6756, 6625, 6568, 6284, 5608]
    C_pp = [2906, 3570, 3715, 3618, 3568]
    A_tg = [203.4, 203.0, 201.9, 201.1, 202.4]
    C_tg = [161.7, 159.6, 159.2, 159.5, 159.3]
    A_vram = [8.174, 8.840, 8.936, 9.128, 9.512]
    C_vram = [2.172, 2.918, 3.014, 3.206, 3.590]
    CEIL, DEPL, VRC = "#5aa0dd", "#2ca02c", "0.45"

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10})

    def panel(ax, tps_A, tps_C, title):
        axr = ax.twinx()
        axr.plot(ctx, A_vram, "--o", color=CEIL, ms=4, lw=1.4, alpha=0.85)
        axr.plot(ctx, C_vram, "--s", color=DEPL, ms=4, lw=1.4, alpha=0.85)
        axr.set_ylim(0, 11)
        axr.set_ylabel("peak VRAM (GB)", color=VRC, labelpad=3)
        axr.tick_params(axis="y", labelcolor=VRC)
        axr.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.plot(ctx, tps_A, "-o", color=CEIL, ms=5, lw=2.0, label="all-resident MoE")
        ax.plot(ctx, tps_C, "-s", color=DEPL, ms=5, lw=2.0, label="temporal (ours)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ctx)
        ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
        ax.set_xlabel("context length (tokens)", labelpad=1.5)
        ax.set_ylabel("throughput (tok/s)", labelpad=2)
        ax.set_ylim(0, max(tps_A) * 1.14)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(title, pad=4)
        ax.grid(True, ls=":", alpha=0.35)
        return axr

    fig, (axp, axd) = plt.subplots(1, 2, figsize=(6.4, 2.42))
    panel(axp, A_pp, C_pp, "Prefill")
    axr = panel(axd, A_tg, C_tg, "Decode (100-token gen)")
    h, l = axd.get_legend_handles_labels()
    h += [Line2D([0], [0], color="0.35", lw=1.6, ls="-"),
          Line2D([0], [0], color="0.35", lw=1.4, ls="--")]
    l += ["throughput (left axis)", "peak VRAM (right axis)"]
    fig.legend(h, l, ncol=2, loc="upper center", fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.83], w_pad=1.6)
    p = f"{OUT}/serving_context_odi.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("wrote", p)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    matplotlib.use("Agg")


if __name__ == "__main__":
    swap_diagram()
    bandwidth_timeline()
    isoflop_panels()
    serving_context()
