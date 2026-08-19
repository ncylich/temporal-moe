import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=[r for r in csv.reader(open("results/ablations/router_wasserstein.csv")) if len(r)==7 and r[0]!="model" and not r[0].startswith("#")]
fig,ax=plt.subplots(figsize=(9,4.8))
colors={"qwen35":"#1e618d","gemma4":"#b03a2e","lfm":"#7d6608"}
labels={"qwen35":"Qwen3.5-35B (R8/256 = 3.1%)","gemma4":"gemma4-26B-IT (R8/128 = 6.25%)","lfm":"LFM2.5-A1B (R4/32 = 12.5%; sigmoid router)"}
for fam in ("qwen35","gemma4","lfm"):
    sub=sorted([(int(r[3]),float(r[4]),float(r[5])) for r in rows if r[1]==fam])
    if not sub: continue
    L=[x[0] for x in sub]
    frac=[l/(len(sub)-1) for l in L]
    ax.plot(frac,[x[1] for x in sub],color=colors[fam],label=labels[fam],lw=2)
    ax.plot(frac,[x[2] for x in sub],color=colors[fam],lw=1,ls="--",alpha=0.6)
ax.set_xlabel("layer depth (fraction of MoE layers)")
ax.set_ylabel("W1 (total variation), free vs constrained router dist.")
ax.set_title("Router probability displacement under R=k residency, WildChat, base IT models\n(solid: mask on free logits; dashed: end-to-end constrained forward)")
ax.legend(frameon=False,fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("results/ablations/figures/router_wasserstein.png",dpi=160)
print("wrote figure")
