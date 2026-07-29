import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open("results/ablations/ple_layer_damage.csv")))
d={r["layer"]:float(r["damage_bpb"]) for r in rows}
lay=[(int(r["layer"]),float(r["damage_bpb"])) for r in rows if r["layer"].isdigit()]
xs=[i for i,_ in lay]; ys=[v for _,v in lay]
full=d["all constrained"]; u=full/16; s=sum(ys)
fig,ax=plt.subplots(figsize=(7.6,4.6))
cols=["#0d3b66" if v>=u else "#5aa0dd" for v in ys]
ax.bar(xs,ys,color=cols,alpha=0.9)
ax.axhline(u,ls="--",color="0.4",lw=1.3,label=f"uniform share of full damage ({u:.4f})")
ax.set_xlabel("MoE layer index"); ax.set_ylabel("BPB increase vs free routing")
ax.set_title("Residency damage per layer (R=8, one layer constrained at a time)")
ax.set_xticks(xs); ax.grid(True,axis="y",ls=":",alpha=0.4); ax.legend()
fig.text(0.5,0.005,
  f"Base OLMoE, no training. BPB all-free 0.6727, all-constrained 2.7507; full damage {full:.4f}. "
  f"Sum of single-layer damage {s:.4f} = {s/full:.3f} of the full {full:.4f}, so the constraint is "
  f"mildly SUPER-additive: constraining every layer costs more than the sum of the parts.",
  ha="center",fontsize=8,wrap=True)
fig.tight_layout(rect=[0,0.07,1,1])
os.makedirs("results/phase0/figures",exist_ok=True)
out="results/phase0/figures/ple_layer_damage.png"
fig.savefig(out,dpi=200); print("wrote",out)
