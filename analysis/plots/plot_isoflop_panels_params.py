#!/usr/bin/env python3
"""Fig. 3 (rebuttal version): isoFLOP panels with every size counted WITH embeddings.

Each curve panel labels its swept sizes in two rows: active / dense parameters (bold) and total MoE
parameters. Embeddings are input + output (untied in every run): vocab 16,000 at 1e16/1e17, 50,304 at
1e18/1e19. BPB values are the same as plot_isoflop_panels.py. Sizes come from analysis/shapes.py
(shapes sm1,s0,s1 | s1,s2,s3 | s192f,s38m,s512f | s19opt); the 1e19 counts match the checkpoints'
train.log argument dumps (N=184,060,800 coarse, 185,766,400 fine, non-embedding).
"""
# Fig. 3 variants: x = active params incl. embeddings, plus total MoE params. Data copied from
# analysis/plots/plot_isoflop_panels.py (BPB values unchanged); sizes from analysis/shapes.py + run logs.
import sys, os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
sys.path.insert(0,os.path.join(REPO,"analysis")); sys.argv=sys.argv[:1]
import shapes as S
DENSE_C="#7f7f7f"; MOE_COARSE,MOE_FINE="#f4756b","#9e0f14"; TMP_COARSE,TMP_FINE="#7ecb7e","#0b5c1c"
BPB={
 "1e16":{"dense":[1.534,1.519,1.591],"moe_c":[1.4766,1.447,1.540],"moe_f":[1.4786,1.4585,1.5352],"tmp_c":[1.4872,1.4599,1.5473],"tmp_f":[1.4976,1.4753,1.5861]},
 "1e17":{"dense":[1.361,1.341,1.408],"moe_c":[1.2803,1.269,1.289],"moe_f":[1.2846,1.2708,1.2815],"tmp_c":[1.3027,1.2821,1.3061],"tmp_f":[1.3065,1.2873,1.3129]},
 "1e18":{"dense":[1.4072,1.3911,1.4256],"moe_c":[1.3272,1.3175,1.3766],"moe_f":[1.3570,1.3478,1.4174],"tmp_c":[1.3198,1.3122,1.3762],"tmp_f":[1.3379,1.3339,1.4047]},
}
SH={"1e16":(["sm1","s0","s1"],16000),"1e17":(["s1","s2","s3"],16000),"1e18":(["s192f","s38m","s512f"],50304),"1e19":(["s19opt"],50304)}
def total_ne(h,L,ffn,moe_ffn,g):
    nexp=64*g; mf=2*round((moe_ffn/g)/2) if g!=1 else moe_ffn; t=0
    for l in range(L):
        t+=4*h*h+2*h; t+=3*h*ffn if l==0 else h*nexp+3*h*2*moe_ffn+nexp*3*h*mf
    return t
def sizes(bud,g):
    names,V=SH[bud]; out=[]
    for n in names:
        sh=S.SHAPES[n]; E=2*V*sh['h']
        out.append(((S.active_nonembed(**sh,grain=g)+E)/1e6,(total_ne(**sh,g=g)+E)/1e6))
    return out
STYLE=[("dense",DENSE_C,1.4),("moe_c",MOE_COARSE,1.9),("moe_f",MOE_FINE,1.9),("tmp_c",TMP_COARSE,1.9),("tmp_f",TMP_FINE,2.9)]
LEG=["dense","standard MoE · coarse","standard MoE · fine","Temporal MoE · coarse","Temporal MoE · fine (ours)"]
DODGE={"dense":0,"moe_c":-0.9,"moe_f":-0.3,"tmp_c":0.3,"tmp_f":0.9}
def fmt(m): return f"{m/1000:.2f}B" if m>=1000 else (f"{m:.0f}M" if m>=20 else f"{m:.1f}M")
def curve(ax,bud,mode,xlabel=True,ylabel=True):
    c=sizes(bud,1); f=sizes(bud,3)
    for z,(k,col,lw) in enumerate(STYLE):
        xs=[s[0] for s in (f if k.endswith("_f") else c)]; ys=BPB[bud][k]
        ax.plot(xs,ys,alpha=0)
        tr=ax.transData+ScaledTranslation(0,DODGE[k]/72,ax.figure.dpi_scale_trans)
        ax.plot(xs,ys,"-o",color=col,mfc=col,mec="white",mew=0.9,ms=7 if k=="tmp_f" else 6,lw=lw,zorder=2+z,transform=tr)
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.tick_params(axis="x",which="minor",length=0)
    ticks=[s[0] for s in c]; ax.set_xticks(ticks); ax.grid(True,which="major",ls=":",alpha=0.4)
    lo,hi=min(ticks)/1.25,max(ticks)*1.25; ax.set_xlim(lo,hi)
    title=f"$10^{{{bud[2:]}}}$ FLOPs"
    if mode=="twin":
        ax.set_xticklabels([fmt(s[0]) for s in c])
        top=ax.twiny(); top.set_xscale("log"); top.set_xlim(lo,hi); top.set_xticks(ticks)
        top.set_xticklabels([fmt(s[1]) for s in c],fontsize=8.5,color="#555"); top.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); top.tick_params(axis="x",which="minor",length=0)
        if xlabel: ax.set_xlabel("active params")
        top.set_xlabel("total MoE params",fontsize=8.5,color="#555",labelpad=1)
        ax.set_title(title,pad=26,fontsize=10.5)
    else:
        ax.set_xticklabels([f"{fmt(s[0])}\n{fmt(s[1])}" for s in c],fontsize=8.5)
        if xlabel: ax.set_xlabel("active / total MoE params")
        ax.set_title(title)
    if ylabel: ax.set_ylabel("test BPB")
P19=[("dense",1.1260,DENSE_C),("Temporal MoE\ncoarse",1.0680,TMP_COARSE),("Temporal MoE\nfine",1.0655,TMP_FINE),("standard MoE\ncoarse",1.0514,MOE_COARSE),("standard MoE\nfine",1.0604,MOE_FINE)]
def bars(ax,mode,ylabel=False):
    a,t=sizes("1e19",3)[0]; ac=sizes("1e19",1)[0][0]
    labels=[p[0] for p in P19]
    b=ax.bar(range(5),[p[1] for p in P19],color=[p[2] for p in P19],width=0.66,edgecolor="k",linewidth=[1.6 if l=="Temporal MoE\nfine" else 0.6 for l in labels])
    for bb,(l,v,_) in zip(b,P19): ax.text(bb.get_x()+bb.get_width()/2,v+0.002,f"{v:.3f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_ylim(1.0,1.16); ax.grid(True,axis="y",ls=":",alpha=0.4)
    ax.set_xticks(range(5)); ax.set_xticklabels([l.replace(" MoE\n","\nMoE\n") for l in labels],fontsize=7.5)
    if mode=="twin":
        ax.set_title("$10^{19}$ FLOPs",pad=22)
        ax.text(0.99,0.97,f"MoE: {fmt(ac)}–{fmt(a)} active, {fmt(t)} total\ndense: {fmt(ac)}",transform=ax.transAxes,ha="right",va="top",fontsize=8.5,color="#555")
    else:
        ax.set_title(f"$10^{{19}}$ FLOPs · MoE {fmt(a)} / {fmt(t)}",fontsize=10)
    if ylabel: ax.set_ylabel("test BPB")
def grid(mode,name,note):
    plt.rcParams.update({"font.size":9.5,"axes.titlesize":11,"axes.labelsize":10})
    fig,ax=plt.subplots(2,2,figsize=(7.0,5.4 if mode=="twin" else 4.8))
    curve(ax[0][0],"1e16",mode); curve(ax[0][1],"1e17",mode,ylabel=False)
    curve(ax[1][0],"1e18",mode); bars(ax[1][1],mode)
    h=[Line2D([0],[0],color=c,lw=3.2 if k=="tmp_f" else 2.6) for k,c,_ in STYLE]
    lg=fig.legend(h,LEG,ncol=5,loc="upper center",fontsize=9.6,frameon=False,bbox_to_anchor=(0.5,1.008),columnspacing=0.9,handlelength=1.5); lg.get_texts()[-1].set_fontweight("bold")
    fig.tight_layout(rect=[0,0,1,0.925]); fig.subplots_adjust(wspace=0.16,hspace=1.0 if mode=="twin" else 0.72)
    fig.savefig(f"{OUT}/{name}.png",dpi=170,bbox_inches="tight",pad_inches=0.02); plt.close(fig)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation, blended_transform_factory as blend
ACT="#000000"; TOT="#000000"
def series(ax,bud):
    c=sizes(bud,1); f=sizes(bud,3)
    for z,(k,col,lw) in enumerate(STYLE):
        xs=[s[0] for s in (f if k.endswith("_f") else c)]; ys=BPB[bud][k]
        ax.plot(xs,ys,alpha=0)
        tr=ax.transData+ScaledTranslation(0,DODGE[k]/72,ax.figure.dpi_scale_trans)
        ax.plot(xs,ys,"-o",color=col,mfc=col,mec="white",mew=0.9,ms=7 if k=="tmp_f" else 6,lw=lw,zorder=2+z,transform=tr)
    ax.set_xscale("log"); ticks=[s[0] for s in c]; lo,hi=min(ticks)/1.3,max(ticks)*1.3
    ax.set_xlim(lo,hi); ax.set_xticks(ticks); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True,which="major",ls=":",alpha=0.4); ax.set_title(f"$10^{{{bud[2:]}}}$ FLOPs")
    return c
def rows(ax,c,headers):
    # two labelled rows under the axis: active/dense (dark) above, MoE total (red, italic) below
    ax.set_xticklabels([]); tb=blend(ax.transData,ax.transAxes); ta=ax.transAxes
    for a,t in c:
        ax.text(a,-0.07,fmt(a),transform=tb,ha="center",va="top",fontsize=9,color=ACT,fontweight="bold")
        ax.text(a,-0.20,fmt(t),transform=tb,ha="center",va="top",fontsize=9,color=TOT)
    if headers:
        ax.text(-0.03,-0.07,"active / dense",transform=ta,ha="right",va="top",fontsize=8.5,color=ACT,fontweight="bold")
        ax.text(-0.03,-0.20,"MoE total",transform=ta,ha="right",va="top",fontsize=8.5,color=TOT)
def twin(ax,c,headers):
    ax.set_xticklabels([fmt(a) for a,_ in c],fontsize=9,color=ACT,fontweight="bold")
    top=ax.twiny(); top.set_xscale("log"); top.set_xlim(ax.get_xlim()); top.set_xticks([a for a,_ in c])
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xticklabels([fmt(t) for _,t in c],fontsize=9,color=TOT); top.tick_params(axis="x",colors=TOT,pad=1)
    for lab in top.get_xticklabels(): lab.set_color(TOT)
    top.spines["top"].set_color(TOT); top.spines["top"].set_linewidth(1.4)
    ax.set_title(ax.get_title(),pad=20)
    if headers:
        ax.text(-0.03,1.035,"MoE total",transform=ax.transAxes,ha="right",va="bottom",fontsize=8.5,color=TOT)
        ax.text(-0.03,-0.045,"active / dense",transform=ax.transAxes,ha="right",va="top",fontsize=8.5,color=ACT,fontweight="bold")
def bars(ax):
    labels=[p[0] for p in P19]
    b=ax.bar(range(5),[p[1] for p in P19],color=[p[2] for p in P19],width=0.66,edgecolor="k",linewidth=[1.6 if l=="Temporal MoE\nfine" else 0.6 for l in labels])
    for bb,(l,v,_) in zip(b,P19): ax.text(bb.get_x()+bb.get_width()/2,v+0.002,f"{v:.3f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_ylim(1.0,1.19); ax.grid(True,axis="y",ls=":",alpha=0.4)
    ax.set_xticks(range(5)); ax.set_xticklabels([l.replace("Temporal","Temp.").replace("standard","std.").replace(" MoE\n","\nMoE\n") for l in labels],fontsize=7.5)
    ax.set_title("$10^{19}$ FLOPs")
    ax.text(0.98,0.97,"265M active / dense",transform=ax.transAxes,ha="right",va="top",fontsize=9,color=ACT,fontweight="bold")
    ax.text(0.98,0.88,"1.26B MoE total",transform=ax.transAxes,ha="right",va="top",fontsize=9,color=TOT)
def make(kind,name):
    plt.rcParams.update({"font.size":9.5,"axes.titlesize":11,"axes.labelsize":10})
    fig,ax=plt.subplots(2,2,figsize=(7.0,5.0))
    for (i,j),bud in zip([(0,0),(0,1),(1,0)],["1e16","1e17","1e18"]):
        c=series(ax[i][j],bud)
        (rows if kind=="rows" else twin)(ax[i][j],c,headers=(j==0))
        if j==0: ax[i][j].set_ylabel("test BPB")
    bars(ax[1][1])
    h=[Line2D([0],[0],color=c,lw=3.2 if k=="tmp_f" else 2.6) for k,c,_ in STYLE]
    lg=fig.legend(h,LEG,ncol=5,loc="upper center",fontsize=9.6,frameon=False,bbox_to_anchor=(0.5,1.008),columnspacing=0.9,handlelength=1.5); lg.get_texts()[-1].set_fontweight("bold")
    fig.tight_layout(rect=[0.06,0,1,0.925]); fig.subplots_adjust(wspace=0.16,hspace=0.75 if kind=="rows" else 0.85)
    for d in (f"{REPO}/results/phase0/figures", f"{REPO}/paper/figures"):
        fig.savefig(f"{d}/{name}.png",dpi=200,bbox_inches="tight",pad_inches=0.03); fig.savefig(f"{d}/{name}.pdf",bbox_inches="tight",pad_inches=0.03); print("wrote",d,name)
    plt.close(fig)
make("rows","isoflop_panels_params_nocaption")