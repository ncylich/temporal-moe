#!/usr/bin/env python3
"""Parse lm-eval JSON outputs (0-shot + 10-shot per model) into t19_lmeval_stderr.csv:
model,task,metric,value,stderr  (metric in acc, acc_norm). Verifies acc reproduces t19_lmeval.csv.
"""
import os, sys, csv, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT   # canonical resolver: $TMOE_ROOT, then git, then file location
RUNS=os.path.join(ROOT,"results/phase0/runs")
OUT=os.path.join(ROOT,"results/ablations/t19_lmeval_stderr.csv")
MODELS=["dense_1e19","moe_coarse_1e19","g1_tmoe_coarse_1e19","temporal_fine_g3_1e19"]
LABEL={"dense_1e19":"dense_1e19","moe_coarse_1e19":"moe_coarse_1e19",
       "g1_tmoe_coarse_1e19":"temporal_coarse_1e19","temporal_fine_g3_1e19":"temporal_fine_1e19"}

def latest_json(d):
    js=glob.glob(os.path.join(d,"**","*.json"),recursive=True)
    js=[j for j in js if "results" in open(j).read(200) or True]
    return max(js,key=os.path.getmtime) if js else None

def main():
    rows=[]
    for run in MODELS:
        res={}
        for shot in (0,10):
            d=os.path.join(RUNS,run,f"lmeval_{shot}shot")
            j=latest_json(d) if os.path.isdir(d) else None
            if not j: continue
            data=json.load(open(j)).get("results",{})
            res.update(data)
        for task,metrics in sorted(res.items()):
            for base in ("acc","acc_norm"):
                # keys look like 'acc,none' / 'acc_stderr,none'
                vk=next((k for k in metrics if k==base or k.startswith(base+",")),None)
                sk=next((k for k in metrics if k==base+"_stderr" or k.startswith(base+"_stderr,")),None)
                if vk is None or metrics.get(vk) is None: continue
                sv=metrics.get(sk)
                rows.append([LABEL[run],task,base,round(float(metrics[vk]),6),
                             (round(float(sv),6) if isinstance(sv,(int,float)) else "")])
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["model","task","metric","value","stderr"]); w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")
    # verify vs t19_lmeval.csv (dense/moe_coarse/temporal_coarse) + t19_lmeval_finegrain.csv (temporal_fine)
    ref={}
    norm={"temporal_fine_g3_1e19":"temporal_fine_1e19"}   # finegrain csv -> our label
    for rp in ("results/ablations/t19_lmeval.csv","results/ablations/t19_lmeval_finegrain.csv"):
        for r in csv.DictReader(open(os.path.join(ROOT,rp))):
            m=norm.get(r["model"],r["model"])
            ref[(m,r["task"],"acc")]=r["acc"]; ref[(m,r["task"],"acc_norm")]=r["acc_norm"]
    print("VERIFY vs t19_lmeval.csv (|delta|>0.02 flagged):")
    for m,t,me,v,s in rows:
        rv=ref.get((m,t,me),"")
        if rv not in ("",None):
            dv=abs(v-float(rv))
            flag=" <== DIFF" if dv>0.02 else ""
            if flag: print(f"  {m} {t} {me}: new {v:.4f} vs ref {float(rv):.4f} d={dv:.4f}{flag}")
    print("  (unflagged = within 0.02 of reference)")

if __name__=="__main__": main()
