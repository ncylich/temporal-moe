#!/usr/bin/env python3
"""Pair the stock lm_eval mbpp_instruct dumps of a Qwen record (3-shot, primed fence, first block,
1536 budget; no per-item pass stored) against the same record under the unified mbpp_chat producer:
re-score the stock raw generations with the stock extraction rule (reproduces the recorded aggregates
exactly), then report per-item agreement, discordance and cap-outs. Diagnostic only.

    $PY analysis/residency/mbpp_stock_pair.py
"""
import json, os, re, subprocess, math, sys
from datasets import load_dataset
plist=list(load_dataset("google-research-datasets/mbpp","full",split="test"))
probs={f"mbpp/{p['task_id']}": p for p in plist}
idx2doc={str(i): f"mbpp/{p['task_id']}" for i,p in enumerate(plist)}
PAT=r"```(?:\w+)?\n?(.*?)\n?```"
def stock_extract(t):
    m=re.findall(PAT,"```"+t,re.DOTALL)
    if not m: m=re.findall(PAT,re.sub(r"```python","```",t),re.DOTALL)
    return m[0] if m else ""
def score(codes,docs):
    p=f"/tmp/mbpp_stock_pair_{os.getpid()}.json"
    tests=[(probs[d].get("test_setup_code") or "")+"\n"+"\n".join(probs[d]["test_list"]) for d in docs]
    json.dump({"preds":[[c] for c in codes],"tests":tests},open(p,"w"))
    out=subprocess.run(["/workspace/venv_fla/bin/python","analysis/residency/heg_scorer.py",p],capture_output=True,text=True)
    lines=[l for l in out.stdout.splitlines() if l.startswith("ITEMS")]
    if not lines: print(out.stdout[-500:], out.stderr[-800:]); sys.exit(1)
    return [c=="1" for c in lines[0].split()[1]]
def load(f):
    d=json.load(open(f)); return d["items"] if isinstance(d,dict) else d
G="results/ablations/genbench_samples/"
for stock,uni,arms in (("qwen35_code_base","qwen35_instruct_mbpp",("free","R8","R32")),("qwen35_ce_online_klT2_lr3e-5_rho0_code","qwen35_ce_online_klT2_lr3e-5_rho0_mbpp",("R8","R32"))):
    for a in arms:
        s=load(f"{G}{stock}_{a}_mbpp_instruct.json"); u={x["doc"]:x for x in load(f"{G}{uni}_{a}_mbpp_chat.json")}
        docs=[idx2doc[str(x.get("doc_id", x["doc"]))] for x in s]; sp=dict(zip(docs,score([stock_extract(x["raw"]) for x in s],docs)))
        scap={d:int(x["gen_toks"])>=1536 for d,x in zip(docs,s)}
        up={d:str(u[d]["pass"])=="True" for d in docs}; ucap={d:str(u[d]["hit_cap"])=="True" or str(u[d]["unfinished"])=="True" for d in docs}
        n=len(docs); b01=[d for d in docs if not sp[d] and up[d]]; b10=[d for d in docs if sp[d] and not up[d]]
        z=(len(b01)-len(b10))/math.sqrt(len(b01)+len(b10))
        print(f"{stock} {a}: stock rescored {sum(sp.values())/n:.3f}  unified {sum(up.values())/n:.3f}  agree {n-len(b01)-len(b10)}/{n}  "
              f"stock-fail/unified-pass {len(b01)} (stock at cap {sum(scap[d] for d in b01)})  stock-pass/unified-fail {len(b10)} (unified capped {sum(ucap[d] for d in b10)})  paired z {z:+.2f}")
