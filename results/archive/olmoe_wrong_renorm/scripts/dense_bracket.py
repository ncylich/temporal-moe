#!/usr/bin/env python3
"""Stage 3 extension (orch 0122): dense-bracket downstream. IDENTICAL 10-task 0-shot lm-eval on two
released DENSE checkpoints — OLMo-1B-0724-hf (era-matched ~1.3B-active-class peer) and OLMo-7B-0724-hf
(dense upper anchor), free routing. Appends olmo1b_free/_se + olmo7b_free/_se columns to
olmoe_adapt_downstream.csv. Era-matched, NOT data-matched (OLMo-1B ~3T Dolma vs OLMoE 5.1T)."""
import sys, types, csv, torch
for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "sciq", "boolq", "lambada_openai", "copa"]
CSV = "/workspace/FLAME-MoE/results/ablations/olmoe_adapt_downstream.csv"
MODELS = [("olmo1b", "allenai/OLMo-1B-0724-hf"), ("olmo7b", "allenai/OLMo-7B-0724-hf")]


def get(res, t, metric):
    d = res.get(t, {})
    vk = next((k for k in d if k == metric or k.startswith(metric + ",")), None)
    sk = next((k for k in d if k.startswith(metric + "_stderr")), None)
    return (float(d[vk]) if vk else None, float(d[sk]) if sk else None)


evals = {}
for tag, mid in MODELS:
    print(f"[dense] loading {mid} ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    tk = AutoTokenizer.from_pretrained(mid)
    lm = HFLM(pretrained=m, tokenizer=tk, batch_size=16)
    evals[tag] = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, bootstrap_iters=1000)["results"]
    accs = [get(evals[tag], t, "acc")[0] for t in TASKS if get(evals[tag], t, "acc")[0] is not None]
    print(f"[dense] {tag} mean acc = {sum(accs)/len(accs):.4f}", flush=True)
    del m, lm; torch.cuda.empty_cache()

# merge into the existing downstream CSV (match by task+metric)
lines = open(CSV).read().splitlines()
hdr_note = [l for l in lines if l.startswith('"#') or l.startswith("#")]
data = [l for l in lines if not (l.startswith('"#') or l.startswith("#"))]
head = data[0].split(","); rows = [r.split(",") for r in data[1:] if r.strip()]
head += ["olmo1b_free", "olmo1b_se", "olmo7b_free", "olmo7b_se"]
for r in rows:
    t, metric = r[0], r[1]
    for tag in ("olmo1b", "olmo7b"):
        v, se = get(evals[tag], t, metric)
        r += [f"{v:.4f}" if v is not None else "", f"{se:.4f}" if se else ""]
hdr_note.append('"# dense-bracket (orch 0122): OLMo-1B-0724-hf (d7cbab74) + OLMo-7B-0724-hf (1ee306df), free routing, same 10-task 0-shot harness. Era-matched released checkpoints, NOT data-matched (OLMo-1B ~3T Dolma vs OLMoE 5.1T)."')
with open(CSV, "w", newline="") as f:
    w = csv.writer(f)
    for h in hdr_note:
        f.write(h + "\n")
    w.writerow(head); w.writerows(rows)
print(f"[dense] merged into {CSV}", flush=True)
# headline
o1 = [get(evals["olmo1b"], t, "acc")[0] for t in TASKS]; o7 = [get(evals["olmo7b"], t, "acc")[0] for t in TASKS]
print(f"[dense] mean acc: OLMo-1B={sum(o1)/len(o1):.4f}  OLMo-7B={sum(o7)/len(o7):.4f}  (CE-adapt-R8 was 0.5888)", flush=True)
print(f"[dense] CE-adapt vs dense-1B peer: {0.5888 - sum(o1)/len(o1):+.4f}", flush=True)
