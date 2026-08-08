#!/usr/bin/env python3
"""Stage 0 impose eval, downstream side: 3 quick lm-eval tasks, base vs base+mask (R=k=8), no
training. Uses lm-eval's HFLM around the OLMoE model with the residency module-swap applied;
enable/disable residency toggles the regime. VLM backends are stubbed for the transformers-5.x
AutoModelForVision2Seq rename. Writes results/ablations/olmoe_adapt_lmeval_impose.csv."""
import sys, types, json, csv, torch
for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES

TASKS = ["arc_easy", "piqa", "openbookqa"]
OUT = "/workspace/FLAME-MoE/results/ablations/olmoe_adapt_lmeval_impose.csv"

model, tok = RES.load_model()
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)


def run(masked):
    RES.enable_residency(R=8) if masked else RES.disable_residency()
    r = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, bootstrap_iters=0)
    return r["results"]


base = run(False)
imp = run(True)
rows = []
for t in TASKS:
    for base_m in ("acc", "acc_norm"):
        vk = next((k for k in base.get(t, {}) if k == base_m or k.startswith(base_m + ",")), None)
        if vk is None:
            continue
        bv = base[t][vk]; iv = imp[t].get(vk)
        rows.append([t, base_m, round(float(bv), 4), round(float(iv), 4), round(float(iv - bv), 4)])
        print(f"{t:12} {base_m:9} base={bv:.4f}  +mask={iv:.4f}  delta={iv-bv:+.4f}")
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "metric", "base", "impose_mask_R8", "delta"]); w.writerows(rows)
print(f"[write] {OUT}: {len(rows)} rows")
