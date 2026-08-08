#!/usr/bin/env python3
"""Stage 3 (1) downstream: widened 10-task lm-eval x 3 cells — base-free, base-impose-R8,
CE-adapted-R8 (merged-CE model) — acc + acc_norm + stderr, the proven pinned-venv path.
Writes results/ablations/olmoe_adapt_downstream.csv."""
import sys, types, csv, torch
for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)
sys.path.insert(0, "/workspace/olmoe-adapt/scripts")
import olmoe_residency as RES

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "sciq", "boolq", "lambada_openai", "copa"]
OUT = "/workspace/FLAME-MoE/results/ablations/olmoe_adapt_downstream.csv"
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate


def evaluate(lm):
    r = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, bootstrap_iters=1000)
    return r["results"]


# --- cell 1+2: base model, free vs impose-R8 ---
model, tok = RES.load_model()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
RES.disable_residency(); print("[ds] cell base-free ...", flush=True); res_free = evaluate(lm)
RES.enable_residency(R=8); print("[ds] cell base-impose-R8 ...", flush=True); res_imp = evaluate(lm)
del model, lm; torch.cuda.empty_cache()

# --- cell 3: CE-adapted-R8 (merged-CE model + residency) ---
from transformers import AutoModelForCausalLM, AutoTokenizer
mc = AutoModelForCausalLM.from_pretrained("/workspace/olmoe-adapt/merged_ce_model",
                                          dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
tc = AutoTokenizer.from_pretrained("/workspace/olmoe-adapt/merged_ce_model")
RES.install_patch(); RES.enable_residency(R=8)
lm2 = HFLM(pretrained=mc, tokenizer=tc, batch_size=16)
print("[ds] cell CE-adapted-R8 ...", flush=True); res_ce = evaluate(lm2)


def get(res, t, metric):
    d = res.get(t, {})
    vk = next((k for k in d if k == metric or k.startswith(metric + ",")), None)
    sk = next((k for k in d if k.startswith(metric + "_stderr")), None)
    return (float(d[vk]) if vk else None, float(d[sk]) if sk else None)


rows = []
for t in TASKS:
    for m in ("acc", "acc_norm"):
        (bf, bfs), (im, ims), (ce, ces) = get(res_free, t, m), get(res_imp, t, m), get(res_ce, t, m)
        if bf is None:
            continue
        rows.append([t, m, f"{bf:.4f}", f"{bfs:.4f}" if bfs else "", f"{im:.4f}", f"{ims:.4f}" if ims else "",
                     f"{ce:.4f}" if ce is not None else "", f"{ces:.4f}" if ces else "",
                     f"{ce-bf:+.4f}" if ce is not None else "", f"{im-bf:+.4f}"])
        print(f"{t:14}{m:9} base={bf:.4f} impose={im:.4f} CE-adapt={ce:.4f}" if ce is not None else f"{t} {m}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["# Stage 3 downstream: 10-task lm-eval, 0-shot, acc+acc_norm+stderr. Cells: base-free / base-impose-R8 / CE-adapted-R8(merged, 0.8147@R8)."])
    w.writerow(["task", "metric", "base_free", "base_free_se", "impose_R8", "impose_R8_se",
                "CE_adapt_R8", "CE_adapt_R8_se", "CEadapt_minus_base", "impose_minus_base"])
    w.writerows(rows)
print(f"[ds] wrote {OUT} ({len(rows)} rows)", flush=True)
# headline: mean acc across tasks per cell
for lbl, res in [("base-free", res_free), ("impose-R8", res_imp), ("CE-adapt-R8", res_ce)]:
    accs = [get(res, t, "acc")[0] for t in TASKS if get(res, t, "acc")[0] is not None]
    print(f"[ds] mean acc {lbl}: {sum(accs)/len(accs):.4f}", flush=True)
