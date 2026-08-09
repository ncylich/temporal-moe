#!/usr/bin/env python3
"""Downstream accuracy at every residency the BPB ladder measured, one stack per invocation.

Ten 0-shot tasks (limit 500), free anchor + each R cell. PROTOCOL: batched lm_eval, i.e.
the pad-warmed variant every published downstream table used — measured sensitivity vs
unbatched cold-fill is ~2 points per task, model-dependent sign (padcheck 08-09: OLMoE R8
arc_easy 0.625 bs8 vs 0.605 bs1; gpt-oss-20b 0.63 bs16 vs 0.67 bs1). Trends across R are
comparable within-protocol.

    downstream_ladder.py --stack olmoe --cells 12,16,24,32 --batch-size 8
    downstream_ladder.py --stack gemma4 --cells free,16,12,8 --batch-size 16
"""
import argparse
import csv
import os
import sys
import types

for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "sciq", "winogrande",
         "openbookqa", "boolq", "lambada_openai", "copa"]


def load_stack(stack):
    """Returns (model, tok, set_R) where set_R(None) = free routing, set_R(int) = residency."""
    if stack == "olmoe":
        import residency as RES
        model, tok = RES.load_model()
        def set_R(R):
            if R is None:
                RES._CFG.update(on=False)
            else:
                RES._CFG.update(on=True, R=R, evict="min_logit", gate_mass="preserve",
                                collect_telem=False, R_map=None, swaps=1)
                RES.set_free_layers(None)
        return model, tok, set_R
    if stack in ("qwen3", "qwen3_5"):
        import residency as RES
        import residency_qwen as RQ
        import train_qwen as TQ
        FAM = TQ.resolve(stack)
        model, tok = RQ.load_model(path=FAM["model"], family=stack)
        def set_R(R):
            if R is None:
                RES._CFG.update(on=False)
            else:
                RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=False,
                                R_map=None, swaps=1)
                RES.set_free_layers(None)
        return model, tok, set_R
    # gemma4 / lfm25: the granularity_ladder patches + plain transformers load
    import granularity_ladder as GL
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = GL.MODELS[stack]
    model = AutoModelForCausalLM.from_pretrained(M["path"], dtype=torch.bfloat16).to("cuda")
    tok = AutoTokenizer.from_pretrained(M["path"])
    {"lfm": GL.patch_lfm, "gemma4": GL.patch_gemma4}[M["arch"]]()
    def set_R(R):
        GL.CFG.update(on=R is not None, R=R or 0)
    return model, tok, set_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True,
                    choices=("olmoe", "qwen3", "qwen3_5", "gemma4", "lfm25"))
    ap.add_argument("--cells", required=True, help="comma list: 'free' and/or R ints")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=500)
    A = ap.parse_args()

    model, tok, set_R = load_stack(A.stack)
    model.eval()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "results", "ablations", "downstream_ladder.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Downstream accuracy at each BPB-ladder residency: ten 0-shot tasks, limit '
                 '500, batched lm_eval (pad-warmed protocol, matches every published downstream '
                 'table; ~2pt/task sensitivity vs unbatched cold-fill, model-dependent sign - '
                 'see producer header). Producer: analysis/ple/downstream_ladder.py"\n')
        w.writerow(["stack", "cell", "R"] + TASKS + ["mean_acc"])

    for cell in A.cells.split(","):
        R = None if cell == "free" else int(cell)
        set_R(R)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=A.batch_size)
        res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                              limit=(A.limit or None), bootstrap_iters=1000)["results"]
        accs = {t: res.get(t, {}).get("acc,none", res.get(t, {}).get("acc_norm,none"))
                for t in TASKS}
        mean = sum(accs.values()) / len(accs)
        w.writerow([A.stack, cell, R or ""] + [f"{accs[t]:.4f}" for t in TASKS]
                   + [f"{mean:.4f}"])
        fh.flush()
        print(f"[dsladder] {A.stack} cell={cell} mean_acc={mean:.4f}", flush=True)
    fh.close()
    print(f"DSLADDER {A.stack} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
