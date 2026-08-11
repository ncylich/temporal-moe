#!/usr/bin/env python3
"""Generative benchmarks under decode-time residency on the vLLM stack.

Same protocol as instruct_genbench.py (batch-fair: ALL arms share the identical engine,
continuous batching, chat template, greedy, prefill free, rule on generated tokens) at
continuous-batching speed. Stack requirements enforced here: in-process engine core,
enforce_eager, prefix caching OFF, gate-mass preserve correction (vllm_glue).

    instruct_genbench_vllm.py --model olmoe_instruct --arms free,R8 \
        --tasks gsm8k_cot_zeroshot=200,ifeval=200 [--path SHM_DIR]
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402
from instruct_selfce import MODELS                                   # noqa: E402
from paths import ABLATIONS                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tasks", default="gsm8k_cot_zeroshot=200,ifeval=200")
    ap.add_argument("--max-gen-toks", type=int, default=640)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--path", default=None)
    A = ap.parse_args()
    M = MODELS[A.model]
    if A.path:
        M = dict(M, path=A.path)

    vllm_glue.install()
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    kw = {}
    if A.max_num_seqs:
        kw["max_num_seqs"] = A.max_num_seqs
    if M["arch"] == "gemma4":
        # transformers 5.15 marks head_dim per-layer on gemma4; vLLM reads it globally.
        # gemma4-26B is homogeneous in practice -- verified by the free-arm score check.
        kw["hf_overrides"] = {"allow_global_per_layer_attribute_access": True}
    lm = VLLM(pretrained=M["path"], batch_size="auto", max_gen_toks=A.max_gen_toks,
              max_model_len=A.max_model_len, gpu_memory_utilization=A.gpu_mem,
              enforce_eager=True, enable_prefix_caching=False, dtype="bfloat16", **kw)

    out = os.path.join(ABLATIONS, "instruct_genbench_vllm.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Generative benchmarks under decode-time residency, vLLM stack: '
                 'continuous batching, chat template, greedy, prefill free (observe), '
                 'stateful rule on generated tokens (vllm_residency walker, parity-tested), '
                 'gate-mass preserve correction for norm_topk_prob=False models, prefix '
                 'caching off, all arms same engine. Same deterministic first-N items '
                 'across arms. Producer: analysis/ple/instruct_genbench_vllm.py"\n')
        w.writerow(["model", "E", "k", "arm", "R", "task", "metric", "value",
                    "limit", "max_gen_toks", "secs"])

    suite = []
    for spec in A.tasks.split(","):
        t, _, lim = spec.partition("=")
        suite.append((t, int(lim or 0) or None))

    for arm_name in A.arms.split(","):
        R = None if arm_name == "free" else int(arm_name.lstrip("R"))
        if R is not None:
            assert R >= M["k"], f"R={R} below top-k={M['k']}"
        for task, lim in suite:
            DEC.update(on=R is not None, R=R or 0, swaps=1)
            DEC["state"].clear()
            t0 = time.time()
            try:
                res = simple_evaluate(model=lm, tasks=[task], limit=lim,
                                      apply_chat_template=True,
                                      gen_kwargs="do_sample=False",
                                      confirm_run_unsafe_code=True, log_samples=True)
            except Exception as e:
                print(f"  [{A.model}] {arm_name} {task}: FAILED {type(e).__name__}: {e}",
                      flush=True)
                continue
            secs = time.time() - t0
            metrics = (res.get("groups") or res["results"]).get(task) or res["results"][task]
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)) and "_stderr" not in mk:
                    w.writerow([A.model, M["E"], M["k"], arm_name, R or "", task,
                                mk, f"{mv:.6f}", lim or "full", A.max_gen_toks,
                                f"{secs:.0f}"])
            fh.flush()
            samp = (res.get("samples") or {}).get(task)
            if samp:
                import json
                sd = os.path.join(ABLATIONS, "genbench_samples")
                os.makedirs(sd, exist_ok=True)
                slim = [{"doc_id": x.get("doc_id"),
                         **{mk: x[mk] for mk in x
                            if mk in ("exact_match", "pass@1", "prompt_level_strict_acc",
                                      "inst_level_strict_acc", "acc")}}
                        for x in samp]
                json.dump(slim, open(os.path.join(
                    sd, f"{A.model}_{arm_name}_{task}.json"), "w"))
            show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
            print(f"  [{A.model}] {arm_name} {task}: {show} ({secs:.0f}s)", flush=True)
    fh.close()
    print(f"GENBENCH-VLLM {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
