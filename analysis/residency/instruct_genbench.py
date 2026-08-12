#!/usr/bin/env python3
"""Generative downstream eval of instruct models under decode-time residency.

The serving-realistic measurement: batch=1, chat template, greedy; prefill runs FREE and
the rolling rule (stateful across generate() forwards, decode_state.py, parity-tested
against the batch scan) is enforced on generated tokens only. Arms share one model load
and the same deterministic first-N benchmark items.

    instruct_genbench.py --model olmoe_instruct --arms free,R8 --tasks gsm8k_cot_zeroshot,ifeval --limit 250
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
import decode_state as DS                                            # noqa: E402
from decode_state import DEC                                         # noqa: E402
from instruct_selfce import MODELS, load                             # noqa: E402


def set_decode_arm(M, R):
    """Configure the constraint for generation. R=None -> fully free."""
    on = R is not None
    DEC.update(on=on, R=R or 0, swaps=1)
    DS.reset()
    if M["arch"] in ("olmoe", "qwen3_5"):
        import residency as RES
        RES._CFG.update(on=on, decode_mode=on, R=R or 0, evict="min_logit",
                        gate_mass="preserve", swaps=1, R_map=None, collect_telem=False,
                        enforce_from=0, free_set=None)
    else:
        import granularity_ladder as GL
        GL.CFG.update(on=on, decode_mode=on, R=R or 0, free_set=None, R_map=None,
                      enforce_from=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--arms", required=True, help="comma list: free,R8,R16,...")
    ap.add_argument("--tasks", default="gsm8k_cot_zeroshot=200,ifeval=200",
                    help="task=limit pairs; limit 0 = full set (per-subtask for groups)")
    ap.add_argument("--max-gen-toks", type=int, default=640)
    ap.add_argument("--path", default=None, help="checkpoint dir override (shm staging)")
    A = ap.parse_args()
    M = MODELS[A.model]
    if A.path:
        M = dict(M, path=A.path)
    assert os.path.isdir(M["path"]), f"checkpoint dir missing: {M['path']}"

    model, tok, _ = load(A.model, M)
    if M["arch"] == "gemma4":
        import granularity_ladder as GL
        GL.tag_gemma4(model)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
    # this lm_eval's HFLM hardcodes max_gen_toks as a property returning 256 and silently
    # swallows the constructor kwarg (defect: every earlier HF-twin generative row ran at
    # 256 tokens; rows before 2026-08-12 carry the intended, NOT the applied, budget)
    type(lm).max_gen_toks = property(lambda self: A.max_gen_toks)
    assert lm.max_gen_toks == A.max_gen_toks
    import genbackoff
    genbackoff.install(lm, A.max_gen_toks)

    out = os.path.join(ABLATIONS, "instruct_genbench.csv")
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Generative benchmarks under decode-time residency: batch=1, chat '
                 'template, greedy, prefill free, stateful rule on generated tokens only '
                 '(decode_state.py, parity-tested vs the batch scan). Same deterministic '
                 'first-N items across arms. Producer: analysis/residency/instruct_genbench.py"\n')
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
            set_decode_arm(M, R)
            t0 = time.time()
            # FAIL FAST (see instruct_genbench_vllm.py): swallowed exceptions obfuscate
            # bugs and let a corrupted engine poison later cells.
            res = simple_evaluate(model=lm, tasks=[task], limit=lim,
                                  apply_chat_template=True,
                                  gen_kwargs="do_sample=True,temperature=1.0,top_p=1.0",
                                  confirm_run_unsafe_code=True, log_samples=True)
            secs = time.time() - t0
            # group tasks (e.g. mmlu_flan_cot_fewshot): report the aggregate, not 57 subtasks
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
    print(f"GENBENCH {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
