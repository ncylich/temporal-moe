#!/usr/bin/env python3
"""Generative benchmarks under decode-time residency on the vLLM stack.

Batch-fair protocol (ALL arms share the identical engine, continuous batching, chat
template, card-recipe sampling -- NEVER greedy, prefill free, rule on generated tokens) at
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

    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--gen-cap", type=int, default=2048,
                    help="per-request generation budget (single pass; also the hard "
                         "cap -- responses finishing here are degeneracy-flagged). "
                         "Thinking arms 4096; ifeval-thinking 8192")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--record-as", default=None,
                    help="model column for the CSV rows (adapted/control variants)")
    ap.add_argument("--samples-json", default=None,
                    help="path to {task: [doc_ids]} for SCREENING runs: evaluate only "
                         "these docs (lm_eval samples=). Screening rows are NOT "
                         "protocol-comparable to full-task rows -- pair with "
                         "--csv-name so they never enter the authoritative CSV")
    ap.add_argument("--csv-name", default="instruct_genbench_vllm.csv",
                    help="output CSV under results/ablations (screening runs use "
                         "screening_genbench.csv)")
    ap.add_argument("--think", choices=("default", "on", "off"), default="default",
                    help="chat-template thinking toggle (enable_thinking kwarg)")

    ap.add_argument("--reasoning-effort", default=None,
                    choices=("low", "medium", "high"), help="gpt-oss harmony effort")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override shipped config (mode-specific recipes, e.g. qwen "
                         "non-thinking: 0.7/0.8)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="fallback when the shipped generation_config omits it but the "
                         "model card requires it (qwen3.5 thinking: 1.5)")
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
    if M["arch"] == "gptoss":
        kw["dtype"] = "auto"                     # MXFP4 checkpoint: keep native quant
    # lm_eval's native reasoning support for every think-in-text stack: task
    # stop-strings stay away from vLLM (they fire inside think blocks -- humaneval's
    # "\ndef", gsm8k's "Q:") and thinking is stripped before stops and scoring.
    # Replaces all hand-rolled answer filters. Marker absent => whole text scored.
    THINK_MARK = {"qwen3_5": "</think>", "lfm": "</think>",
                  "gemma4": "<channel|>",
                  "gptoss": "<|channel|>final<|message|>"}.get(M["arch"])
    if M["arch"] == "qwen3_5" and A.think == "off":
        THINK_MARK = None            # instruct mode: no think segment to strip
    lm = VLLM(pretrained=M["path"], batch_size="auto", max_gen_toks=A.gen_cap,
              max_model_len=A.max_model_len, gpu_memory_utilization=A.gpu_mem,
              enforce_eager=True, enable_prefix_caching=False,
              **({"dtype": "bfloat16"} | kw))

    # Thinking-mode control: inject template kwargs at the tokenizer level (unknown
    # jinja context vars are inert, so enable_thinking is safe to pass everywhere);
    # --think-prefill closes an empty think block for models with no template toggle.
    if A.think != "default" or A.reasoning_effort:
        _tk = lm.tokenizer
        _orig_act = _tk.apply_chat_template
        _extra = {}
        if A.think != "default":
            _extra["enable_thinking"] = A.think == "on"
        if A.reasoning_effort:
            _extra["reasoning_effort"] = A.reasoning_effort

        _tk.apply_chat_template = lambda *aa, **kk: _orig_act(
            *aa, **{**kk, **_extra})
        print(f"[genbench] thinking={A.think} effort={A.reasoning_effort}", flush=True)

    import genprotocol
    genprotocol.install(lm, cap=A.gen_cap, think_marker=THINK_MARK)

    # Sampling per the model's own generation_config (greedy is NEVER used: thinking
    # models degenerate into repetition loops under it, which both corrupts answers and
    # burns the token budget). Fallbacks are the HF semantic defaults (temp 1.0 = the
    # model's deployment behavior when it ships no recommendation). Seeded per request
    # for reproducibility.

    import json as _json
    try:
        _gc = _json.load(open(os.path.join(M["path"], "generation_config.json")))
    except (FileNotFoundError, ValueError):
        _gc = {}
    # No-recipe fallback: community-standard 0.7/0.95, NOT HF's ancestral 1.0/1.0
    # (ancestral sampling depressed OLMoE ~5-30 pts across tasks vs its card, worst
    # on code). Models shipping a recipe are unaffected.
    _has_recipe = any(k in _gc for k in ("temperature", "top_p", "top_k"))
    _dt, _dp = (1.0, 1.0) if _has_recipe else (0.7, 0.95)
    _t = A.temperature if A.temperature is not None else _gc.get("temperature", _dt)
    _p = A.top_p if A.top_p is not None else _gc.get("top_p", _dp)
    _k = _gc.get("top_k") or -1
    gen_kwargs = f"do_sample=True,temperature={_t},top_p={_p},top_k={_k},seed=1234"
    # carry the FULL shipped recipe: qwen3.5's thinking mode needs presence_penalty=1.5
    # (its rambling guard) -- omitting it produced non-terminating planning monologues
    _pp = _gc.get("presence_penalty", A.presence_penalty)
    _mp = _gc.get("min_p")
    if _pp:
        gen_kwargs += f",presence_penalty={_pp}"
    if _mp:
        gen_kwargs += f",min_p={_mp}"
    print(f"[genbench] sampling: temp={_t} top_p={_p} top_k={_k} pp={_pp} mp={_mp} "
          f"(model config)", flush=True)
    if M["arch"] == "gptoss":
        gen_kwargs += ",skip_special_tokens=False"   # channel markers must survive

    assert not (A.samples_json and A.csv_name == "instruct_genbench_vllm.csv"), \
        "doc-subset screening rows must not enter the authoritative CSV"
    sub = None
    if A.samples_json:
        import json as _sj
        sub = {k: list(map(int, v)) for k, v in
               _sj.load(open(A.samples_json)).items()}
        print(f"[genbench] SCREENING subsets: "
              f"{ {k: len(v) for k, v in sub.items()} }", flush=True)
    out = os.path.join(ABLATIONS, A.csv_name)
    exists = os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not exists:
        fh.write('"# Generative benchmarks under decode-time residency, vLLM stack: '
                 'continuous batching, chat template, card-recipe sampling (never '
                 'greedy), single pass at max_gen_toks, prefill free (observe), '
                 'stateful rule on generated tokens (vllm_residency walker, parity-tested), '
                 'gate-mass preserve correction for norm_topk_prob=False models, prefix '
                 'caching off, all arms same engine. Same deterministic first-N items '
                 'across arms. Producer: analysis/residency/instruct_genbench_vllm.py"\n')
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
            # FAIL FAST, no per-task isolation: this is a results pipeline, not a
            # service. A swallowed exception obfuscates a bug that needs fixing and
            # lets a poisoned engine corrupt every later cell (2026-08-12: one walker
            # crash silently took out five cells under the old try/except-continue).
            res = simple_evaluate(model=lm, tasks=[task], limit=lim,
                                  apply_chat_template=True,
                                  gen_kwargs=gen_kwargs,
                                  confirm_run_unsafe_code=True, log_samples=True,
                                  **({"samples": {task: sub[task]}}
                                     if sub and task in sub else {}))
            secs = time.time() - t0
            metrics = (res.get("groups") or res["results"]).get(task) or res["results"][task]
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)) and "_stderr" not in mk:
                    w.writerow([A.record_as or A.model, M["E"], M["k"], arm_name, R or "", task,
                                mk, f"{mv:.6f}", lim or "full", A.gen_cap,
                                f"{secs:.0f}"])
            fh.flush()
            samp = (res.get("samples") or {}).get(task)
            if samp:
                import json
                sd = os.path.join(ABLATIONS, "genbench_samples")
                os.makedirs(sd, exist_ok=True)
                def _lens(x):
                    # x["resps"] is the POST-STRIP scoring text: valid for gen_toks
                    # of the scored answer only. Think lengths come exclusively from
                    # the raw doc-keyed capture (think_toks_by_doc below); a per-item
                    # think_toks measured here was a defect (0 for closed blocks,
                    # full length for cap-truncated ones) and is deliberately gone.
                    resp = (x.get("resps") or [[""]])[0]
                    resp = resp[0] if resp else ""
                    return {"gen_toks": len(lm.tokenizer(
                        resp, add_special_tokens=False).input_ids)}

                slim = [{"doc_id": x.get("doc_id"), **_lens(x),
                         **{mk: x[mk] for mk in x
                            if mk in ("exact_match", "pass@1", "prompt_level_strict_acc",
                                      "inst_level_strict_acc", "acc")}}
                        for x in samp]
                blob = {"items": slim}
                if THINK_MARK and genprotocol.FINALS:
                    # doc-aligned, one entry per item (continuation era)
                    blob["think_toks_by_doc"] = {
                        str(d): len(lm.tokenizer(
                            t.rsplit(THINK_MARK, 1)[0] if THINK_MARK in t else t,
                            add_special_tokens=False).input_ids)
                        for d, t in genprotocol.FINALS.items()}
                json.dump(blob, open(os.path.join(
                    sd, f"{A.record_as or A.model}_{arm_name}_{task}.json"), "w"))
                # full response token ids (re-tokenized; INSTRUCT_ANALYSIS_PLAN.md) --
                # workspace only, never committed
                import torch as _torch
                td = "/workspace/instruct-traj/genbench_tokens"
                os.makedirs(td, exist_ok=True)
                def _raw_or_resp(x):
                    if x.get("doc_id") in genprotocol.FINALS:
                        return genprotocol.FINALS[x["doc_id"]]
                    r = (x.get("resps") or [[""]])[0]
                    return r[0] if r else ""
                _torch.save({"items": [
                    {"doc_id": x.get("doc_id"),
                     "ids": lm.tokenizer(_raw_or_resp(x),
                                         add_special_tokens=False).input_ids}
                    for x in samp]},
                    os.path.join(td, f"{A.record_as or A.model}_{arm_name}_{task}.pt"))
            show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
            print(f"  [{A.model}] {arm_name} {task}: {show} ({secs:.0f}s)", flush=True)
    fh.close()
    print(f"GENBENCH-VLLM {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
