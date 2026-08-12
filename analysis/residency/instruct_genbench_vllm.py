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
    ap.add_argument("--backoff-cap", type=int, default=2048,
                    help="hard generation cap; thinking/high-effort arms need 4096")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--record-as", default=None,
                    help="model column for the CSV rows (adapted/control variants)")
    ap.add_argument("--think", choices=("default", "on", "off"), default="default",
                    help="chat-template thinking toggle (enable_thinking kwarg)")
    ap.add_argument("--think-prefill", default=None,
                    help="string appended after the chat template for --think off on "
                         "models whose template has no toggle (empty think block)")
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
    lm = VLLM(pretrained=M["path"], batch_size="auto", max_gen_toks=A.max_gen_toks,
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

        def _act(*aa, **kk):
            out = _orig_act(*aa, **{**kk, **_extra})
            if A.think == "off" and A.think_prefill and isinstance(out, str):
                out = out + A.think_prefill
            return out

        _tk.apply_chat_template = _act
        print(f"[genbench] thinking={A.think} effort={A.reasoning_effort} "
              f"prefill={A.think_prefill!r}", flush=True)

    import genbackoff
    genbackoff.install(lm, A.max_gen_toks, cap=A.backoff_cap)

    # Sampling per the model's own generation_config (greedy is NEVER used: thinking
    # models degenerate into repetition loops under it, which both corrupts answers and
    # burns the token budget). Fallbacks are the HF semantic defaults (temp 1.0 = the
    # model's deployment behavior when it ships no recommendation). Seeded per request
    # for reproducibility; the backoff retry reuses the seed so the resampled prefix
    # stays stable.
    import json as _json
    try:
        _gc = _json.load(open(os.path.join(M["path"], "generation_config.json")))
    except (FileNotFoundError, ValueError):
        _gc = {}
    _t = A.temperature if A.temperature is not None else _gc.get("temperature", 1.0)
    _p = A.top_p if A.top_p is not None else _gc.get("top_p", 1.0)
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
        # harmony format: keep special tokens so the analysis/final channel structure
        # survives detokenization, then score the FINAL channel only (the analysis
        # channel is free-form reasoning and would fail e.g. IFEval's format rules).
        # A response truncated before its final channel scores as empty -- a real
        # outcome, applied identically to every arm.
        gen_kwargs += ",skip_special_tokens=False"
        orig_gu = lm.generate_until
        goss_think = []          # per-response analysis-channel token counts (this cell)

        def _final_channel(text):
            if "<|channel|>final<|message|>" in text:
                pre, text = text.rsplit("<|channel|>final<|message|>", 1)
                goss_think.append(len(lm.tokenizer(pre, add_special_tokens=False)
                                      .input_ids))
            elif "<|channel|>" in text:
                goss_think.append(len(lm.tokenizer(text, add_special_tokens=False)
                                      .input_ids))
                return ""        # truncated inside the analysis channel
            else:
                goss_think.append(0)
            return text.split("<|return|>")[0].split("<|end|>")[0]

        lm.generate_until = lambda reqs, **k2: [_final_channel(t)
                                                for t in orig_gu(reqs, **k2)]

    if M["arch"] in ("gemma4", "lfm", "qwen3_5"):
        # score the ANSWER segment only, exactly as the gpt-oss final-channel filter
        # does: judging thinking text against task formats punishes the thinking mode
        # artifactually (gemma think-on IFEval collapsed to 0.25 before this). No-op
        # when no think segment appears. Think lengths captured here since the scored
        # text loses them.
        orig_gu_g = lm.generate_until
        gemma_think = []

        def _answer_channel(text):
            if M["arch"] == "gemma4":
                if "<|channel>" not in text:
                    gemma_think.append(0)
                    return text
                if "<channel|>" in text:
                    pre, ans = text.rsplit("<channel|>", 1)
                else:
                    pre, ans = text, ""          # truncated inside the thought channel
            else:                                # lfm / qwen: </think> closes thinking
                if "</think>" in text:
                    pre, ans = text.split("</think>", 1)
                elif "<think>" in text or M["arch"] == "qwen3_5" and A.think != "off":
                    pre, ans = text, ""          # opened (or prompt-opened), never closed
                else:
                    gemma_think.append(0)
                    return text
            gemma_think.append(len(lm.tokenizer(pre, add_special_tokens=False)
                                   .input_ids))
            return ans

        lm.generate_until = lambda reqs, **k2: [_answer_channel(t)
                                                for t in orig_gu_g(reqs, **k2)]

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
                                  confirm_run_unsafe_code=True, log_samples=True)
            secs = time.time() - t0
            metrics = (res.get("groups") or res["results"]).get(task) or res["results"][task]
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)) and "_stderr" not in mk:
                    w.writerow([A.record_as or A.model, M["E"], M["k"], arm_name, R or "", task,
                                mk, f"{mv:.6f}", lim or "full", A.max_gen_toks,
                                f"{secs:.0f}"])
            fh.flush()
            samp = (res.get("samples") or {}).get(task)
            if samp:
                import json
                import re as _re
                sd = os.path.join(ABLATIONS, "genbench_samples")
                os.makedirs(sd, exist_ok=True)
                THINK_RE = {"lfm": r"<think>.*?</think>", "qwen3_5": r"<think>.*?</think>",
                            "gemma4": r"<\|channel>.*?<channel\|>"}.get(M["arch"])
                OPEN_TAG = {"lfm": "<think>", "qwen3_5": "<think>",
                            "gemma4": "<|channel>"}.get(M["arch"])

                def _lens(x):
                    resp = (x.get("resps") or [[""]])[0]
                    resp = resp[0] if resp else ""
                    d = {"gen_toks": len(lm.tokenizer(resp,
                                                      add_special_tokens=False).input_ids)}
                    if THINK_RE:
                        spans = "".join(_re.findall(THINK_RE, resp, _re.S))
                        if not spans and OPEN_TAG and OPEN_TAG in resp:
                            spans = resp[resp.index(OPEN_TAG):]   # truncated mid-think
                        if not spans and OPEN_TAG == "<think>" and "</think>" in resp:
                            # template pre-opens the think block in the prompt (qwen):
                            # thinking = response start through the first closing tag
                            spans = resp[: resp.index("</think>")]
                        d["think_toks"] = len(lm.tokenizer(
                            spans, add_special_tokens=False).input_ids) if spans else 0
                        # backtracking markers separate "uniform dilution" (slower
                        # progress per token) from "error reaction" (correction bursts)
                        d["backtracks"] = len(_re.findall(
                            r"\b([Ww]ait|[Aa]ctually|[Hh]mm|[Ll]et me re|[Dd]ouble.check)",
                            spans))
                    return d

                slim = [{"doc_id": x.get("doc_id"), **_lens(x),
                         **{mk: x[mk] for mk in x
                            if mk in ("exact_match", "pass@1", "prompt_level_strict_acc",
                                      "inst_level_strict_acc", "acc")}}
                        for x in samp]
                blob = {"items": slim}
                if M["arch"] == "gptoss":
                    # per-item resps are post-filter (final channel only); analysis-
                    # channel lengths captured in generation order, unaligned to doc_id
                    blob["analysis_toks"] = list(goss_think) if "goss_think" in dir() else []
                if M["arch"] == "gemma4":
                    blob["analysis_toks"] = list(gemma_think)
                json.dump(blob, open(os.path.join(
                    sd, f"{A.record_as or A.model}_{arm_name}_{task}.json"), "w"))
                # full response token ids (re-tokenized; INSTRUCT_ANALYSIS_PLAN.md) --
                # workspace only, never committed
                import torch as _torch
                td = "/workspace/instruct-traj/genbench_tokens"
                os.makedirs(td, exist_ok=True)
                _torch.save({"items": [
                    {"doc_id": x.get("doc_id"),
                     "ids": lm.tokenizer((x.get("resps") or [[""]])[0][0]
                                         if (x.get("resps") or [[""]])[0] else "",
                                         add_special_tokens=False).input_ids}
                    for x in samp]},
                    os.path.join(td, f"{A.record_as or A.model}_{arm_name}_{task}.pt"))
            if M["arch"] == "gptoss":
                goss_think.clear()
            if M["arch"] == "gemma4":
                gemma_think.clear()
            show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
            print(f"  [{A.model}] {arm_name} {task}: {show} ({secs:.0f}s)", flush=True)
    fh.close()
    print(f"GENBENCH-VLLM {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
