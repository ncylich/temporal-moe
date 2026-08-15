#!/usr/bin/env python3
"""Bulk free-model trajectory generation on vLLM (training data for CE adaptation).

FREE model, card-recipe sampling (greedy loops on thinking models), chat template, continuous batching. No residency machinery is loaded:
these are baseline trajectories (prefix caching is safe and on, since nothing scans).
Output matches gen_trajectories.py's schema: rows of {idx, prompt_len, ids}.

    gen_traj_vllm.py --model /dev/shm/gemma4-26b-it --tag gemma4_train5k \
        --prompts /workspace/olmoe-adapt/data/wildchat_prompts_train5k.jsonl
"""
import argparse
import hashlib
import json
import os

import torch

sys_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--max-prompt-tok", type=int, default=512)
    ap.add_argument("--out", default="/workspace/instruct-traj")
    ap.add_argument("--think", choices=("default", "on", "off"), default="default",
                    help="chat-template thinking toggle for the generated data")
    A = ap.parse_args()

    prompts = [json.loads(l) for l in open(A.prompts)]
    sha = hashlib.sha256(open(A.prompts, "rb").read()).hexdigest()

    import vllm_glue                      # gemma4 per-layer config fixes; residency stays off
    vllm_glue.install()
    from transformers import AutoTokenizer
    _t = AutoTokenizer.from_pretrained(A.model)
    def _plen(p):
        enc = _t.apply_chat_template([{"role": "user", "content": p["text"]}],
                                     add_generation_prompt=True, tokenize=True,
                                     return_dict=True)
        return len(enc["input_ids"])
    kept = [p for p in prompts if _plen(p) <= A.max_prompt_tok]
    print(f"[genv] {len(prompts) - len(kept)} prompts over {A.max_prompt_tok} tokens dropped "
          f"pre-submission", flush=True)
    prompts = kept
    from vllm import LLM, SamplingParams
    llm = LLM(model=A.model, enforce_eager=False, gpu_memory_utilization=0.9,
              max_model_len=A.max_prompt_tok + A.max_new)
    # sampling per the model's own generation_config (greedy loops on thinking models);
    # seeded so the trajectory set is reproducible
    import json as _json
    try:
        _gc = _json.load(open(os.path.join(A.model, "generation_config.json")))
    except (FileNotFoundError, ValueError):
        _gc = {}
    sp = SamplingParams(temperature=_gc.get("temperature", 1.0),
                        top_p=_gc.get("top_p", 1.0), top_k=_gc.get("top_k") or -1,
                        seed=1234, max_tokens=A.max_new)
    print(f"[genv] sampling temp={sp.temperature} top_p={sp.top_p} top_k={sp.top_k}",
          flush=True)
    msgs = [[{"role": "user", "content": p["text"]}] for p in prompts]
    ctk = ({} if A.think == "default"
           else {"chat_template_kwargs": {"enable_thinking": A.think == "on"}})
    outs = llm.chat(msgs, sp, **ctk)

    rows, skipped = [], 0
    for p, o in zip(prompts, outs):
        pids = list(o.prompt_token_ids)
        gids = list(o.outputs[0].token_ids)
        if len(pids) > A.max_prompt_tok or not gids:
            skipped += 1
            continue
        rows.append({"idx": p["idx"], "prompt_len": len(pids),
                     "ids": torch.tensor(pids + gids, dtype=torch.int32)})
    os.makedirs(A.out, exist_ok=True)
    torch.save({"rows": rows, "meta": {
        "model": A.model, "prompt_set": A.prompts, "prompt_sha256": sha,
        "max_new_tokens": A.max_new, "decoding": "greedy vllm continuous-batching",
        "stack": "vllm"}}, os.path.join(A.out, f"{A.tag}.pt"))
    resp = [len(r["ids"]) - r["prompt_len"] for r in rows]
    print(f"[genv] DONE {A.tag}: {len(rows)} trajectories, {skipped} skipped, "
          f"{sum(resp)} response tokens (mean {sum(resp)/max(1,len(resp)):.0f})", flush=True)


if __name__ == "__main__":
    main()
