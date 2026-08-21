#!/usr/bin/env python3
"""Constrained self-generation for distillation: generate responses from an
ADAPTED half-grain checkpoint UNDER the residency rule (the states where its
failures live), and save a trainer-compatible trajectory file
({rows: [{ids, prompt_len}]}). Boot mirrors the audited constrained stack
(vllm_glue + vllm_residency + DEC, enforce_eager; prefill free, rule on
generated tokens).

    selfgen_traj.py --path <merged dir> --R 48 --prompts d7_prompts.jsonl \
        --out /workspace/instruct-traj/gemma4_selfgen_r48.pt
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=4500)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--think", choices=("on", "off"), default=None,
                    help="pass enable_thinking to the chat template (qwen); omit "
                         "for templates without the kwarg (gemma default = low)")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.path)
    llm = LLM(model=A.path, enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
              max_model_len=2048, enable_prefix_caching=False)
    DEC.update(on=True, R=A.R, swaps=1)
    DEC["state"].clear()

    prompts = [json.loads(l) for l in open(A.prompts)][: A.n]
    texts = [p.get("prompt") or p.get("text") for p in prompts]
    msgs = [[{"role": "user", "content": t}] for t in texts]
    sp = SamplingParams(temperature=0.7, top_p=0.8, seed=1234, max_tokens=A.max_new)
    ctk = ({} if A.think is None
           else {"chat_template_kwargs": {"enable_thinking": A.think == "on"}})
    outs = llm.chat(msgs, sp, use_tqdm=True, **ctk)

    rows = []
    trunc = 0
    for t, o in zip(texts, outs):
        enc = tok.apply_chat_template([{"role": "user", "content": t}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_dict=True)
        pids = list(enc["input_ids"])
        gids = list(o.outputs[0].token_ids)
        if len(gids) >= A.max_new:
            trunc += 1
        rows.append({"ids": torch.tensor(pids + gids, dtype=torch.int32),
                     "prompt_len": len(pids)})
    torch.save({"rows": rows}, "/tmp/selfgen_tmp.pt")
    import shutil
    shutil.move("/tmp/selfgen_tmp.pt", A.out)
    print(f"[sg] DONE {len(rows)} rows -> {A.out}; cap-hit {trunc} "
          f"({100*trunc/len(rows):.1f}%); mean gen len "
          f"{sum(len(r['ids'])-r['prompt_len'] for r in rows)/len(rows):.0f}", flush=True)


if __name__ == "__main__":
    main()
