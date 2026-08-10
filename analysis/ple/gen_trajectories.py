#!/usr/bin/env python3
"""Generate the frozen free-model trajectories for the instruct-residency program.

The FREE model (no residency rule anywhere -- these are the baseline reference) greedily
decodes each frozen WildChat prompt through its own chat template, batch=1, fixed cap.
Every later arm scores against these exact token sequences, so this is paid once per model.

Output {out}/{tag}.pt: list of {idx, prompt_len, ids} (full sequence incl. prompt) plus a
meta dict (model path, prompt-set sha, generation config, transformers version). Saved
incrementally every 25 prompts; rerunning resumes from what is already saved.

    gen_trajectories.py --model /workspace/instruct-models/olmoe-0125-instruct --tag olmoe_instruct
    gen_trajectories.py --model ... --tag ... --limit 3        # smoke: 3 prompts, asserts
"""
import argparse
import hashlib
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="/workspace/olmoe-adapt/data/wildchat_prompts_500.jsonl")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--max-prompt-tok", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only this many prompts")
    ap.add_argument("--out", default="/workspace/instruct-traj")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    path = os.path.join(A.out, f"{A.tag}.pt")

    prompts = [json.loads(l) for l in open(A.prompts)]
    sha = hashlib.sha256(open(A.prompts, "rb").read()).hexdigest()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16).to("cuda")
    model.eval()

    done, rows = {}, []
    if os.path.exists(path) and A.limit is None:
        prev = torch.load(path, weights_only=False)
        rows = prev["rows"]
        done = {r["idx"] for r in rows}
        print(f"[gen] resuming: {len(rows)} trajectories already saved", flush=True)

    def save():
        torch.save({"rows": rows, "meta": {
            "model": A.model, "prompt_set": A.prompts, "prompt_sha256": sha,
            "max_new_tokens": A.max_new, "decoding": "greedy batch=1",
            "transformers": __import__("transformers").__version__}}, path)

    t0, gen_tok, skipped = time.time(), 0, 0
    todo = prompts[: A.limit] if A.limit else prompts
    for p in todo:
        if p["idx"] in done:
            continue
        ids = tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                      add_generation_prompt=True, return_tensors="pt")
        if ids.shape[1] > A.max_prompt_tok:
            skipped += 1
            continue
        with torch.no_grad():
            out = model.generate(ids.to("cuda"), do_sample=False, max_new_tokens=A.max_new,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.eos_token_id)
        n_new = out.shape[1] - ids.shape[1]
        assert n_new > 0, f"empty generation for prompt {p['idx']}"
        gen_tok += n_new
        rows.append({"idx": p["idx"], "prompt_len": int(ids.shape[1]),
                     "ids": out[0].cpu().to(torch.int32)})
        if A.limit:
            print(f"  [smoke {p['idx']}] prompt {ids.shape[1]} tok -> +{n_new} tok: "
                  f"{tok.decode(out[0, ids.shape[1]:ids.shape[1]+40])!r}", flush=True)
        if len(rows) % 25 == 0:
            save()
            el = time.time() - t0
            print(f"  [gen] {len(rows)}/{len(todo)} ({gen_tok/el:.0f} tok/s)", flush=True)
    if A.limit is None:
        save()
        resp = [len(r["ids"]) - r["prompt_len"] for r in rows]
        print(f"[gen] DONE {A.tag}: {len(rows)} trajectories, {skipped} skipped (long prompt), "
              f"{sum(resp)} response tokens (mean {sum(resp)/max(len(resp),1):.0f})", flush=True)
    else:
        print(f"[gen] SMOKE OK {A.tag}", flush=True)


if __name__ == "__main__":
    main()
