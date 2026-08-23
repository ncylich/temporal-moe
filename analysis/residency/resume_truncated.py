#!/usr/bin/env python3
"""Extend a cell's budget WITHOUT recomputing what is already done.

Given a prior dump that carries engine token IDs, this:
  * REUSES every generation that finished under the old budget, verbatim, at zero
    GPU cost -- they would end identically at a larger budget, and re-drawing only
    the truncated ones while keeping these would be a resampling bias;
  * CONTINUES every truncated generation from its exact token prefix, generating
    only the additional budget rather than the whole trajectory again.

Cost is therefore (new_cap - old_cap) tokens over the truncated items, instead of
new_cap tokens over every item.

Two things make the continuation faithful, and both are load-bearing:

  1. The prefix should be the ENGINE's token IDs. The same text has many valid
     token sequences (' answer' == ' '+'answer'), so a text-rebuilt prefix is
     A valid tokenization of what the model wrote, but not necessarily THE one it
     emitted. Dumps written before token IDs were persisted therefore resume in
     BEST-EFFORT mode (--allow-retokenized-prefix): the continuation is conditioned
     on byte-identical text, and every item records which prefix it used in
     `prefix_source`. That is a far better use of 3000 already-generated tokens
     than throwing them away, but it is an approximation and is labelled as one.
  2. The residency state is rebuilt by re-walking the generated prefix under the
     rule, since the resident set is path-dependent. The original prompt stays
     free (protocol); the previously-generated tokens carry the rule because that
     is how they were produced. DEC["resume_map"] carries that boundary to the
     walker, which splits the prefill accordingly (partial residency prefill).

    resume_truncated.py --dump gemma4_think_on_R8_humaneval_gemma_fixed \\
        --path /dev/shm/gemma4-26b-it --new-cap 8192 --record-as gemma4_cap8k
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402

SAMP = os.path.join(ABLATIONS, "genbench_samples")


def _rebuild_prompts(A, tk):
    """Reconstruct prompt_ids for dumps that predate prompt-id capture.

    Unlike the generated tail, the prompt IS exactly recoverable: it is the dataset
    item through the model's chat template, and templates tokenize deterministically
    (verified per family in tests/test_tokenizer_families.py). --think must match
    the original run or the rebuilt prefix is simply the wrong prompt."""
    from datasets import load_dataset
    probs = list(load_dataset("openai/openai_humaneval", split="test"))
    instr = ("Complete the following Python function. Provide the complete function "
             "in a single ```python code block.\n\n")
    ck = {} if A.think == "default" else {"enable_thinking": A.think == "on"}
    out = {}
    for p in probs:
        text = tk.apply_chat_template(
            [{"role": "user", "content": instr + p["prompt"]}],
            tokenize=False, add_generation_prompt=True, **ck)
        out[p["task_id"]] = tk(text, add_special_tokens=False).input_ids
    print(f"[resume] rebuilt {len(out)} prompts via the chat template "
          f"(think={A.think})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="dump stem under genbench_samples, e.g. "
                         "gemma4_think_on_R8_humaneval_gemma_fixed")
    ap.add_argument("--path", required=True, help="model directory")
    ap.add_argument("--new-cap", type=int, required=True)
    ap.add_argument("--record-as", required=True, help="stem for the merged dump")
    ap.add_argument("--arm", default=None, help="free or R<n>; default: parsed from --dump")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--allow-retokenized-prefix", action="store_true",
                    help="resume items that lack engine token IDs by re-tokenizing "
                         "their saved text. Byte-identical text, possibly different "
                         "segmentation; each item is labelled prefix_source=retokenized")
    ap.add_argument("--rebuild-prompt", default=None,
                    choices=("humaneval_gemma", "humaneval_gptoss", "humaneval_think"),
                    help="reconstruct prompt_ids for dumps that predate prompt-id "
                         "capture. The prompt IS exactly recoverable: it is the "
                         "dataset item through the model's chat template")
    ap.add_argument("--think", choices=("on", "off", "default"), default="default",
                    help="template thinking flag used by the ORIGINAL run "
                         "(must match, or the rebuilt prompt is the wrong prefix)")
    A = ap.parse_args()

    import genprotocol
    genprotocol.check_dump_dir()
    src = os.path.join(SAMP, A.dump + ".json")
    blob = json.load(open(src))
    items = blob["items"]
    old_cap = max(i["gen_toks"] for i in items)

    missing = [i for i in items if not i.get("gen_ids")]
    if missing and not A.allow_retokenized_prefix:
        sys.exit(f"REFUSING: {len(missing)}/{len(items)} items carry no engine token "
                 f"IDs. Pass --allow-retokenized-prefix to resume them from their "
                 f"saved text instead (byte-identical text, possibly different "
                 f"segmentation; see test_tokenizer_families.py), or re-run this "
                 f"cell once on the current harness, which persists IDs.")

    trunc = [i for i in items if i["gen_toks"] >= old_cap - 8]
    done = [i for i in items if i["gen_toks"] < old_cap - 8]
    add = A.new_cap - old_cap
    assert add > 0, f"--new-cap {A.new_cap} must exceed the dump's cap {old_cap}"
    print(f"[resume] {A.dump}: {len(items)} items, {len(done)} finished (reused "
          f"as-is), {len(trunc)} truncated -> continuing each by up to {add} tokens",
          flush=True)
    print(f"[resume] tokens to decode: {len(trunc) * add} "
          f"(a full re-run would decode ~{len(items) * A.new_cap})", flush=True)
    if not trunc:
        print("[resume] nothing truncated; nothing to do")
        return

    arm = A.arm
    if arm is None:
        for part in A.dump.split("_"):
            if part == "free" or (part.startswith("R") and part[1:].isdigit()):
                arm = part
    assert arm, "could not parse arm from --dump; pass --arm"
    R = None if arm == "free" else int(arm.lstrip("R"))

    vllm_glue.install()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    mml = A.max_model_len or (A.new_cap + 2048)
    llm = LLM(model=A.path, enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
              max_model_len=mml, enable_prefix_caching=False)

    # Build the resume prompts: original prompt + everything generated so far. The
    # boundary tells the walker which part stays free.
    tk = llm.get_tokenizer()
    rebuilt = _rebuild_prompts(A, tk) if A.rebuild_prompt else None

    prompts, meta, src_count = [], [], {"engine_ids": 0, "retokenized": 0}
    DEC["resume_map"].clear()
    DEC["enforce_from"].clear()
    for it in trunc:
        if it.get("gen_ids"):
            gids, src = list(it["gen_ids"]), "engine_ids"
        else:
            # best effort: a valid tokenization of exactly the text the model wrote
            gids = tk(it["raw"], add_special_tokens=False).input_ids
            src = "retokenized"
        src_count[src] += 1
        pids = list(it.get("prompt_ids") or [])
        if not pids and rebuilt is not None:
            pids = rebuilt.get(it["doc"])
        assert pids, ("no prompt_ids in the dump and none rebuilt; pass "
                      "--rebuild-prompt (with the original --think) so the "
                      "free/enforced boundary can be placed")
        it["_prefix_source"] = src
        ids = pids + gids
        key = (len(ids), hash(tuple(ids[:16])), hash(tuple(ids[-16:])))
        DEC["resume_map"][key] = len(pids)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        meta.append(it)

    gc = {}
    try:
        gc = json.load(open(os.path.join(A.path, "generation_config.json")))
    except (FileNotFoundError, ValueError):
        pass
    has = any(k in gc for k in ("temperature", "top_p", "top_k"))
    dt, dp = (1.0, 1.0) if has else (0.7, 0.95)
    sp = SamplingParams(temperature=gc.get("temperature", dt),
                        top_p=gc.get("top_p", dp), top_k=gc.get("top_k") or -1,
                        seed=1234, max_tokens=add, skip_special_tokens=False)

    DEC.update(on=R is not None, R=R or 0, swaps=1)
    DEC["state"].clear()
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    secs = time.time() - t0

    merged = list(done)
    still = 0
    for it, o in zip(meta, outs):
        tail = o.outputs[0]
        new_ids = list(it["gen_ids"]) + list(tail.token_ids)
        merged.append({**{k: v for k, v in it.items() if k != "_prefix_source"},
                       "raw": it["raw"] + tail.text,
                       "gen_ids": new_ids,
                       "gen_toks": len(new_ids),
                       "resumed_from": it["gen_toks"],
                       "prefix_source": it["_prefix_source"]})
        still += len(new_ids) >= A.new_cap - 8
    order = {i["doc"]: n for n, i in enumerate(items)}
    merged.sort(key=lambda x: order.get(x["doc"], 1 << 30))
    # task name = the dump stem after "<record>_<arm>_", so the merged dump lands
    # under the same task the original cell used
    task = A.dump.split(f"_{arm}_", 1)[1] if f"_{arm}_" in A.dump else "resumed"
    genprotocol.write_dump(A.record_as, arm, task, merged, len(items))
    print(f"[resume] prefixes: {src_count['engine_ids']} exact (engine ids), "
          f"{src_count['retokenized']} best-effort (re-tokenized text)", flush=True)
    print(f"[resume] continued {len(trunc)} items in {secs:.0f}s; "
          f"{still} still at the new cap ({100*still/len(items):.1f}% of the cell)",
          flush=True)


if __name__ == "__main__":
    main()
