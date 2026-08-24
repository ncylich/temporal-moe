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

import genprotocol                                                   # noqa: E402

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
    if A.reasoning_effort:          # gpt-oss: effort is part of the prompt
        ck["reasoning_effort"] = A.reasoning_effort
    out = {}
    for p in probs:
        text = tk.apply_chat_template(
            [{"role": "user", "content": instr + p["prompt"]}],
            tokenize=False, add_generation_prompt=True, **ck)
        out[p["task_id"]] = tk(text, add_special_tokens=False).input_ids
    print(f"[resume] rebuilt {len(out)} prompts via the chat template "
          f"(think={A.think}, effort={A.reasoning_effort})", flush=True)
    return out


def _score(A, merged, arm, R, secs):
    """Score the merged dump with the originating harness's extractor, so a resumed
    cell is directly comparable to the cell it extends."""
    import csv
    import json as _j
    import re
    import subprocess
    from datasets import load_dataset
    probs = {p["task_id"]: p for p in load_dataset("openai/openai_humaneval",
                                                   split="test")}
    FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
    CHANNEL = re.compile(r"<\|channel>.*?(?:<channel\|>|\Z)", re.S)

    def extract(raw, unfinished):
        if unfinished:
            return ""                     # still unfinished at the NEW cap
        if A.score == "humaneval_gemma":
            t = CHANNEL.sub("", raw)
            b = FENCE.findall(t)
            return b[-1] if b else t
        if A.score == "humaneval_gptoss":
            MK = "<|channel|>final<|message|>"
            t = raw.rsplit(MK, 1)[1] if MK in raw else ("" if "<|channel|>" in raw else raw)
        else:
            t = raw.split("</think>", 1)[1] if "</think>" in raw else raw
        return (FENCE.findall(t) or [t])[-1]

    mk = {"humaneval_gemma": "<channel|>"}.get(A.score, "</think>")
    if A.score == "humaneval_gptoss":
        mk = "<|channel|>final<|message|>"
    preds, tests = [], []
    unfin = []
    for it in merged:
        u = it["gen_toks"] >= A.new_cap - 8 and mk not in it["raw"]
        unfin.append(u)
        preds.append([extract(it["raw"], u)])
        p = probs[it["doc"]]
        tests.append(p["test"] + f"\ncheck({p['entry_point']})")
    _j.dump({"preds": preds, "tests": tests}, open("/tmp/heg_preds.json", "w"))
    out = subprocess.run(["/workspace/venv_fla/bin/python",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "heg_scorer.py")], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("PASS1")]
    assert line, f"scorer failed: {out.stderr[-400:]}"
    p1 = float(line[0].split()[1])
    il = [l for l in out.stdout.splitlines() if l.startswith("ITEMS")]
    bits = il[0].split()[1] if il else ""
    for i, it in enumerate(merged):
        it["pass"] = (bits[i] == "1") if i < len(bits) else None
        it["unfinished"] = unfin[i]
    print(f"[resume] {A.record_as} {arm} pass@1 = {p1:.4f} "
          f"({len(merged)} items, {sum(unfin)} still unfinished)", flush=True)
    with open(os.path.join(ABLATIONS, A.csv_name), "a", newline="") as fh:
        csv.writer(fh).writerow(
            [A.record_as, 128, 8, arm, R or "", A.score.replace("humaneval_gemma",
             "humaneval_gemma_fixed"), "pass@1,channel-aware", f"{p1:.6f}",
             "full", A.new_cap, f"{secs:.0f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="dump stem(s) under genbench_samples, comma-separated to run "
                         "several cells through ONE engine boot (loading a 49GB model "
                         "takes ~9 min, several times the generation it serves here). "
                         "Per-cell --arm/--old-cap are parsed from the stems unless "
                         "given explicitly.")
    ap.add_argument("--path", required=True, help="model directory")
    ap.add_argument("--new-cap", type=int, required=True)
    ap.add_argument("--old-cap", type=int, default=None,
                    help="the budget the dump was generated at. MUST be given when "
                         "the cell may not have truncated: inferring it from "
                         "max(gen_toks) would label the single longest (naturally "
                         "finished) generation as truncated and 'resume' past its EOS")
    ap.add_argument("--record-as", required=True, help="stem for the merged dump")
    ap.add_argument("--arm", default=None, help="free or R<n>; default: parsed from --dump")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise every non-GPU code path (load, truncation split, "
                         "prompt rebuild, merge, score, dump write) with a stub "
                         "engine; writes dumps suffixed _dryrun")
    ap.add_argument("--allow-retokenized-prefix", action="store_true",
                    help="resume items that lack engine token IDs by re-tokenizing "
                         "their saved text. Byte-identical text, possibly different "
                         "segmentation; each item is labelled prefix_source=retokenized")
    ap.add_argument("--rebuild-prompt", default=None,
                    choices=("humaneval_gemma", "humaneval_gptoss", "humaneval_think"),
                    help="reconstruct prompt_ids for dumps that predate prompt-id "
                         "capture. The prompt IS exactly recoverable: it is the "
                         "dataset item through the model's chat template")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=("low", "medium", "high"),
                    help="gpt-oss effort used by the ORIGINAL run; part of the "
                         "prompt, so it must match or the rebuilt prefix is wrong")
    ap.add_argument("--score", default=None,
                    choices=("humaneval_gemma", "humaneval_gptoss", "humaneval_think"),
                    help="score the merged dump with that harness's extractor and "
                         "append a CSV row (default: dump only)")
    ap.add_argument("--csv-name", default="screening_genbench.csv")
    ap.add_argument("--think", choices=("on", "off", "default"), default="default",
                    help="template thinking flag used by the ORIGINAL run "
                         "(must match, or the rebuilt prompt is the wrong prefix)")
    A = ap.parse_args()

    genprotocol.check_dump_dir()
    if A.dry_run:
        # Validate the WHOLE path (load, split, prompt rebuild, merge, score,
        # dump write) with a stub engine. Everything here is a no-GPU code path,
        # and two GPU-hours were lost to bugs in it before this existed.
        from transformers import AutoTokenizer
        _tk = AutoTokenizer.from_pretrained(A.path)
        _rebuilt = _rebuild_prompts(A, _tk) if A.rebuild_prompt else None
        for _d in A.dump.split(","):
            _one(A, _d.strip(), None, _tk, _rebuilt)
        print("[resume] DRY RUN OK: plumbing validated, no generation performed")
        return
    vllm_glue.install()
    from vllm import LLM
    _llm = LLM(model=A.path, enforce_eager=True, gpu_memory_utilization=A.gpu_mem,
               max_model_len=A.max_model_len or (A.new_cap + 2048),
               enable_prefix_caching=False)
    _tk = _llm.get_tokenizer()
    _rebuilt = _rebuild_prompts(A, _tk) if A.rebuild_prompt else None
    for _d in A.dump.split(","):
        _one(A, _d.strip(), _llm, _tk, _rebuilt)


def _one(A, dump, llm, tk, rebuilt):
    src = os.path.join(SAMP, dump + ".json")
    blob = json.load(open(src))
    items = blob["items"]
    old_cap = A.old_cap or max(i["gen_toks"] for i in items)

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
    print(f"[resume] {dump}: {len(items)} items, {len(done)} finished (reused "
          f"as-is), {len(trunc)} truncated -> continuing each by up to {add} tokens",
          flush=True)
    print(f"[resume] tokens to decode: {len(trunc) * add} "
          f"(a full re-run would decode ~{len(items) * A.new_cap})", flush=True)
    if not trunc:
        print("[resume] nothing truncated; nothing to do")
        return

    # with several dumps in one boot the arm MUST come from each stem; a global
    # --arm would silently label every cell with the first one's arm
    arm = A.arm if "," not in A.dump else None
    if arm is None:
        for part in dump.split("_"):
            if part == "free" or (part.startswith("R") and part[1:].isdigit()):
                arm = part
    assert arm, "could not parse arm from --dump; pass --arm"
    R = None if arm == "free" else int(arm.lstrip("R"))

    if not A.dry_run:               # dry-run must import nothing GPU-side, so it
        from vllm import SamplingParams        # can run in any venv
        from vllm.inputs import TokensPrompt
    else:
        SamplingParams = dict
        def TokensPrompt(prompt_token_ids):
            return {"prompt_token_ids": prompt_token_ids}

    # Build the resume prompts: original prompt + everything generated so far. The
    # boundary tells the walker which part stays free.
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
        it["_prefix_ids"] = gids      # the prefix ACTUALLY used (engine or retokenized);
                                      # the merge needs it, and for a retokenized item
                                      # there is no "gen_ids" on the item to fall back to
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
    sp = SamplingParams(temperature=gc.get("temperature", dt),   # noqa: F841
                        top_p=gc.get("top_p", dp), top_k=gc.get("top_k") or -1,
                        seed=1234, max_tokens=add, skip_special_tokens=False)

    DEC.update(on=R is not None, R=R or 0, swaps=1)
    DEC["state"].clear()
    t0 = time.time()
    if A.dry_run:
        class _Out:                       # minimal stand-in for a RequestOutput
            def __init__(self):
                self.outputs = [type("o", (), {"token_ids": [0, 1, 2],
                                               "text": "<dry-run tail>"})()]
        outs = [_Out() for _ in prompts]
    else:
        outs = llm.generate(prompts, sp)
    secs = time.time() - t0

    merged = list(done)
    still = 0
    for it, o in zip(meta, outs):
        tail = o.outputs[0]
        new_ids = list(it["_prefix_ids"]) + list(tail.token_ids)
        merged.append({**{k: v for k, v in it.items()
                          if k not in ("_prefix_source", "_prefix_ids")},
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
    task = dump.split(f"_{arm}_", 1)[1] if f"_{arm}_" in dump else "resumed"
    genprotocol.write_dump(A.record_as + ("_dryrun" if A.dry_run else ""),
                           arm, task, merged, len(items))
    if A.score:
        _score(A, merged, arm, R, secs)
    print(f"[resume] prefixes: {src_count['engine_ids']} exact (engine ids), "
          f"{src_count['retokenized']} best-effort (re-tokenized text)", flush=True)
    print(f"[resume] continued {len(trunc)} items in {secs:.0f}s; "
          f"{still} still at the new cap ({100*still/len(items):.1f}% of the cell)",
          flush=True)


if __name__ == "__main__":
    main()
