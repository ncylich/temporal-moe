#!/usr/bin/env python3
"""Rebuild the d7 prompt pool: 9,173 benchmark-free prompts in four lanes.

The original pool died with the training pod and NO builder was ever committed -- a grep
across every commit in the repository, not just the tip, finds zero code hits for
domain8k, mathlane, mcq-writer or mcq_writer. What survives is a specification in prose
(results/ablations/gemma_adapt_RESULTS.md, "Recipe (all settings load-bearing)"):

    domain8k              4,958   (including 431 code rows)
    mathlane_v2           2,341
    d5 few-shot variants  1,183
    mcq-writer              691
    total                 9,173

This is therefore a REBUILD, not a reproduction. A differently constituted pool moves the
result -- that is exactly what the D1-vs-D4 ablation demonstrated -- so adapters trained on
this pool land NEAR, not ON, the published Section 8 numbers. See RECOVER_DATA_PLAN section
1.5 for the two honest dispositions.

Lineage, the load-bearing constraint
------------------------------------
The rule is "no benchmark-family data in any form (test/train splits, synthetic
derivatives)", and gemma_adapt_RESULTS.md records why: an Orca-Math lane, seeded from
GSM8K-train, produced "a fake +8 GSM8K that vanished when the lane was removed --
style-matching, not constraint robustness".

So the ban is enforced by PROVENANCE, at lane-selection time, not by post-filtering. Every
source here is a corpus of real human-written requests:

    allenai/WildChat-1M          real user<->model chats            ODC-BY
    OpenAssistant/oasst2         human-written instruction trees    Apache-2.0

None is derived from any benchmark. Math prompts come from people asking math questions,
never from a math benchmark or its synthetic descendants -- which is precisely why
Orca-Math and MetaMathQA (both seeded from GSM8K/MATH train splits) are excluded despite
being the obvious way to fill a math lane.

The 8-gram screen against the GSM8K, MMLU, HumanEval and IFEval TEST sets is then an AUDIT
of that provenance claim, not the mechanism enforcing it. halfgrain_RESULTS.md records the
expected result for prompts: 0 overlaps out of 2,793.

Determinism
-----------
No RNG anywhere. Sources are read in a fixed order and the first N rows passing each lane's
filter are kept, so a regeneration is byte-identical and checkable against the sha256 in
the meta json -- the discipline build_wildchat_prompts.py already uses.

    build_d7_prompts.py [--out DIR] [--scan-cap N]
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_TOKEN", open(os.path.expanduser(
    "~/.cache/huggingface/token")).read().strip())

# Lane sizes are the specification. Do not "round" them: the budget these feed is stated in
# response tokens, but the pool's composition is what the ablation ladder was run against.
# mcq-writer (691) is deliberately ABSENT. It appeared once in the whole repository, in a
# single prose parenthesis, and no builder for it was ever committed. It was also never
# isolated -- the screening set has scr_d1..scr_d12 and scr_dom{,2}_cand but no
# scr_mcq_cand -- so no published number can be attributed to it. Real human-written
# corpora cannot fill it either: the strict multiple-choice pattern yields 184 matches
# across all of WildChat-1M against the 691 needed, so the original lane was generated
# rather than sampled. Rather than manufacture 691 templated rows to make a count match,
# the lane is dropped and the removal is stated. Pre-registered consequence: it was the
# only MMLU-format-facing lane, so if re-measured MMLU lands well below the published
# -1.8 (R8) / -2.8 (free), this absence is the first suspect.
LANES = {"domain8k": 4958, "mathlane_v2": 2341, "d5_fewshot": 1183}
CODE_ROWS = 431                      # of domain8k, per the recipe

NGRAM = 8

_WORD = re.compile(r"[a-z0-9]+")


def grams(text, n=NGRAM):
    """Normalized n-gram set: lowercase alphanumeric words, punctuation and case dropped.

    Matching on raw text would miss a paraphrase-free copy that differs only in
    punctuation, which is the case the screen exists to catch."""
    w = _WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def benchmark_grams():
    """Every 8-gram in the four benchmark TEST sets the pool must not touch.

    Question text AND answer/choice text: a prompt that reproduces an MMLU option or a
    HumanEval body is as much a lineage breach as one reproducing the stem."""
    from datasets import load_dataset
    out, per = set(), {}
    def add(tag, texts):
        g = set()
        for t in texts:
            if t:
                g |= grams(t)
        per[tag] = len(g)
        out.update(g)
        print(f"[d7] {tag}: {len(g)} distinct {NGRAM}-grams", flush=True)

    d = load_dataset("openai/gsm8k", "main", split="test")
    add("gsm8k/test", [r["question"] for r in d] + [r["answer"] for r in d])

    d = load_dataset("cais/mmlu", "all", split="test")
    add("mmlu/test", [r["question"] for r in d]
        + [c for r in d for c in (r["choices"] or [])])

    d = load_dataset("openai/openai_humaneval", split="test")
    add("humaneval", [r["prompt"] for r in d] + [r["canonical_solution"] for r in d])

    d = load_dataset("google/IFEval", split="train")
    add("ifeval", [r["prompt"] for r in d])

    print(f"[d7] screen: {len(out)} distinct {NGRAM}-grams over four test sets", flush=True)
    return out


def cached_screen(path):
    """benchmark_grams(), memoised. The four TEST sets are frozen, so the screen is a
    pure function of the datasets; rebuilding it each run costs minutes and buys nothing."""
    if path and os.path.exists(path):
        with open(path) as fh:
            g = {ln.rstrip("\n") for ln in fh}
        print(f"[d7] screen: {len(g)} {NGRAM}-grams from cache {path}", flush=True)
        return g
    g = benchmark_grams()
    if path:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(sorted(g)))
        os.replace(tmp, path)               # atomic: a torn cache must never be readable
        print(f"[d7] screen cached -> {path}", flush=True)
    return g


# ----------------------------------------------------------------- lane classifiers ----
# Applied in priority order (mcq -> math -> code -> domain), so a prompt asking for
# multiple-choice math questions lands in mcq_writer rather than mathlane, matching the
# lane names' plain meaning: mcq-writer is a WRITING task, not a math task.

_MCQ = re.compile(r"\b(multiple[- ]choice|mcq|quiz|exam questions?|test questions?|"
                  r"practice questions?|comprehension questions?)\b", re.I)
_MAKE = re.compile(r"\b(write|create|generate|make|compose|come up with|draft|design|"
                   r"prepare|produce|give me|provide|list)\b", re.I)
_MATH = re.compile(r"\b(solve|calculate|compute|equation|derivative|integral|probability|"
                   r"algebra|geometry|arithmetic|factorial|polynomial|logarithm|theorem|"
                   r"fraction|percentage|simplify|evaluate the (?:expression|limit))\b", re.I)
# Deliberately NARROW. An earlier draft matched "function", "class", "api" and
# "algorithm", which swallowed the math lane whole: "find the derivative of the function
# f(x)" is a math prompt, not a code prompt, and mathlane collected 174 rows where it
# should have collected thousands. Only unambiguous programming signals belong here --
# a fenced block, a named language, or an explicitly programming verb phrase.
_CODE = re.compile(
    r"```"
    r"|\b(?:python|javascript|typescript|golang|rust|sql|regex|html|css|bash|powershell"
    r"|shell script|kotlin|swift|php|java|c\+\+|c#|def |stack trace|traceback|compiler?"
    r"|debug|syntax error|pull request)"
    r"|write (?:a |the )?(?:program|script|query|function in)", re.I)


def lane_of(text):
    if _CODE.search(text):
        return "code"                      # a sub-quota inside domain8k
    if _MATH.search(text) or re.search(r"\d+\s*[+\-*/^=]\s*\d+", text):
        return "mathlane_v2"
    return "domain"


def clean(text):
    t = " ".join((text or "").split())
    return t if 30 <= len(t) <= 2000 else None


# ------------------------------------------------------------------------- sources ----

def wildchat(scan_cap):
    """First user turn of English, single-turn, non-toxic, non-redacted conversations."""
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    for i, r in enumerate(ds):
        if i >= scan_cap:
            return
        if r["language"] != "English" or r["turn"] != 1 or r["toxic"] or r["redacted"]:
            continue
        conv = r["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue
        t = clean(conv[0]["content"])
        if t:
            yield "wildchat", r["conversation_hash"], t


def oasst2_pairs():
    """Human-written (instruction, response) pairs: a root prompter message and its first
    human assistant reply.

    This lane is the only one needing an answer side, because a few-shot prompt carries
    exemplar responses inside the prompt text. Those responses must be human-written --
    WildChat's replies are model outputs, and seeding the context with another model's
    prose makes the lane an indirect distillation channel rather than a formatting drill.
    oasst2 is used rather than no_robots because it is Apache-2.0 (no_robots is
    CC-BY-NC-4.0, which does not belong in a pool released with the paper) and far larger,
    so the lane is not drawn almost entirely from one small annotation batch."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    roots, replies = {}, {}
    for r in ds:
        if r.get("lang") != "en" or r.get("deleted"):
            continue
        if not r.get("parent_id") and r.get("role") == "prompter":
            roots[r["message_id"]] = r.get("text")
        elif r.get("role") == "assistant" and r.get("parent_id"):
            replies.setdefault(r["parent_id"], r.get("text"))
    for mid, q in roots.items():                    # dict order == dataset order: deterministic
        t, resp = clean(q), clean(replies.get(mid))
        if t and resp:
            yield "oasst2", mid, t, resp


def oasst2(scan_cap):
    """Root prompts of English message trees (human-written, Apache-2.0)."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
    for i, r in enumerate(ds):
        if i >= scan_cap:
            return
        if r.get("parent_id") or r.get("lang") != "en" or r.get("deleted"):
            continue
        t = clean(r.get("text"))
        if t:
            yield "oasst2", r["message_id"], t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data")
    ap.add_argument("--scan-cap", type=int, default=1_000_000,
                    help="max rows to stream per streaming source")
    ap.add_argument("--max-prompt-tok", type=int, default=1024,
                    help="every row must fit gen_traj_vllm.py's --max-prompt-tok, which "
                         "drops longer prompts PRE-SUBMISSION, so screening here means "
                         "the built counts are the counts that reach training. 1024 is "
                         "chosen against the training sequence length, not vLLM: the "
                         "recipe trains at seq 4096, so 1024 of prompt leaves 3072 for "
                         "the response. Prompt tokens are not in the loss (response "
                         "tokens only); they cost sequence budget and KL-forward compute. "
                         "MUST match the value passed to gen_traj_vllm.py")
    ap.add_argument("--tok", nargs="+",
                    default=["/dev/shm/gemma4-26b-it", "/dev/shm/qwen35-35b-a3b"],
                    help="both consumers' tokenizers: a row is kept only if it fits for "
                         "EVERY model that will generate trajectories from this pool")
    ap.add_argument("--cache", default="/workspace/d7_screen_cache.txt",
                    help="8-gram screen cache; the four test sets are fixed, so building "
                         "735k grams on every run is pure waste")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    screen = cached_screen(A.cache)

    # Every consumer's tokenizer, so a row that fits gemma but not qwen never enters the
    # pool. gen_traj_vllm.py applies the chat template before measuring, so this must too.
    from transformers import AutoTokenizer
    toks = [AutoTokenizer.from_pretrained(t) for t in A.tok]
    print(f"[d7] prompt-length gate: <= {A.max_prompt_tok} tok for all of {A.tok}",
          flush=True)

    def fits(text):
        for t in toks:
            enc = t.apply_chat_template([{"role": "user", "content": text}],
                                        add_generation_prompt=True, tokenize=True,
                                        return_dict=True)
            if len(enc["input_ids"]) > A.max_prompt_tok:
                return False
        return True

    kept = {k: [] for k in LANES}
    code_kept = []
    seen_ids, seen_norm = set(), set()
    screened_out = {k: 0 for k in list(LANES) + ["code", "domain"]}
    too_long = {}

    def take(src, sid, text, lane):
        """Deduplicate, screen, and file into a lane. Returns True if kept."""
        if sid in seen_ids:
            return False
        norm = text.lower()
        if norm in seen_norm:
            return False
        if grams(text) & screen:
            screened_out[lane] = screened_out.get(lane, 0) + 1
            return False
        if not fits(text):
            too_long[lane] = too_long.get(lane, 0) + 1
            return False
        seen_ids.add(sid)
        seen_norm.add(norm)
        # "code" and "domain" are both selection lanes inside the single domain8k lane:
        # code is a sub-quota of it, so the emitted lane label collapses back to domain8k.
        target = code_kept if lane == "code" else kept["domain8k" if lane == "domain"
                                                      else lane]
        target.append({"lane": "domain8k" if lane in ("code", "domain") else lane,
                       "source": src, "source_id": str(sid), "text": text,
                       "is_code": lane == "code"})
        return True

    def need(lane):
        if lane == "code":
            return len(code_kept) < CODE_ROWS
        if lane == "domain":
            return len(kept["domain8k"]) < LANES["domain8k"] - CODE_ROWS
        return len(kept[lane]) < LANES[lane]

    # --- lanes 1-3, from the chat corpora, in a fixed source order -------------------
    for src, sid, text in wildchat(A.scan_cap):
        lane = lane_of(text)
        key = "domain8k" if lane == "domain" else lane
        if lane == "domain":
            if need("domain"):
                take(src, sid, text, "domain")
        elif need(lane):
            take(src, sid, text, lane)
        # stop as soon as every quota this source feeds is satisfied
        if (len(kept["mathlane_v2"]) >= LANES["mathlane_v2"]
                and len(code_kept) >= CODE_ROWS
                and len(kept["domain8k"]) >= LANES["domain8k"] - CODE_ROWS):
            break
        del key
    print(f"[d7] after WildChat: " + ", ".join(
        f"{k}={len(kept[k])}" for k in kept) + f", code={len(code_kept)}", flush=True)

    for src, sid, text in oasst2(A.scan_cap):
        lane = lane_of(text)
        if lane == "domain" and need("domain"):
            take(src, sid, text, "domain")
        elif lane != "domain" and need(lane):
            take(src, sid, text, lane)
    print(f"[d7] after oasst2: " + ", ".join(
        f"{k}={len(kept[k])}" for k in kept) + f", code={len(code_kept)}", flush=True)

    # --- lane 4: few-shot variants ---------------------------------------------------
    # The one lane that needs an answer side, so it is built from oasst2, whose
    # responses are human-written. Two real (instruction, response) exemplars are
    # prepended to a real third instruction; exemplars are the two rows immediately
    # preceding the target in stream order, so the construction is deterministic and
    # every token in the resulting prompt still came from the corpus.
    pool = [(s, i, t, r) for s, i, t, r in oasst2_pairs() if r]
    for n in range(2, len(pool)):
        if len(kept["d5_fewshot"]) >= LANES["d5_fewshot"]:
            break
        (_, _, t1, r1), (_, _, t2, r2) = pool[n - 2], pool[n - 1]
        src, sid, target, _ = pool[n]
        # Exemplars are trimmed, not truncated: a few-shot row that blows the prompt
        # budget is silently dropped by gen_traj_vllm.py, and an earlier build lost
        # 54% of this lane that way. The token gate in take() is the authority; the
        # cheap character pre-filter just avoids tokenising hopeless candidates.
        text = f"{t1}\n\n{r1}\n\n{t2}\n\n{r2}\n\n{target}"
        if len(text) > 3600:
            continue
        take(src, f"fewshot:{sid}", text, "d5_fewshot")
    print(f"[d7] after oasst2 few-shot: d5_fewshot={len(kept['d5_fewshot'])}", flush=True)

    kept["domain8k"] = kept["domain8k"] + code_kept

    # --- emit ------------------------------------------------------------------------
    short = {k: LANES[k] - len(v) for k, v in kept.items() if len(v) < LANES[k]}
    if short:
        print(f"[d7] WARNING short of spec: {short} -- raise --scan-cap or widen a lane",
              flush=True)

    rows = []
    for lane in ("domain8k", "mathlane_v2", "d5_fewshot"):
        for r in kept[lane][: LANES[lane]]:
            rows.append(dict(r, idx=len(rows)))

    path = os.path.join(A.out, "d7_prompts.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()

    counts = {}
    for r in rows:
        counts[r["lane"]] = counts.get(r["lane"], 0) + 1
    meta = {
        "path": path, "sha256": sha, "n": len(rows), "spec": LANES,
        "counts": counts, "code_rows": sum(1 for r in rows if r.get("is_code")),
        "ngram_screen": {"n": NGRAM, "test_sets": ["gsm8k", "mmlu", "humaneval", "ifeval"],
                         "screened_out": screened_out},
        "prompt_length_gate": {"max_prompt_tok": A.max_prompt_tok, "tokenizers": A.tok,
                               "rejected": too_long},
        "dropped_lane": {"mcq_writer": 691,
                         "reason": "never isolated in screening; no builder ever "
                                   "committed; unfillable from real corpora (184 strict "
                                   "matches in all of WildChat-1M). Removal stated "
                                   "rather than backfilled with templated rows."},
        "sources": ["allenai/WildChat-1M (ODC-BY)",
                    "OpenAssistant/oasst2 (Apache-2.0)"],
        "builder": "analysis/residency/build_d7_prompts.py",
        "note": "REBUILD of a lost pool, not a reproduction; see RECOVER_DATA_PLAN 1.1/1.5",
    }
    with open(os.path.join(A.out, "d7_prompts.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[d7] wrote {len(rows)} prompts -> {path}\n[d7] sha256 {sha}", flush=True)


if __name__ == "__main__":
    main()
