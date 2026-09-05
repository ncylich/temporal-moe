#!/usr/bin/env python3
"""Generation protocol for lm_eval generative runs: budget, stops, think-strip, capture.

One component owns the generation lifecycle:

1. BUDGET: every request runs in a single pass at `cap` (vLLM allocates KV on demand,
   so unused budget is free; early-EOS sequences release their slots). Responses that
   finish at the cap are counted and scored as-is (degeneracy suspects).
2. STOPS: vLLM receives eos-only stops (task stop-strings fire inside think blocks);
   the task's stops are applied after the think-strip.
3. SCORING TEXT: think segment stripped (text after the LAST `think_marker`; marker
   absent => whole text), then task stops applied.
4. CAPTURE: FINALS maps (task_name, doc_id) -> final raw text, one entry per item.
   The task_name component is load-bearing: grouped suites (mmlu_*) restart doc_id
   at 0 per subject, so a bare-doc_id key silently overwrites across subjects.

HOW TO COUNT GENERATION LENGTH FROM A DUMP
------------------------------------------
`gen_toks` does NOT mean the same thing in every dump, and reading it as the total
is wrong. It is the POST-STRIP scored answer of step 3; the thinking segment is
counted separately. Dumps also differ by era in which fields they carry, so decide
the route per dump, from the fields present, in this order:

  1. `raw_toks` present            total    = raw_toks           (authoritative)
                                   answer   = gen_toks
                                   thinking = raw_toks - gen_toks
  2. `raw` text present            total    = gen_toks           (this era's
                                   gen_toks already spans the whole generation)
                                   thinking = per-item `think_toks`
  3. `think_toks_by_doc` present   total    = gen_toks + think_by_doc
                                   answer   = gen_toks
     EXCEPT when think_by_doc == gen_toks. That is the marker-absent case: the
     generation ended inside its thinking block, so the producer stored the whole
     generation in think_by_doc and the strip removed nothing. The two fields are
     then ONE number and adding them double-counts. Use total = gen_toks.
  4. none of the above             total = gen_toks (surface has no thinking)

Two traps this rule exists to avoid. Both produced confident, wrong numbers before
being caught:

  * The per-item `think_toks` field is trustworthy ONLY alongside `raw`. In older
    dumps it measured post-strip text and reads near zero, so using it as a
    denominator reported a thinking ratio of 3.0x where the truth is 1.19x.
  * Item keys are `doc_id` (int) in older dumps and `doc` (str) in newer ones. Cast
    to str before pairing, or an old-to-new comparison silently intersects to
    nothing and the cell vanishes with no error.

Validated on the 3800 items whose dumps carry both `raw_toks` and
`think_toks_by_doc`: 49% are the marker-absent case, 51% satisfy
answer + thinking = total within 2%, together 100%, median ratio 1.000.
Reference implementation: `lengths()` in analysis/residency/length_figs.py.

Cap-hit (truncation) is measured against the cell's DECLARED budget, never the
observed maximum: one over-cap outlier shifts a max-based reference past the
pile-up at the cap and hides it. One arm read 0.5% truncated by observed max and
8.0% against its declared 8192.
"""
import copy

HARD_CAP = 2048
FINALS = {}          # (task_name, doc_id) -> final raw response text, current cell
GEN_IDS = {}         # (task_name, doc_id) -> engine token IDs for that response
# Engine IDs are the ONLY faithful prefix for resuming a truncated generation:
# the same text has many valid token sequences (a gemma item measured 3072 engine
# tokens against 3061 from re-tokenizing its own byte-identical text), and text
# alone cannot say which the sampler took.


def install(lm, cap=HARD_CAP, think_marker=None):
    orig = lm.generate_until
    tok = lm.tokenizer

    def _ntoks(text):
        return len(tok(text, add_special_tokens=False).input_ids)

    def _score_text(raw, until):
        # A response that ran out of budget INSIDE its thinking block never emitted
        # an answer. Splitting on an absent marker returns the whole trace, which
        # hands the raw deliberation to the answer extractor -- and deliberation is
        # full of "the answer is (D)... or (B)" asides, so the extractor scores a
        # coin flip on the model's own scratch work (measured on qwen MMLU: 27% of
        # items unfinished, 0.66 accuracy on them against 0.96 on finished ones).
        # Unfinished => no answer. Only cap-truncated responses count as unfinished;
        # a short response with no marker simply did not think.
        if think_marker and think_marker not in raw and _ntoks(raw) >= cap - 8:
            return ""
        text = raw.split(think_marker)[-1] if think_marker else raw
        for term in until or []:
            if term:
                text = text.split(term)[0]
        return text

    def _stash_ids(outs_raw):
        """vLLM RequestOutputs -> per-request token IDs, in submission order."""
        try:
            return [list(o.outputs[0].token_ids) for o in outs_raw]
        except Exception:
            return None

    def gu(requests, **kw):
        # eos-only stops during generation; remember task stops for post-strip
        reqs, untils = [], []
        for r in requests:
            ctx, gk = r.args[0], dict(r.args[1])
            untils.append(gk.get("until") or [])
            gk["until"] = []
            rr = copy.copy(r)
            rr.arguments = (ctx, gk)
            reqs.append(rr)

        # SINGLE PASS AT THE CAP: in vLLM, unused max_tokens is free (early-EOS
        # sequences release their slot; concurrency is set by max_model_len, which
        # is already sized to prompt+cap). Submitting at the cap keeps every
        # trajectory's KV resident until completion -- no truncation, no
        # continuation, no re-prefill, no re-roll bias. The ladder is retired.
        for rr in reqs:
            ctx, gk = rr.args[0], dict(rr.args[1])
            gk["max_gen_toks"] = cap
            rr.arguments = (ctx, gk)
        _cap_ids = []
        _mg = getattr(lm, "_model_generate", None)
        if _mg is not None and not getattr(lm, "_gp_wrapped", False):
            def _wrap(*a, **k):
                r = _mg(*a, **k)
                ids = _stash_ids(r)
                if ids:
                    _cap_ids.extend(ids)
                return r
            lm._model_generate = _wrap
            lm._gp_wrapped = True
        outs = orig(reqs, **kw)
        capped = sum(_ntoks(o) >= cap - 8 for o in outs)
        if capped:
            print(f"  [backoff] {capped} responses at hard cap {cap} "
                  f"(degeneracy suspects, scored as-is)", flush=True)

        FINALS.clear()
        GEN_IDS.clear()
        for idx, (r, raw) in enumerate(zip(requests, outs)):
            key = (getattr(r, "task_name", None), getattr(r, "doc_id", None))
            FINALS[key] = raw
            if idx < len(_cap_ids):
                GEN_IDS[key] = _cap_ids[idx]
        return [_score_text(raw, u) for raw, u in zip(outs, untils)]

    lm.generate_until = gu


# --- per-item trajectory dumps: default-on, verified ---------------------------
DUMP_DIR = None


def check_dump_dir():
    """Startup gate, called BEFORE the engine boots: the dump directory must be
    writable or the run must not start. A benchmark run without its trajectory
    dump is unrepeatable evidence loss."""
    global DUMP_DIR
    import os
    from paths import ABLATIONS
    d = os.path.join(ABLATIONS, "genbench_samples")
    os.makedirs(d, exist_ok=True)
    probe = os.path.join(d, ".write_probe")
    open(probe, "w").close()
    os.remove(probe)
    DUMP_DIR = d
    return d


def write_dump(record, arm, task, items, expected_n, extra=None):
    """Write the (record, arm, task) per-item dump and verify the round-trip count.
    Items must carry a doc key ("doc") and the raw pre-strip text ("raw", thinking
    markers / channels intact). expected_n is the number of items evaluated; a
    mismatch is a hard failure (a per-subject overwrite once silently kept 4 of
    228 items -- this makes that bug class detected, not survivable)."""
    import json
    import os
    d = DUMP_DIR or check_dump_dir()
    assert items and len(items) == expected_n, \
        f"dump has {len(items)} items, evaluated {expected_n} ({record} {arm} {task})"
    missing = [i for i, x in enumerate(items) if "doc" not in x or "raw" not in x]
    assert not missing, f"items missing doc/raw keys at idx {missing[:5]}"
    path = os.path.join(d, f"{record}_{arm}_{task}.json")
    json.dump({"items": items, **(extra or {})}, open(path, "w"))
    n_back = len(json.load(open(path))["items"])
    assert n_back == expected_n, f"dump verify failed ({n_back} != {expected_n}): {path}"
    print(f"  [dump] {path}: {n_back} items verified", flush=True)
    return path
