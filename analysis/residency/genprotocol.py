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
"""
import copy

HARD_CAP = 2048
FINALS = {}          # (task_name, doc_id) -> final raw response text, current cell


def install(lm, cap=HARD_CAP, think_marker=None):
    orig = lm.generate_until
    tok = lm.tokenizer

    def _ntoks(text):
        return len(tok(text, add_special_tokens=False).input_ids)

    def _score_text(raw, until):
        text = raw.split(think_marker)[-1] if think_marker else raw
        for term in until or []:
            if term:
                text = text.split(term)[0]
        return text

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
        outs = orig(reqs, **kw)
        capped = sum(_ntoks(o) >= cap - 8 for o in outs)
        if capped:
            print(f"  [backoff] {capped} responses at hard cap {cap} "
                  f"(degeneracy suspects, scored as-is)", flush=True)

        FINALS.clear()
        for r, raw in zip(requests, outs):
            FINALS[(getattr(r, "task_name", None), getattr(r, "doc_id", None))] = raw
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
