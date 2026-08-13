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
4. CAPTURE: FINALS maps doc_id -> final raw text, one entry per item.
"""
import copy

HARD_CAP = 2048
FINALS = {}          # doc_id -> final raw response text for the current cell


def install(lm, base_toks, cap=HARD_CAP, think_marker=None):
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
            FINALS[getattr(r, "doc_id", None)] = raw
        return [_score_text(raw, u) for raw, u in zip(outs, untils)]

    lm.generate_until = gu
