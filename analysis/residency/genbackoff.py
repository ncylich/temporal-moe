#!/usr/bin/env python3
"""Continuation backoff + reasoning-aware scoring for lm_eval generative runs.

One component owns the generation lifecycle so no layer can hide raw text from another
(the layering bugs of 2026-08-12/13: filtered dumps fed truncation audits; stripped
text almost fed continuations):

1. STOPS: every request is sent to vLLM with eos-only stops (task stop-strings fire
   inside think blocks -- "Q:", "\\ndef"); the task's stops are applied AFTER the
   think-strip, matching lm_eval's think_end_token semantics exactly.
2. CONTINUATION, never regeneration: a response that hits its budget is resubmitted as
   context + raw partial output with an incremental budget (+B, +2B, ... to `cap`).
   Regeneration under sampling is a fresh trajectory draw -- a hidden retry biased
   toward the items the model struggles on, at arm-dependent rates -- and re-pays the
   whole prefix. Continuation keeps the original trajectory committed and spends only
   the marginal tokens.
3. SCORING TEXT: think segment stripped (text after the LAST `think_marker`; marker
   absent => whole text, lm_eval-identical), then task stops applied.
4. CAPTURE: FINALS maps doc_id -> final raw text, exactly one entry per item -- the
   doc-aligned source for dumps and think-length analysis (resolves the
   retry-inclusive oversampling limitation in PROTOCOL_ERAS.md for cells produced
   after 2026-08-13 ~21:30 UTC).

Responses still at `cap` are scored as-is and counted (degeneracy suspects).
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

        outs = orig(reqs, **kw)
        budget = {i: int(dict(r.args[1]).get("max_gen_toks", base_toks))
                  for i, r in enumerate(reqs)}
        live = list(range(len(reqs)))
        while True:
            trunc = [i for i in live
                     if budget[i] < cap and _ntoks(outs[i]) >= budget[i] - 8]
            if not trunc:
                break
            retry = []
            for i in trunc:
                inc = min(budget[i], cap - budget[i])
                rr = copy.copy(reqs[i])
                ctx, gk = rr.args[0], dict(rr.args[1])
                gk["max_gen_toks"] = inc
                rr.arguments = (ctx + outs[i], gk)     # CONTINUATION from raw partial
                retry.append((i, inc, rr))
            print(f"  [backoff] continuing {len(trunc)}/{len(reqs)} truncated "
                  f"responses (+{retry[0][1]} tokens)", flush=True)
            tails = orig([rr for _, _, rr in retry], **kw)
            for (i, inc, _), tail in zip(retry, tails):
                outs[i] += tail
                budget[i] += inc
            live = trunc
        capped = sum(_ntoks(o) >= cap - 8 for o in outs)
        if capped:
            print(f"  [backoff] {capped} responses at hard cap {cap} "
                  f"(degeneracy suspects, scored as-is)", flush=True)

        FINALS.clear()
        for r, raw in zip(requests, outs):
            FINALS[getattr(r, "doc_id", None)] = raw
        return [_score_text(raw, u) for raw, u in zip(outs, untils)]

    lm.generate_until = gu
