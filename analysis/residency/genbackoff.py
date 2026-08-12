#!/usr/bin/env python3
"""Truncation backoff for lm_eval generative runs: retry budget-truncated responses at
doubled max_gen_toks until they complete naturally or hit a hard cap (default 2048).

Under greedy decoding a retried prompt regenerates the same continuation with more
room, so the result equals running every prompt at the cap while only the truncated
tail pays for it. Detection: a response whose token count reaches the round's budget
was cut by it (stop-string-trimmed responses land below). The identical schedule runs
on every arm, so generation budget stops being a confounder between free and
constrained cells. Responses still at the cap are scored as-is and counted -- a
capped-at-2048 response is a degeneracy signature, not a budget problem.

Install BEFORE any output-rewriting wrapper (e.g. the gpt-oss final-channel filter):
token counts must see the raw generation.
"""
import copy

HARD_CAP = 2048


def install(lm, base_toks, cap=HARD_CAP):
    orig = lm.generate_until
    tok = lm.tokenizer

    def _ntoks(text):
        return len(tok(text, add_special_tokens=False).input_ids)

    def gu(requests, **kw):
        outs = orig(requests, **kw)
        B, live = base_toks, list(range(len(requests)))
        while B < cap:
            trunc = [i for i in live if _ntoks(outs[i]) >= B - 8]
            if not trunc:
                return outs
            B = min(2 * B, cap)
            retry = []
            for i in trunc:
                r = copy.copy(requests[i])
                ctx, gk = r.args[0], dict(r.args[1])
                gk["max_gen_toks"] = B
                r.args = (ctx, gk)
                retry.append(r)
            print(f"  [backoff] {len(trunc)}/{len(requests)} truncated, retrying at {B}",
                  flush=True)
            for i, t in zip(trunc, orig(retry, **kw)):
                outs[i] = t
            live = trunc
        capped = sum(_ntoks(outs[i]) >= cap - 8 for i in live)
        if capped:
            print(f"  [backoff] {capped} responses still at hard cap {cap} "
                  f"(degeneracy suspects, scored as-is)", flush=True)
        return outs

    lm.generate_until = gu
