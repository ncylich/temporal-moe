#!/usr/bin/env python3
"""Per-family tokenizer round-trip tests: what survives text, and what does not.

Resuming a truncated generation means rebuilding its token sequence. This isolates
which parts of that are safe and which are not, per model family, on synthetic
prompts and answers (thinking on and off wherever the family supports both):

  1. CHAT TEMPLATE / PROMPT - tokenizing the templated prompt is deterministic and
     stable, so the prompt half of a resume is exactly recoverable.
  2. THINK / CHANNEL MARKERS - each family's markers are single stable tokens that
     decode back to themselves, in isolation and in context.
  3. ANSWER TEXT - decode(encode(text)) is byte-identical.
  4. SEGMENTATION - and yet the token sequence is NOT recoverable: the same text
     has multiple valid tokenizations, and a sampler may emit any of them. This is
     the one failure mode, and it is why dumps must carry engine token IDs.

Test 4 constructs the failure the way it actually happens: build an id sequence
the way a sampler could emit it (' ' + 'Python'), decode it, re-tokenize, and
compare ids -- identical text, different ids.

Run: python analysis/residency/tests/test_tokenizer_families.py
"""
import os
import sys

FAMILIES = [
    # (name, tokenizer, think modes to exercise, markers)
    ("gemma4", "google/gemma-4-26B-A4B-it", [True, False], ["<|channel>", "<channel|>"]),
    ("qwen35", "/workspace/instruct-models/qwen35-35b-a3b-instruct", [True, False],
     ["<think>", "</think>"]),
    ("lfm25", "LiquidAI/LFM2.5-8B-A1B", [None], ["<think>", "</think>"]),
    ("gptoss", "openai/gpt-oss-120b", [None],
     ["<|channel|>", "<|message|>", "<|start|>"]),
    ("olmoe", "allenai/OLMoE-1B-7B-0125-Instruct", [None], []),
]
PROMPTS = ["What is 2 + 2?",
           "Complete the function:\n\ndef add(a, b):",
           "Which is larger, 9.11 or 9.9?"]
ANSWERS = ["The answer is (B).",
           "```python\ndef add(a, b):\n    return a + b\n```",
           "Let's think step by step. 9.9 > 9.11, so the answer is 9.9."]


def main():
    from transformers import AutoTokenizer
    ok = True
    for name, path, modes, marks in FAMILIES:
        try:
            tk = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            print(f"{name}: SKIP (tokenizer unavailable: {type(e).__name__})")
            continue
        print(f"\n=== {name}")

        # 1. chat template + prompt: deterministic and stable
        for mode in modes:
            kw = {} if mode is None else {"enable_thinking": mode}
            label = "default" if mode is None else ("think-on" if mode else "think-off")
            try:
                a = tk.apply_chat_template([{"role": "user", "content": PROMPTS[0]}],
                                           tokenize=False, add_generation_prompt=True, **kw)
                b = tk.apply_chat_template([{"role": "user", "content": PROMPTS[0]}],
                                           tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:
                print(f"  prompt/{label:9s}: SKIP ({type(e).__name__})")
                continue
            stable = a == b
            ids1 = tk(a, add_special_tokens=False).input_ids
            ids2 = tk(a, add_special_tokens=False).input_ids
            det = ids1 == ids2
            rt = tk.decode(ids1) == a
            good = stable and det and rt
            ok &= good
            print(f"  prompt/{label:9s}: template stable={stable} tokenization "
                  f"deterministic={det} decode-round-trip={rt}  "
                  f"{'ok' if good else 'FAIL'}")

        # 2. markers: single stable tokens, alone and in context
        for m in marks:
            ids = tk(m, add_special_tokens=False).input_ids
            solo = len(ids) == 1 and tk.decode(ids) == m
            ctx = f"reasoning text {m} answer text"
            cids = tk(ctx, add_special_tokens=False).input_ids
            inctx = tk.decode(cids) == ctx and ids[0] in cids
            ok &= solo and inctx
            print(f"  marker {m!r:14s}: single-token={solo} survives-in-context={inctx}"
                  f"  {'ok' if solo and inctx else 'FAIL'}")

        # 3. answers: text round-trips byte-identically
        bad = [a for a in ANSWERS
               if tk.decode(tk(a, add_special_tokens=False).input_ids) != a]
        ok &= not bad
        print(f"  answer text round-trip: {len(ANSWERS) - len(bad)}/{len(ANSWERS)} "
              f"byte-identical  {'ok' if not bad else 'FAIL'}")

        # 4. THE failure mode: same text, different valid id sequence
        demo = None
        for a in ANSWERS:
            greedy = tk(a, add_special_tokens=False).input_ids
            for pos, t in enumerate(greedy):
                s = tk.decode([t])
                if len(s) < 4:
                    continue
                for cut in range(1, len(s)):
                    x, y = tk.tokenize(s[:cut]), tk.tokenize(s[cut:])
                    if len(x) == len(y) == 1:
                        ix = tk.convert_tokens_to_ids(x) + tk.convert_tokens_to_ids(y)
                        if tk.decode(ix) == s:
                            alt = greedy[:pos] + ix + greedy[pos + 1:]
                            demo = (s, tk.decode(ix[:1]), tk.decode(ix[1:]),
                                    tk.decode(alt) == a, len(greedy), len(alt))
                            break
                if demo:
                    break
            if demo:
                break
        if demo:
            s, p1, p2, same_text, n1, n2 = demo
            print(f"  segmentation: {s!r} == {p1!r}+{p2!r} -> same text={same_text}, "
                  f"{n1} vs {n2} tokens  <-- NOT recoverable from text")
        else:
            print("  segmentation: no split site found in these samples")
    print("\n" + ("PASS (markers, prompts and answer text all recoverable; "
                  "only segmentation is not)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
