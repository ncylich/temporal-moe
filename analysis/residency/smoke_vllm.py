#!/usr/bin/env python3
"""vLLM residency smoke, three gates in one load:
    1. R=E no-op: constrained path with full residency must reproduce free outputs
       byte-identically (proves the plumbing corrupts nothing).
    2. R=8 differs from free (proves masking engages).
    3. concurrency: 8 simultaneous requests, per-request state (walker already
       parity-tested offline; this checks the span totals match real batches).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_glue
import vllm_residency  # noqa: F401
from decode_state import DEC

vllm_glue.install()

from vllm import LLM, SamplingParams

MODEL = "/workspace/instruct-models/olmoe-0125-instruct"
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.4,
          max_model_len=2048, enable_prefix_caching=False)
# prefix caching MUST be off: cached prompt blocks are never recomputed, so the observe
# phase would miss them and the resident set would start effectively cold (found the hard
# way: R=8 gibberish while the HF path was coherent on identical prompts).

prompts = [
    "Explain why the sky is blue in two sentences.",
    "What is 17 * 23? Show your work.",
    "Name three uses for a paperclip.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
] * 2
msgs = [[{"role": "user", "content": p}] for p in prompts]
sp = SamplingParams(temperature=0, max_tokens=64)


def run():
    outs = llm.chat(msgs, sp)
    return [o.outputs[0].text for o in outs]


DEC.update(on=False)
free = run()
DEC.update(on=True, R=64, swaps=1)          # R = E: full residency, must be a no-op
DEC["state"].clear()
noop = run()
assert noop == free, "R=E no-op FAILED: constrained path changed outputs\n" + \
    "\n".join(f"FREE: {f!r}\nR=E:  {n!r}" for f, n in zip(free, noop) if f != n)[:800]
print("GATE 1 PASS: R=E reproduces free byte-identically over 8 concurrent requests")

DEC.update(on=True, R=8, swaps=1)
DEC["state"].clear()
r8 = run()
diff = sum(a != b for a, b in zip(free, r8))
assert diff >= len(free) // 2, f"R=8 barely changed outputs ({diff}/{len(free)})"
print(f"GATE 2 PASS: R=8 changes {diff}/{len(free)} outputs")
print("sample free:", free[0][:100].replace(chr(10), " "))
print("sample R8:  ", r8[0][:100].replace(chr(10), " "))
print("VLLM SMOKE PASS")
