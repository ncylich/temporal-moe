#!/usr/bin/env python3
"""Generate the SELF-GENERATED math and code lanes the original d7 pool used.

Why this exists. The first rebuild sourced every lane from real corpora, and the result
was unambiguous: three of four benchmarks reproduced or beat the published row on both
gemma4 and qwen3.5, while GSM8K failed on both (-5.5 vs +0.0 published on gemma, -10.0 vs
-3.5 on qwen) and gemma's HumanEval only partly recovered (-4.9 vs -1.2, base -6.1).

Those are exactly the two lanes the rebuild changed in KIND. A saved transcript records the
original composition: math_selfgen (the model generating its own problems, alongside
selfmath_v2_3000.jsonl) and code (Magicoder-style, generated from OSS seeds). Only the chat
and math_user lanes were mined from real conversations -- and those two are precisely the
lanes whose benchmarks reproduced.

So the hypothesis under test is narrow: constraint-robustness on math and code comes from
training on the model's OWN generated problems, not from real-corpus prompts on the same
topics. This script produces those lanes; everything downstream is unchanged.

Lineage: still benchmark-free BY CONSTRUCTION. The seeds are topic strings and OSS code
snippets, never benchmark items, and the 8-gram screen in build_d7_prompts.py still runs
over the result. Orca-Math and MetaMathQA remain excluded -- the D1/D4 ablation is why.

    build_selfgen_lanes.py --model /dev/shm/gemma4-26b-it --lane math --n 2341
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Topic seeds. Deliberately mundane and broad: the point is problem VARIETY, not topical
# overlap with any benchmark. No benchmark subject taxonomy is used -- notably NOT MMLU's
# 57 subject names, which would be benchmark-adjacent by the back door.
MATH_TOPICS = [
    "buying groceries", "splitting a restaurant bill", "planning a road trip",
    "a school bake sale", "filling a swimming pool", "stacking firewood",
    "a paper route", "sharing sweets between friends", "painting a fence",
    "train timetables", "knitting a scarf", "a lemonade stand",
    "renting bicycles", "counting farm animals", "a charity run",
    "tiling a kitchen floor", "saving pocket money", "a book club",
    "mixing paint colours", "loading a delivery van", "a garden allotment",
    "scoring a sports tournament", "a camping trip", "recycling bottles",
    "a bus timetable", "buying school supplies", "a pizza party",
    "building a bookshelf", "a swimming lesson schedule", "feeding pets",
    "a craft fair stall", "planting a vegetable patch", "a car journey's fuel",
    "sorting laundry", "a music practice schedule", "buying concert tickets",
    "a bicycle repair shop", "measuring rainfall", "a museum field trip",
    "packing moving boxes", "a coffee shop order", "raising chickens",
    "a marathon training plan", "wallpapering a room", "a fish tank",
    "selling handmade jewellery", "a school raffle", "baking bread",
    "a taxi fare", "counting library books", "a snow shovelling round",
    "a farmers market stall", "assembling flat-pack furniture", "a kite festival",
]
MATH_STYLES = [
    "a multi-step word problem needing at least three arithmetic operations",
    "a word problem involving fractions or percentages",
    "a word problem about rates, speed or work done over time",
    "a word problem needing a simple algebraic unknown",
    "a word problem comparing two options to find which is cheaper",
]
# A third axis, because topics x styles alone repeated every 120 prompts and sampling then
# produced 1931 near-duplicates out of 2341. Varying the concrete constraints forces
# genuinely different problems rather than paraphrases of the same one.
MATH_TWISTS = [
    "Use two-digit quantities throughout.",
    "Include a discount or a surcharge.",
    "Include a unit conversion.",
    "Make one quantity unknown until the final step.",
    "Include a leftover or remainder that matters.",
    "Involve three different people or groups.",
    "Span two different days or sessions.",
    "Include a rate given per unit.",
    "Require rounding to a sensible whole number.",
    "Include one piece of information that is not needed.",
]
CODE_STYLES = [
    "a self-contained Python function with a docstring describing the task",
    "a small Python utility that processes a list and returns a result",
    "a Python function that parses or reformats a string",
    "a Python function implementing a simple algorithm with edge cases",
    "a Python function that validates its input and raises on bad values",
    "a Python function that aggregates records into a summary",
]
# Same third axis as the math lane, for the same reason: styles alone repeat every few
# rows and sampling then returns paraphrases instead of distinct tasks.
CODE_DOMAINS = [
    "text processing", "dates and times", "file paths", "simple statistics",
    "sorting and ranking", "dictionaries and lookups", "number formatting",
    "sequences and windows", "parsing configuration", "deduplication",
    "unit conversion", "search and filtering", "simple geometry", "tokenising",
    "grouping and counting", "validation of user input",
]
CODE_TWISTS = [
    "It must handle an empty input gracefully.",
    "It must return a tuple of two values.",
    "It must not use any imports.",
    "It should be case-insensitive where that makes sense.",
    "It must preserve the original order of the input.",
    "It should raise a ValueError on malformed input.",
    "It must work for both integers and floats.",
    "It should ignore entries that are None.",
    "It must run in a single pass over the input.",
    "It should accept an optional keyword argument with a default.",
]


def math_prompts(n):
    """Instructions asking the model to AUTHOR a problem, deterministic order."""
    out = []
    nt, ns = len(MATH_TOPICS), len(MATH_STYLES)
    for i in range(n):
        topic = MATH_TOPICS[i % nt]
        style = MATH_STYLES[(i // nt) % ns]
        twist = MATH_TWISTS[(i // (nt * ns)) % len(MATH_TWISTS)]
        # Ask for the PROBLEM ONLY. The original lane is a prompts file
        # (selfmath_v2_3000.jsonl): it holds model-authored problems, and the trajectory is
        # the model SOLVING them. Asking for problem+solution here would put the solution
        # in the prompt and train on the wrong thing.
        out.append(f"Write {style}, set in the context of {topic}. "
                   f"{twist} "
                   f"Output ONLY the problem statement. Do not solve it, do not show any "
                   f"working, and do not state the answer.")
    return out


def code_prompts(n, seeds):
    out = []
    ns = len(CODE_STYLES)
    for i in range(n):
        style = CODE_STYLES[i % ns]
        domain = CODE_DOMAINS[(i // ns) % len(CODE_DOMAINS)]
        style = f"{style}, in the domain of {domain}"
        twist = CODE_TWISTS[(i // (ns * len(CODE_DOMAINS))) % len(CODE_TWISTS)]
        seed = seeds[i % len(seeds)] if seeds else ""
        s = (f"Here is a fragment of open-source code for inspiration:\n\n"
             f"{seed}\n\n" if seed else "")
        out.append(f"{s}Write {style}. {twist} Output ONLY the task description a "
                   f"programmer would be given, as a short instruction. Do not write "
                   f"the code.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=("math", "code"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data")
    ap.add_argument("--seed-file", default=None,
                    help="jsonl of OSS code snippets for the code lane (Magicoder-style)")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    seeds = []
    if A.lane == "code" and A.seed_file and os.path.exists(A.seed_file):
        seeds = [json.loads(l).get("content", "")[:600] for l in open(A.seed_file)]
        print(f"[selfgen] {len(seeds)} OSS seed snippets", flush=True)

    prompts = math_prompts(A.n) if A.lane == "math" else code_prompts(A.n, seeds)
    path = os.path.join(A.out, f"selfgen_{A.lane}_{A.n}.jsonl")
    with open(path, "w") as fh:
        for i, t in enumerate(prompts):
            fh.write(json.dumps({"idx": i, "lane": f"selfgen_{A.lane}", "text": t}) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"[selfgen] wrote {len(prompts)} {A.lane} prompts -> {path}", flush=True)
    print(f"[selfgen] sha256 {sha}", flush=True)


if __name__ == "__main__":
    main()
