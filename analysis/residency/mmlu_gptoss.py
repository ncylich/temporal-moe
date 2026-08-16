#!/usr/bin/env python3
"""MMLU for gpt-oss with harmony-tolerant answer extraction.

The stock mmlu_flan_cot_fewshot get-answer filter only accepts "The answer is (X)";
gpt-oss answers in its final channel as e.g. "**Answer: (B) 4**", flooring every cell
regardless of correctness (probe: reasoning and letter correct, extraction 0). Here the
suite path is unchanged (chat template, greedy, final-channel filter, stateful rule on
generated tokens) and only the extraction differs: last "(A)-(D)"-style letter, with
"The answer is (X)" still matched first. Rows append to instruct_genbench_vllm.csv with
task mmlu_gptoss_relaxed, applied identically to every arm.

    mmlu_gptoss.py --model gptoss_120b --arms free,R4,R16
"""
import argparse
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402
from instruct_selfce import MODELS                                   # noqa: E402
import vllm_glue                                                     # noqa: E402
import vllm_residency  # noqa: F401,E402
from decode_state import DEC                                         # noqa: E402

STRICT = re.compile(r"[Tt]he answer is \(?([A-D])\)?")
RELAXED = re.compile(r"\(([A-D])\)|\*\*?\s*[Aa]nswer\s*[:\-]?\s*\(?([A-D])\)?")


def extract(text):
    m = STRICT.search(text)
    if m:
        return m.group(1)
    hits = RELAXED.findall(text)
    if hits:
        a, b = hits[-1]
        return a or b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--limit", type=int, default=4, help="items per subject")

    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--gen-cap", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=("low", "medium", "high"))
    ap.add_argument("--record-as", default=None)
    ap.add_argument("--path", default=None, help="override checkpoint dir (merged adapters)")
    ap.add_argument("--csv-name", default="instruct_genbench_vllm.csv",
                    help="diagnostics use screening_genbench.csv")
    A = ap.parse_args()
    M = MODELS[A.model]
    if A.path:
        M = dict(M, path=A.path)

    vllm_glue.install()
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    lm = VLLM(pretrained=M["path"], batch_size="auto", max_gen_toks=A.gen_cap,
              max_model_len=5632, gpu_memory_utilization=A.gpu_mem,
              enforce_eager=True, enable_prefix_caching=False, dtype="auto")

    if A.reasoning_effort:
        _tk = lm.tokenizer
        _orig_act = _tk.apply_chat_template
        _tk.apply_chat_template = lambda *aa, **kk: _orig_act(
            *aa, **{**kk, "reasoning_effort": A.reasoning_effort})

    import genprotocol
    genprotocol.install(lm, cap=A.gen_cap,
                        think_marker="<|channel|>final<|message|>")

    out = os.path.join(ABLATIONS, A.csv_name)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    for arm in A.arms.split(","):
        R = None if arm == "free" else int(arm.lstrip("R"))
        DEC.update(on=R is not None, R=R or 0, swaps=1)
        DEC["state"].clear()
        t0 = time.time()
        res = simple_evaluate(model=lm, tasks=["mmlu_flan_cot_fewshot"], limit=A.limit,
                              apply_chat_template=True,
                              gen_kwargs="do_sample=True,temperature=1.0,top_p=1.0,"
                                         "seed=1234,skip_special_tokens=False",
                              log_samples=True)
        n = hit = miss_extract = 0
        for task, samp in res["samples"].items():
            for x in samp:
                resp = x["resps"][0][0] if x.get("resps") else ""
                gold = re.search(r"\(([A-D])\)", str(x.get("target", "")))
                pred = extract(resp)
                if gold is None:
                    continue
                n += 1
                miss_extract += pred is None
                hit += pred == gold.group(1)
        acc = hit / max(1, n)
        secs = time.time() - t0
        print(f"  [{A.model}] {arm} mmlu_gptoss_relaxed: acc={acc:.4f} "
              f"({n} items, {miss_extract} unextracted, {secs:.0f}s)", flush=True)
        w.writerow([A.record_as or A.model, M["E"], M["k"], arm, R or "", "mmlu_gptoss_relaxed",
                    "acc,relaxed-extract", f"{acc:.6f}", A.limit, A.gen_cap,
                    f"{secs:.0f}"])
        fh.flush()
    fh.close()
    print(f"MMLU-GPTOSS {A.model} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
