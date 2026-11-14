#!/usr/bin/env python3
"""Answer-only rescoring for models whose thinking text sat in the judged output.

gpt-oss was always scored on its final channel; gemma/qwen/LFM think-in-text responses
were judged raw, so format-sensitive metrics (IFEval, MMLU get-answer) punished the
thinking mode artifactually. This rescores the SAME generations (token dumps ->
decoded -> think segment stripped) offline: no GPU, no regeneration. Rows append to
instruct_genbench_vllm.csv with metric suffix ",answer-only".

Think-segment conventions: gemma "<|channel>...<channel|>" (answer = after last close);
qwen (prompt-opened) and LFM answer = after first "</think>". Truncated-in-think -> "".

    rescore_answer_only.py            # all affected records
"""
import csv
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

TD = "/workspace/instruct-traj/genbench_tokens"

RECORDS = {  # record -> (tokenizer path, strip mode, arms)
    "gemma4_think_on": ("/dev/shm/gemma4-26b-it", "gemma", ["free", "R8", "R16"]),
    "qwen35_instruct": ("/dev/shm/qwen35-35b-a3b-instruct", "think", ["free", "R8", "R32"]),
    "lfm25_instruct": ("/workspace/instruct-models/lfm25-8b-a1b", "think", ["free", "R4"]),
}
FALLBACK_TOK = {  # weights cycle off shm; tokenizer from HF cache by repo id
    "gemma4_think_on": "google/gemma-4-26B-A4B-it",
    "qwen35_instruct": "Qwen/Qwen3.5-35B-A3B",
    "lfm25_instruct": "LiquidAI/LFM2.5-8B-A1B",
}


def strip_answer(text, mode):
    if mode == "gemma":
        if "<|channel>" not in text:
            return text
        return text.rsplit("<channel|>", 1)[1] if "<channel|>" in text else ""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    if "<think>" in text:                          # opened, never closed: truncated
        return ""
    return text


def score_gsm8k(resp, doc):
    gold = doc["answer"].split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?[\d,]*\.?\d+", resp.replace(",", ""))
    return float(bool(nums) and nums[-1].rstrip(".") == gold)


def score_mmlu(resp, gold_letter):
    m = re.search(r"[Tt]he answer is \(?([A-D])\)?", resp)
    return float(bool(m) and m.group(1) == gold_letter)


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    sys.path.insert(0, "/opt/venv_vllm/lib/python3.11/site-packages")
    from lm_eval.tasks.ifeval import utils as ifu

    gsm_docs = list(load_dataset("openai/gsm8k", "main", split="test"))
    ifeval_docs = list(load_dataset("google/IFEval", split="train"))

    out = os.path.join(ABLATIONS, "instruct_genbench_vllm.csv")
    fh = open(out, "a", newline="")
    w = csv.writer(fh)

    for record, (tp, mode, arms) in RECORDS.items():
        try:
            tok = AutoTokenizer.from_pretrained(tp)
        except (OSError, ValueError):
            tok = AutoTokenizer.from_pretrained(FALLBACK_TOK[record])
        # E/k from any existing row of this record
        Ek = None
        for r in csv.reader(open(out)):
            if r and r[0] == record and len(r) > 3:
                Ek = (r[1], r[2])
                break
        assert Ek, f"no existing rows for {record}"
        for arm in arms:
            R = "" if arm == "free" else arm.lstrip("R")
            for task, scorer in (("gsm8k_cot_zeroshot", "gsm"),
                                 ("ifeval", "ifeval"),
                                 ("mmlu_flan_cot_fewshot", "mmlu")):
                p = os.path.join(TD, f"{record}_{arm}_{task}.pt")
                if not os.path.exists(p):
                    print(f"  [skip] no dump: {record} {arm} {task}")
                    continue
                items = torch.load(p, weights_only=False)["items"]
                seen = set()
                items = [i for i in items
                         if not (i["doc_id"] in seen or seen.add(i["doc_id"]))]
                vals = []
                if scorer == "gsm":
                    for i in items:
                        resp = strip_answer(tok.decode(i["ids"]), mode)
                        vals.append(score_gsm8k(resp, gsm_docs[i["doc_id"]]))
                    metric = "exact_match,answer-only"
                elif scorer == "ifeval":
                    for i in items:
                        resp = strip_answer(tok.decode(i["ids"]), mode)
                        r = ifu.process_results(ifeval_docs[i["doc_id"]], [resp])
                        vals.append(float(r["prompt_level_strict_acc"]))
                    metric = "prompt_level_strict_acc,answer-only"
                else:
                    # mmlu dumps are per-subject concatenations in generation order;
                    # doc_id alignment to subjects is not recoverable -> rescore only
                    # extraction-independent? No: skip gold-dependent mmlu here.
                    print(f"  [skip] mmlu rescoring needs per-doc gold: {record} {arm}")
                    continue
                acc = sum(vals) / len(vals)
                w.writerow([record, Ek[0], Ek[1], arm, R, task, metric,
                            f"{acc:.6f}", len(vals), "rescored", 0])
                fh.flush()
                print(f"  {record:16s} {arm:4s} {task:22s} answer-only = {acc:.4f} "
                      f"(n={len(vals)})")
    fh.close()


if __name__ == "__main__":
    main()
