#!/usr/bin/env python3
"""Have Claude Haiku classify WHY each constrained generation went wrong.

Input: the per-category jsonl files written by failure_filter.py (question, gold, and every
arm's generation). For each problem, Haiku sees the FREE-routing generation (correct) next
to the CONSTRAINED one (wrong) and names the first point of divergence and its kind.

Classes (fixed vocabulary so counts aggregate):
  arithmetic_slip   right plan, right operands, wrong result of a basic operation
  wrong_plan        sets up the wrong computation (wrong operation, missed a step)
  misread           misreads a quantity or condition in the problem
  unit_error        unit/conversion mistake (minutes vs hours, cents vs dollars)
  format_only       the right number is there; presentation defeats the scorer ($88.00)
  incomplete        stops before finishing
  other

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile). Without one this script
does nothing useful; failure_filter.py's mechanical stats and a hand-read sample are the
fallback and were used first (REBUILD_RESULTS.md, 2026-08-27).

    haiku_failure_analysis.py --model qwen35 --category damage_unfixed [--n 60]
"""
import argparse
import collections
import json
import os
import sys

import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

CLASSES = ["arithmetic_slip", "wrong_plan", "misread", "unit_error", "format_only",
           "incomplete", "other"]
SYSTEM = (
    "You are auditing a math model that runs under a memory constraint. You get one GSM8K "
    "problem, its gold answer, the model's UNCONSTRAINED solution (correct), and its "
    "CONSTRAINED solution (wrong). Find the FIRST step where the constrained solution "
    "diverges from a correct computation and classify it. Be literal: if the plan is right "
    "and a single addition or multiplication has the wrong result, that is arithmetic_slip "
    "even if later steps compound it.\n\n"
    "Reply with ONE JSON object and nothing else: "
    '{"class": <one of ' + "|".join(CLASSES) + '>, '
    '"first_wrong_step": "<quote the wrong step verbatim, max 120 chars>", '
    '"should_be": "<the correct value or step, max 60 chars>", '
    '"note": "<one sentence, max 160 chars>"}'
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("qwen35", "gemma4"))
    ap.add_argument("--category", default="damage_unfixed")
    ap.add_argument("--arm", default=None, help="constrained arm key; default adapted_<tight>")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--haiku", default="claude-haiku-4-5")
    A = ap.parse_args()

    D = json.load(open(f"{ABLATIONS}/failure_analysis/{A.model}_categories.json"))
    arm = A.arm or f"adapted_{D['tight']}"
    rows = [json.loads(l) for l in open(f"{ABLATIONS}/failure_analysis/{A.model}_{A.category}.jsonl")]
    rows = [r for r in rows if r["base_free"]["correct"] and not r[arm]["correct"]][: A.n]
    print(f"[haiku] {A.model} {A.category} arm={arm}: {len(rows)} problems", flush=True)

    client = anthropic.Anthropic()
    out = []
    for i, r in enumerate(rows):
        user = (f"PROBLEM:\n{r['question']}\n\nGOLD ANSWER: {r['gold']}\n\n"
                f"UNCONSTRAINED (correct):\n{r['base_free']['raw']}\n\n"
                f"CONSTRAINED (wrong):\n{r[arm]['raw']}")
        try:
            resp = client.messages.create(
                model=A.haiku, max_tokens=400, system=SYSTEM,
                messages=[{"role": "user", "content": user}])
            txt = next(b.text for b in resp.content if b.type == "text").strip()
            txt = txt[txt.find("{"): txt.rfind("}") + 1]
            j = json.loads(txt)
        except anthropic.APIStatusError as e:
            print(f"  doc {r['doc_id']}: API error {e.status_code}", flush=True); continue
        except (json.JSONDecodeError, StopIteration, ValueError) as e:
            print(f"  doc {r['doc_id']}: unparseable reply ({e})", flush=True); continue
        j["doc_id"] = r["doc_id"]
        out.append(j)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    outp = f"{ABLATIONS}/failure_analysis/{A.model}_{A.category}_{arm}_haiku.jsonl"
    with open(outp, "w") as f:
        for j in out:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    c = collections.Counter(j.get("class", "other") for j in out)
    print(f"\n{A.model} {A.category} ({arm}), n={len(out)}:")
    for k in CLASSES:
        if c[k]:
            print(f"  {k:<17}{c[k]:>4}  {100*c[k]/len(out):5.1f}%")
    print(f"\nexamples of the top class:")
    top = c.most_common(1)[0][0] if c else None
    for j in [x for x in out if x.get("class") == top][:5]:
        print(f"  doc {j['doc_id']}: {j.get('first_wrong_step','')!r} -> should be {j.get('should_be','')!r}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
