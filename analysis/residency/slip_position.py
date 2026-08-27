#!/usr/bin/env python3
"""Where in a generation does the arithmetic slip happen, and how hard was the arithmetic?

Why. Under the constraint, teacher-forced CE on digit tokens is LOW (digit-weight run: 0.40
vs 0.62 unweighted), i.e. given the prefix the model predicts digits well. Yet 84%/66% of
constrained failures contain a false equation. If the slips cluster late in the generation
(after many expert swaps) and on trivial operands, the failure is residency-STATE drift that
only appears on the model's own prefixes -- which prefix-independent CE weighting cannot
reach and on-policy distillation can. If they are uniform and on hard operands, it is a
capacity problem CE can address.

Reads failure_filter.py's <model>_<category>.jsonl. Pure stdlib.
    slip_position.py --model qwen35
"""
import argparse, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"   # real thousands grouping only: "1, 2, 3, ..." lists must not chain
OP = r"\s*(?:[-+*/x×÷]|\\times|\\cdot)\s*"
EQ = re.compile(rf"({NUM}(?:{OP}{NUM})+)\s*=\s*({NUM})")
TOKS = re.compile(rf"{NUM}|[-+*/x×÷]|\\times|\\cdot")


def evaluate(lhs):
    """Left-to-right * and / first then + and -, i.e. normal precedence on a flat chain."""
    t = TOKS.findall(lhs)
    vals, ops = [float(t[0].replace(",", ""))], []
    for i in range(1, len(t) - 1, 2):
        op, v = t[i], float(t[i + 1].replace(",", ""))
        op = {"x": "*", "×": "*", "\\times": "*", "\\cdot": "*", "÷": "/"}.get(op, op)
        if op in "*/":
            vals[-1] = vals[-1] * v if op == "*" else (vals[-1] / v if v else float("nan"))
        else:
            vals.append(v if op == "+" else -v)
    return sum(vals), [abs(float(x.replace(",", ""))) for x in TOKS.findall(lhs) if x[0].isdigit() or x[0] == "-"]


def equations(raw):
    """[(word_frac, is_false, max_operand)] for every a op b [op c] = r in the text."""
    raw = re.sub(r"\\text\{[^}]*\}", " ", raw)               # 216 \text{ sq ft} \times 58
    raw = re.sub(r"\\[,;! ]", "", raw).replace("\\$", "")     # \$12 + \$2, 12\,528
    raw = raw.replace("$", "").replace("*", "").replace("\\(", "").replace("\\)", "")
    out = []
    nwords = max(1, len(raw.split()))
    for m in EQ.finditer(raw):
        try:
            lhs, operands = evaluate(m.group(1)); rhs = float(m.group(2).replace(",", ""))
        except (ValueError, ZeroDivisionError):
            continue
        if lhs != lhs: continue
        false = abs(lhs - rhs) > 0.011 * max(1.0, abs(rhs))
        frac = len(raw[: m.start()].split()) / nwords
        out.append((frac, false, max(operands) if operands else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("qwen35", "gemma4"))
    ap.add_argument("--dir", default=f"{ABLATIONS}/failure_analysis", help="failure_filter.py --out")
    A = ap.parse_args()
    D = json.load(open(f"{A.dir}/{A.model}_categories.json"))
    T = D["tight"]
    print(f"{A.model} tight={T}. Position = fraction of words before the equation (0=start, 1=end).\n")
    hdr = f"{'category / arm':<30}{'gens':>5}{'w/false':>8}{'eqs':>6}{'false':>6}{'medpos all':>11}{'medpos 1stF':>12}{'F late>0.5%':>12}{'all late%':>10}{'triv(<=20)%':>12}{'triv all%':>10}"
    print(hdr)
    for cat in ("damage_unfixed", "adapter_broke", "damage_fixed", "always_right"):
        rows = [json.loads(l) for l in open(f"{A.dir}/{A.model}_{cat}.jsonl")]
        for arm in ("base_free", f"base_{T}", f"adapted_{T}"):
            allpos, fpos, ftriv, atriv, nfalse, neq, wfalse = [], [], [], [], 0, 0, 0
            for r in rows:
                if r[arm]["correct"] and cat != "always_right": continue   # only the WRONG generations
                eqs = equations(r[arm]["raw"])
                neq += len(eqs); allpos += [e[0] for e in eqs]; atriv += [e[2] <= 20 for e in eqs]
                f = [e for e in eqs if e[1]]
                nfalse += len(f)
                if f:
                    wfalse += 1; fpos.append(f[0][0]); ftriv.append(f[0][2] <= 20)
            if not allpos: continue
            print(f"{cat+' / '+arm:<30}{len(rows):>5}{wfalse:>8}{neq:>6}{nfalse:>6}"
                  f"{statistics.median(allpos):>11.2f}{(statistics.median(fpos) if fpos else float('nan')):>12.2f}"
                  f"{(100*sum(p>0.5 for p in fpos)/len(fpos) if fpos else float('nan')):>12.0f}"
                  f"{100*sum(p>0.5 for p in allpos)/len(allpos):>10.0f}"
                  f"{(100*sum(ftriv)/len(ftriv) if ftriv else float('nan')):>12.0f}{100*sum(atriv)/len(atriv):>10.0f}")
        print()
    print("PER-ARM RATES, all docs (false-eq rate = arithmetic damage; scorer FN = points lost to extraction)")
    rates(D)


BOLD = re.compile(r"\*\*([^*]{1,300})\*\*")          # bounded: unclosed ** before a 2k-char digit run was cubic
_BNUM = re.compile(r"-?\d[\d,]*\.?\d*")
LMEVAL = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")   # gsm8k_cot_zeroshot flexible-extract, group_select -1


def _norm(x):
    return x.replace("$", "").replace(",", "").rstrip(".").strip()


def rates(D):
    """Per arm over all docs: false-equation rate (scoring-independent arithmetic damage) and
    scorer false negatives (scored wrong, but the last **bold** number is the gold answer and
    lm-eval's last-number extraction took something else, e.g. '**$132** after 12 hours')."""
    T = D["tight"]; n = len(D["rows"])
    print(f"{'arm':<14}{'eqs':>7}{'false':>7}{'rate%':>7}{'gens w/false%':>15}   {'wrong':>6}{'scorer FN':>10}{'pts':>6}")
    arms = list(next(iter(D["rows"].values()))["arms"])
    for arm in sorted(arms, key=lambda a: (a.split("_")[0] != "base", a.split("_")[1] != "free", a)):
        ne = nf = wf = wrong = fn = 0
        for r in D["rows"].values():
            a = r["arms"][arm]; eqs = equations(a["raw"]); f = sum(e[1] for e in eqs)
            ne += len(eqs); nf += f; wf += f > 0
            if a["correct"]: continue
            wrong += 1; bolds = [_norm(m.group(0)) for span in BOLD.findall(a["raw"]) for m in [_BNUM.search(span)] if m]
            m = LMEVAL.findall(a["raw"]); last = _norm(m[-1][0] or m[-1][1]) if m else None
            fn += bool(bolds) and bolds[-1] == r["gold"] and last != r["gold"]
        print(f"{arm:<14}{ne:>7}{nf:>7}{100*nf/ne:>7.2f}{100*wf/n:>15.1f}   {wrong:>6}{fn:>10}{100*fn/n:>6.1f}")


if __name__ == "__main__":
    main()
