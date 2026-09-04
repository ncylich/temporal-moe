#!/usr/bin/env python3
"""Re-score an MBPP dump offline from its stored raw generations under two extraction rules:
whole last fenced block (the rule the recorded mbpp_gemma rows used) and function-only (the
model's `if __name__` self-test scaffold and trailing top-level asserts dropped). No model runs.

    mbpp_rescore.py <dump.json> --arch gemma4 [--limit N]
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbpp_chat import strip_thinking, FENCE, function_only  # noqa: E402


def score(preds, tests):
    p = f"/tmp/mbpp_rescore_{os.getpid()}.json"
    json.dump({"preds": [[c] for c in preds], "tests": tests}, open(p, "w"))
    out = subprocess.run(["/workspace/venv_fla/bin/python", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "heg_scorer.py"), p], capture_output=True, text=True)
    os.remove(p)
    items = [l for l in out.stdout.splitlines() if l.startswith("ITEMS")][0].split()[1]
    return [c == "1" for c in items]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--arch", required=True)
    ap.add_argument("--limit", type=int, default=None)
    A = ap.parse_args()
    from datasets import load_dataset
    probs = {f"mbpp/{p['task_id']}": p for p in load_dataset("google-research-datasets/mbpp", "full", split="test")}
    d = json.load(open(A.dump)); items = d["items"] if isinstance(d, dict) else d
    if A.limit:
        items = items[: A.limit]
    raw_codes, fn_codes, tests = [], [], []
    for it in items:
        p = probs[it["doc"]]
        vis, unfinished = strip_thinking(A.arch, it["raw"])
        blocks = FENCE.findall(vis)
        code = "" if unfinished else (blocks[-1] if blocks else vis)
        raw_codes.append(code); fn_codes.append(function_only(code) if code else "")
        tests.append((p.get("test_setup_code") or "") + "\n" + "\n".join(p["test_list"]))
    a = score(raw_codes, tests); b = score(fn_codes, tests)
    flipped = [it["doc"] for it, x, y in zip(items, a, b) if x != y]
    stored = [str(it.get("pass")) == "True" for it in items]
    print(f"{os.path.basename(A.dump)}: n={len(items)} whole-block pass {sum(a)/len(a):.4f} (stored {sum(stored)/len(items):.4f}) "
          f"function-only pass {sum(b)/len(b):.4f} | flipped {len(flipped)} (fail->pass {sum(1 for x,y in zip(a,b) if not x and y)}, pass->fail {sum(1 for x,y in zip(a,b) if x and not y)})")
    if flipped: print("   flipped:", " ".join(flipped[:12]))


if __name__ == "__main__":
    main()
