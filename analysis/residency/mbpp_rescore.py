#!/usr/bin/env python3
"""Re-score an MBPP dump offline from its stored raw generations under three extraction rules:
whole last fenced block (the scored rule, the one the recorded mbpp_gemma rows used), def-block
(the last fenced block that defines a function, for models that append a second block holding
only the prompt's asserts), and function-only (the model's `if __name__` self-test scaffold and
trailing top-level asserts dropped from the def-block). Diagnostic only, no model runs.

    mbpp_rescore.py <dump.json> --arch gemma4 [--limit N] [--csv out.csv]
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
    ap.add_argument("--csv", default=None, help="append a row (record_arm,n,whole_block,stored,def_block,function_only,rescued_def,rescued_fn)")
    A = ap.parse_args()
    from datasets import load_dataset
    probs = {f"mbpp/{p['task_id']}": p for p in load_dataset("google-research-datasets/mbpp", "full", split="test")}
    d = json.load(open(A.dump)); items = d["items"] if isinstance(d, dict) else d
    if A.limit:
        items = items[: A.limit]
    raw_codes, def_codes, fn_codes, tests = [], [], [], []
    for it in items:
        p = probs[it["doc"]]
        vis, unfinished = strip_thinking(A.arch, it["raw"])
        blocks = FENCE.findall(vis)
        code = "" if unfinished else (blocks[-1] if blocks else vis)
        defs = [b for b in blocks if "def " in b]
        dcode = "" if unfinished else (defs[-1] if defs else code)
        raw_codes.append(code); def_codes.append(dcode); fn_codes.append(function_only(dcode) if dcode else "")
        tests.append((p.get("test_setup_code") or "") + "\n" + "\n".join(p["test_list"]))
    a = score(raw_codes, tests); d = score(def_codes, tests); b = score(fn_codes, tests)
    stored = [str(it.get("pass")) == "True" for it in items]
    n = len(items)
    rd = sum(1 for x, y in zip(a, d) if not x and y); rf = sum(1 for x, y in zip(a, b) if not x and y)
    print(f"{os.path.basename(A.dump)}: n={n} whole-block pass {sum(a)/n:.4f} (stored {sum(stored)/n:.4f}) "
          f"def-block {sum(d)/n:.4f} (rescued {rd}) function-only {sum(b)/n:.4f} (rescued {rf}, "
          f"pass->fail {sum(1 for x,y in zip(a,b) if x and not y)})")
    fl = [it["doc"] for it, x, y in zip(items, a, b) if x != y]
    if fl: print("   flipped:", " ".join(fl[:12]))
    if A.csv:
        name = os.path.basename(A.dump).replace("_mbpp_chat.json", "").replace("_mbpp_gemma.json", "")
        open(A.csv, "a").write(f"{name},{n},{sum(a)/n:.4f},{sum(stored)/n:.4f},{sum(d)/n:.4f},{sum(b)/n:.4f},{rd},{rf}\n")


if __name__ == "__main__":
    main()
