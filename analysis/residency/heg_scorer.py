#!/usr/bin/env python3
"""Direct HumanEval scorer: exec each prediction+test in a fresh subprocess with a
timeout (the sandboxing code_eval performs, without its fork-under-filelock issue).
Reads the preds file named by argv[1] (default /tmp/heg_preds.json)
{preds: [[code]], tests: [test]}, prints 'PASS1 <frac>'. The path is an argument so two
producers can score concurrently without clobbering each other's file."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/heg_preds.json"))
def run_one(args):
    pred, test = args
    try:
        r = subprocess.run([sys.executable, "-c", pred + "\n\n" + test],
                           capture_output=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False
pairs = [(p[0], t) for p, t in zip(d["preds"], d["tests"])]
with ThreadPoolExecutor(max_workers=16) as ex:
    results = list(ex.map(run_one, pairs))
print("PASS1", sum(results) / len(results))
print("ITEMS", "".join("1" if r else "0" for r in results))
