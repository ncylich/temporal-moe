#!/usr/bin/env python3
"""Compare two MBPP dumps item by item (same problems, same seed): pass agreement, generation
length, and the first token index at which the two generations diverge. Used for the fast
presence-penalty processor against vLLM's native one.

    mbpp_compare_dumps.py A.json B.json
"""
import json
import sys


def load(p):
    d = json.load(open(p)); items = d["items"] if isinstance(d, dict) else d
    return {it["doc"]: it for it in items}


a, b = load(sys.argv[1]), load(sys.argv[2])
docs = sorted(set(a) & set(b))
same_pass = ident = 0; div = []; pa = pb = 0
for doc in docs:
    x, y = a[doc], b[doc]
    ia = x["gen_ids"] if isinstance(x["gen_ids"], list) else json.loads(x["gen_ids"])
    ib = y["gen_ids"] if isinstance(y["gen_ids"], list) else json.loads(y["gen_ids"])
    pa += str(x["pass"]) == "True"; pb += str(y["pass"]) == "True"
    same_pass += (str(x["pass"]) == str(y["pass"]))
    if ia == ib:
        ident += 1
    else:
        k = next((i for i, (u, v) in enumerate(zip(ia, ib)) if u != v), min(len(ia), len(ib)))
        div.append((doc, k, len(ia), len(ib)))
print(f"{len(docs)} shared problems: pass A {pa} B {pb}, same pass/fail on {same_pass}, "
      f"identical token sequences {ident}")
if div:
    ks = sorted(k for _, k, _, _ in div)
    print(f"divergent {len(div)}: first-divergence token index median {ks[len(ks) // 2]}, min {ks[0]}, max {ks[-1]}")
    for doc, k, la, lb in div[:8]:
        print(f"  {doc}: diverge at {k} (lengths {la} / {lb})")
