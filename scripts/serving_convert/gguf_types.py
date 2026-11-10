#!/usr/bin/env python3
"""Step 4 check: tensor types and sizes in a GGUF, grouped by role.

    python scripts/serving_convert/gguf_types.py --llama-cpp ~/Documents/llama.cpp-android \
        $WORK/fine_1e19-q4_0.gguf
"""
import argparse, collections, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--llama-cpp", required=True)
ap.add_argument("gguf")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.expanduser(a.llama_cpp), "gguf-py"))
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(a.gguf)
groups = collections.defaultdict(lambda: collections.Counter())
size = collections.Counter()
for t in r.tensors:
    role = re.sub(r"^blk\.\d+\.", "blk.N.", t.name)
    groups[role][t.tensor_type.name] += 1
    size[role] += int(t.n_bytes)
print(f"{a.gguf}: {os.path.getsize(a.gguf)} bytes ({os.path.getsize(a.gguf)/1e9:.3f} GB), {len(r.tensors)} tensors")
for role in sorted(groups, key=lambda k: -size[k]):
    types = ", ".join(f"{k} x{v}" for k, v in groups[role].items())
    print(f"  {role:34s} {types:16s} {size[role]/1e6:9.1f} MB")
exps = {t.name: t.tensor_type.name for t in r.tensors if re.search(r"ffn_(gate|up|down)_exps", t.name)}
bad = {k: v for k, v in exps.items() if v != "Q4_0"}
print(f"routed expert tensors: {len(exps)}, not Q4_0: {len(bad)} {list(bad.items())[:3]}")
