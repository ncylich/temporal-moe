"""Path contract for the OLMoE adaptation artifacts.

The base checkpoint (27 GB) and the adaptation corpus (4.4 GB) are far too large to live in the
repo, so they sit in a working directory outside it. Resolution order, highest first:

    $TMOE_OLMOE_MODEL / $TMOE_OLMOE_DATA   point at the two directories individually
    $TMOE_OLMOE_HOME                       a working dir holding model/ and data/
    <repo parent>/olmoe-adapt              the layout the adaptation program used

Nothing here is a hardcoded absolute path: the last fallback is derived from `analysis.paths.ROOT`.

The corpus files carry sha256 in results/MANIFEST.csv. `verify_corpus()` checks a local copy
against it, which is what licenses using local disk as a cache instead of re-downloading from
Hugging Face. See scripts/artifacts.py to fetch them if they are absent.
"""

import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT  # noqa: E402

__all__ = ["OLMOE_HOME", "MODEL_DIR", "DATA_DIR", "MANIFEST", "verify_corpus"]

OLMOE_HOME = os.environ.get("TMOE_OLMOE_HOME") or os.path.join(os.path.dirname(ROOT), "olmoe-adapt")
MODEL_DIR = os.environ.get("TMOE_OLMOE_MODEL") or os.path.join(OLMOE_HOME, "model")
DATA_DIR = os.environ.get("TMOE_OLMOE_DATA") or os.path.join(OLMOE_HOME, "data")
MANIFEST = os.path.join(ROOT, "results", "MANIFEST.csv")


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_corpus(names=("finetune_ids.pt", "bpb_slice_ids.pt", "bpb_slice_meta.json",
                         "finetune_meta.json"), full=True):
    """Check local corpus files against results/MANIFEST.csv.

    Returns {name: (ok, detail)}. `full=False` checks size only, which is quick; `full=True`
    hashes, which is the check that actually establishes the file is the published artifact.
    """
    want = {}
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            base = os.path.basename(row["local_path"])
            if base in names and "olmoe-adapt/data" in row["local_path"]:
                want[base] = (int(row["bytes"]), row["sha256"])
    out = {}
    for name in names:
        path = os.path.join(DATA_DIR, name)
        if name not in want:
            out[name] = (False, "not listed in MANIFEST.csv")
        elif not os.path.exists(path):
            out[name] = (False, f"absent at {path}")
        else:
            n_want, sha_want = want[name]
            n_got = os.path.getsize(path)
            if n_got != n_want:
                out[name] = (False, f"size {n_got} != manifest {n_want}")
            elif not full:
                out[name] = (True, f"size {n_got} ok (hash not checked)")
            else:
                sha_got = _sha256(path)
                out[name] = ((sha_got == sha_want),
                             "sha256 ok" if sha_got == sha_want else f"sha256 {sha_got[:16]} != {sha_want[:16]}")
    return out


if __name__ == "__main__":
    print(f"OLMOE_HOME  {OLMOE_HOME}")
    print(f"MODEL_DIR   {MODEL_DIR}   exists={os.path.isdir(MODEL_DIR)}")
    print(f"DATA_DIR    {DATA_DIR}   exists={os.path.isdir(DATA_DIR)}")
    quick = "--full" not in sys.argv
    for name, (ok, detail) in verify_corpus(full=not quick).items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: {detail}")
