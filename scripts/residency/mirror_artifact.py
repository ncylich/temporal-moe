#!/usr/bin/env python3
"""Mirror a heavy artifact to Hugging Face and record it in results/MANIFEST.csv.

This is the enforcement of the standing rule in RECOVER_DATA_PLAN Part 2: every adapter
and trajectory goes off-pod the moment it is written, not at the end of a program. The
August adaptation program did not do this -- all four ncylich/temporal-moe-* repos were
last written 2026-07-27, three weeks before it ran -- and that single omission is why the
whole recovery plan exists. snapshot_cells.sh makes the same argument for a 56 KB file:
"There is no reason for the record of a program's results to be less durable than the code
that produced them."

Writes the MANIFEST row only after the upload returns, so a row never claims a file that
is not actually on the hub. Re-mirroring the same local_path replaces its row rather than
appending a duplicate.

    mirror_artifact.py --path /workspace/instruct-traj/gemma4_d7.pt --kind trajectory
"""
import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # scripts/residency/x.py -> repo root
from analysis.paths import ROOT                                      # noqa: E402

REPO = "ncylich/temporal-moe-extras"
# where each kind lands inside the repo; keeps the adaptation program's artifacts together
DEST = {"trajectory": "adaptation/instruct-traj", "adapter": "adaptation/adapters",
        "pool": "adaptation/data"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--kind", required=True, choices=sorted(DEST))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    A = ap.parse_args()

    assert os.path.exists(A.path), f"missing: {A.path}"
    digest, size = sha256(A.path), os.path.getsize(A.path)
    hf_path = f"{DEST[A.kind]}/{os.path.basename(A.path)}"
    print(f"[mirror] {A.path} ({size/1e6:.1f} MB, sha256 {digest[:16]}...)\n"
          f"[mirror]   -> {A.repo}:{hf_path}", flush=True)
    if A.dry_run:
        print("[mirror] dry run, nothing uploaded"); return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or open(os.path.expanduser(
        "~/.cache/huggingface/token")).read().strip())
    api.upload_file(path_or_fileobj=A.path, path_in_repo=hf_path, repo_id=A.repo,
                    repo_type="model")
    print("[mirror] upload ok", flush=True)

    # MANIFEST last: a row that names a file not on the hub is worse than no row.
    man = os.path.join(ROOT, "results/MANIFEST.csv")
    local_rel = os.path.relpath(A.path, ROOT) if A.path.startswith(ROOT) else A.path
    with open(man, newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], [r for r in rows[1:] if r and r[0] != local_rel]
    body.append([local_rel, A.repo, hf_path, str(size), digest, "", ""])
    body.sort(key=lambda r: r[0])
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(body)
    print(f"[mirror] MANIFEST.csv updated ({len(body)} rows)", flush=True)


if __name__ == "__main__":
    main()
