#!/usr/bin/env python3
"""Mirror a phase0 run's final checkpoint to ncylich/temporal-moe-ckpts and record every file in
results/MANIFEST.csv, the same layout the July uploads used (hf_path = local path relative to
results/phase0/runs). Uploads the final iter_* directory, latest_checkpointed_iteration.txt,
run.meta and train.log. MANIFEST rows are written only after the upload returns, and a re-mirror
replaces the run's existing rows instead of duplicating them.

    mirror_checkpoint.py --run moe_fine_g3_1e19 [--run ...] [--dry-run]
"""
import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.paths import ROOT, RUNS  # noqa: E402

REPO = "ncylich/temporal-moe-ckpts"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_for(run):
    d = os.path.join(RUNS, run)
    it = open(os.path.join(d, "ckpt/latest_checkpointed_iteration.txt")).read().strip()
    ck = os.path.join(d, "ckpt", f"iter_{int(it):07d}")
    out = [os.path.join(ck, f) for f in sorted(os.listdir(ck))]
    out.append(os.path.join(d, "ckpt/latest_checkpointed_iteration.txt"))
    for extra in ("run.meta", "train.log"):
        if os.path.exists(os.path.join(d, extra)):
            out.append(os.path.join(d, extra))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--dry-run", action="store_true")
    A = ap.parse_args()
    man = os.path.join(ROOT, "results/MANIFEST.csv")
    with open(man, newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    api = None
    if not A.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or open(os.path.expanduser(
            "~/.cache/huggingface/token")).read().strip())
    for run in A.run:
        files = files_for(run)
        total = sum(os.path.getsize(f) for f in files)
        print(f"[mirror] {run}: {len(files)} files, {total/1e9:.2f} GB", flush=True)
        new_rows = []
        for f in files:
            rel = os.path.relpath(f, ROOT)
            hf_path = os.path.relpath(f, RUNS)
            if not A.dry_run:
                api.upload_file(path_or_fileobj=f, path_in_repo=hf_path, repo_id=REPO,
                                repo_type="model")
                print(f"[mirror]   up {hf_path} ({os.path.getsize(f)/1e6:.1f} MB)", flush=True)
            new_rows.append([rel, REPO, hf_path, str(os.path.getsize(f)), sha256(f), run, ""])
        if A.dry_run:
            continue
        keep = [r for r in body if not (r and r[0] in {n[0] for n in new_rows})]
        body = keep + new_rows
        body.sort(key=lambda r: r[0])
        with open(man, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(body)
        print(f"[mirror] {run}: MANIFEST updated ({len(new_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
