#!/usr/bin/env python3
"""Fetch the published artifacts named in results/MANIFEST.csv into the layout the analysis
scripts already expect.

The repository holds code and result tables. Everything heavy, checkpoints, router traces,
tokenized corpus, lives in four Hugging Face repositories, and MANIFEST.csv is the only thing that
maps one to the other. This connects them.

    scripts/artifacts.py pull --cited --repo extras --dry-run
    scripts/artifacts.py pull --run g3_tmoe_s2_1e17
    scripts/artifacts.py pull --glob 'ablations/*.csv'
    scripts/artifacts.py verify --repo extras

Destination root is resolved through the same contract as everything else: $TMOE_ROOT, then
analysis/paths.py (git, then file location). Nothing here writes to MANIFEST.csv.

Downloads use only the standard library, so this runs under `scripts/setup.sh analysis` with no
torch, no CUDA, and no huggingface_hub. The published repositories are public; no token is needed.
"""

import argparse
import concurrent.futures
import csv
import fnmatch
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
# The corpus is a dataset repo; the other three are model repos. Dataset URLs carry /datasets/.
DATASET_REPOS = {"ncylich/temporal-moe-corpus"}

# hf_path's first segment -> directory relative to the destination root.
# Keyed on (repo short name, first segment). "*" matches any first segment.
LAYOUT = {
    # Every ckpts path is already <run>/... , _batch_logs/... or _lmeval_scratch/... .
    ("ckpts", "*"): "results/phase0/runs",
    ("extras", "ablations"): "results",
    ("extras", "figures"): "results/phase0",
    ("extras", "run_captures"): "results/phase0/runs",
    ("extras", "tokenizer_tok16k"): "data/_tok16k_from_extras",
    ("extras", "olmoe_adapt"): "artifacts/olmoe-adapt",
    ("extras", "merged_ce_model"): "artifacts/olmoe-adapt",
    ("router-adapt", "adapt_ckpts"): "results/archive/olmoe_wrong_renorm",
    ("router-adapt", "olmoe_adapt"): "artifacts/olmoe-adapt/data",
    ("router-adapt", "metadata"): "artifacts/olmoe-adapt/metadata",
    ("corpus", "dclm_tokenized"): "data",
    ("corpus", "tok16k_full"): "data",
    ("corpus", "tokenizer"): "data/tok16k_from_corpus",
}
# Repo-level docs and checksum sidecars: keep them, but out of the analysis tree.
SIDECAR = {"README.md", ".gitattributes", "parquet_sha256.txt", "jsonl_sha256.txt"}


def repo_root():
    env = os.environ.get("TMOE_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "analysis"))
    try:
        from paths import ROOT  # canonical resolver
        return ROOT
    except Exception:
        return os.path.dirname(here)


def short(repo):
    return repo.split("/")[-1].replace("temporal-moe-", "")


def destination(root, repo, hf_path):
    """Repo-relative destination for one manifest row."""
    s = short(repo)
    first = hf_path.split("/")[0]
    if first in SIDECAR or hf_path in SIDECAR:
        return os.path.join(root, "artifacts", s, hf_path)
    base = LAYOUT.get((s, first)) or LAYOUT.get((s, "*"))
    if base is None:
        return os.path.join(root, "artifacts", s, hf_path)
    # tokenizer/ and tokenizer_tok16k/ collapse onto their own dirs; strip the segment so the
    # files land directly inside, not one level deeper. run_captures/ is the same case and matters
    # more: every probe script reads results/phase0/runs/<run>/router_log.pt, which is also what
    # this row's local_path says, so leaving the segment in place delivered the file one level
    # below where anything looks for it.
    if (s, first) in {("corpus", "tokenizer"), ("extras", "tokenizer_tok16k"),
                      ("extras", "run_captures")}:
        hf_path = "/".join(hf_path.split("/")[1:])
    # Renorm-era OLMoE files are quarantined (results/archive/olmoe_wrong_renorm/README.md);
    # reroute their pulls there so a fetch can never rehydrate them into results/ablations.
    name = hf_path.split("/")[-1]
    if s == "extras" and first == "ablations" and (
            name.startswith(("olmoe_adapt_", "olmoe_minflow_", "olmoe_cal", "olmoe_scratch"))):
        return os.path.join(root, "results", "archive", "olmoe_wrong_renorm", name)
    return os.path.join(root, base, hf_path)


def load_manifest(root):
    p = os.path.join(root, "results", "MANIFEST.csv")
    if not os.path.exists(p):
        sys.exit(f"manifest not found: {p}")
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["bytes"] = int(r["bytes"])
        r["dest"] = destination(root, r["hf_repo"], r["hf_path"])
    return rows


def select(rows, args):
    out = rows
    if args.repo:
        want = {r.lower() for r in args.repo}
        out = [r for r in out if short(r["hf_repo"]).lower() in want]
    if args.run:
        want = set(args.run)
        out = [r for r in out if r["run_name"] in want]
    if args.cited:
        out = [r for r in out if r["cited"] == "cited"]
    if args.uncited:
        out = [r for r in out if r["cited"] == "uncited"]
    if args.glob:
        out = [r for r in out if any(fnmatch.fnmatch(r["hf_path"], g) for g in args.glob)]
    if args.max_bytes:
        out = [r for r in out if r["bytes"] <= args.max_bytes]
    return out


def sha256_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def url_for(repo, hf_path):
    kind = "datasets/" if repo in DATASET_REPOS else ""
    return f"{HF}/{kind}{repo}/resolve/main/{urllib.parse.quote(hf_path)}"


def already_good(row):
    d = row["dest"]
    if not os.path.exists(d) or os.path.getsize(d) != row["bytes"]:
        return False
    return sha256_of(d) == row["sha256"]


def fetch_one(row, force):
    """Returns (status, row, detail). status in ok/skip/size/sha/http/error."""
    d = row["dest"]
    if not force and already_good(row):
        return ("skip", row, "already present and verified")
    os.makedirs(os.path.dirname(d), exist_ok=True)
    tmp = None
    try:
        req = urllib.request.Request(url_for(row["hf_repo"], row["hf_path"]),
                                     headers={"User-Agent": "temporal-moe-artifacts/1"})
        with urllib.request.urlopen(req, timeout=120) as resp, \
                tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(d)) as fh:
            tmp = fh.name
            shutil.copyfileobj(resp, fh, length=8 * 1024 * 1024)
    except urllib.error.HTTPError as e:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        return ("http", row, f"HTTP {e.code}")
    except Exception as e:  # network, DNS, timeout
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        return ("error", row, f"{type(e).__name__}: {e}")

    size = os.path.getsize(tmp)
    if size != row["bytes"]:
        os.unlink(tmp)
        return ("size", row, f"expected {row['bytes']} bytes, got {size}")
    got = sha256_of(tmp)
    if got != row["sha256"]:
        os.unlink(tmp)
        return ("sha", row, f"expected {row['sha256'][:16]}..., got {got[:16]}...")
    os.replace(tmp, d)
    return ("ok", row, "")


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or u == "TiB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def cmd_pull(args, root, rows):
    sel = select(rows, args)
    total = sum(r["bytes"] for r in sel)
    print(f"selected {len(sel)} of {len(rows)} files, {human(total)}")
    if not sel:
        return 0
    if args.dry_run:
        for r in sel[: args.list_limit]:
            print(f"  {human(r['bytes']):>10}  {short(r['hf_repo']):<13} {r['hf_path']}")
            print(f"              -> {os.path.relpath(r['dest'], root)}")
        if len(sel) > args.list_limit:
            print(f"  ... and {len(sel) - args.list_limit} more")
        return 0

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fetch_one, r, args.force): r for r in sel}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
            done += 1
            if done % max(1, len(sel) // 20) == 0 or done == len(sel):
                print(f"  {done}/{len(sel)}", flush=True)

    by = {}
    for status, row, detail in results:
        by.setdefault(status, []).append((row, detail))
    print()
    print(f"  downloaded : {len(by.get('ok', []))}")
    print(f"  already ok : {len(by.get('skip', []))}")
    failed = 0
    for status, label in (("sha", "SHA256 MISMATCH"), ("size", "SIZE MISMATCH"),
                          ("http", "HTTP ERROR"), ("error", "TRANSFER ERROR")):
        items = by.get(status, [])
        if not items:
            continue
        failed += len(items)
        print(f"  {label}: {len(items)}")
        for row, detail in items:
            print(f"    {short(row['hf_repo'])}/{row['hf_path']}  -> {detail}")
    if failed:
        print(f"\n{failed} file(s) failed. Nothing partial was left on disk; "
              f"re-run to retry just those.")
        return 1
    print("\nall selected files present and verified against MANIFEST.csv")
    return 0


def cmd_verify(args, root, rows):
    sel = select(rows, args)
    print(f"verifying {len(sel)} files against MANIFEST.csv")
    missing = bad = ok = 0
    for r in sel:
        d = r["dest"]
        if not os.path.exists(d):
            missing += 1
            if args.verbose:
                print(f"  MISSING  {os.path.relpath(d, root)}")
        elif os.path.getsize(d) != r["bytes"] or sha256_of(d) != r["sha256"]:
            bad += 1
            print(f"  CORRUPT  {os.path.relpath(d, root)}")
        else:
            ok += 1
    print(f"\n  verified: {ok}    missing: {missing}    corrupt: {bad}")
    return 1 if bad else 0


def cmd_push(args, root, rows):
    """Secondary. Deliberately refuses to act without an explicit flag."""
    sel = select(rows, args)
    print(f"push would consider {len(sel)} files, {human(sum(r['bytes'] for r in sel))}")
    changed = []
    for r in sel:
        d = r["dest"]
        if os.path.exists(d) and (os.path.getsize(d) != r["bytes"] or sha256_of(d) != r["sha256"]):
            changed.append(r)
    print(f"  differing from the manifest: {len(changed)}")
    for r in changed[:20]:
        print(f"    {short(r['hf_repo'])}/{r['hf_path']}")
    if not args.yes:
        print("\npush is dry-run only unless --yes is passed. Re-uploading would invalidate the\n"
              "sha256 column of a manifest this tool is not allowed to modify, so this is a\n"
              "deliberate speed bump rather than an oversight.")
        return 0
    print("\n--yes given, but upload is not implemented here: it needs huggingface_hub and a write\n"
          "token, neither of which the analysis environment has. Use huggingface_hub directly and\n"
          "regenerate MANIFEST.csv afterwards.")
    return 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pull", "verify", "push"):
        p = sub.add_parser(name)
        p.add_argument("--repo", action="append",
                       help="ckpts | extras | router-adapt | corpus (repeatable)")
        p.add_argument("--run", action="append", help="run_name from the manifest (repeatable)")
        p.add_argument("--cited", action="store_true", help="only rows marked cited")
        p.add_argument("--uncited", action="store_true", help="only rows marked uncited")
        p.add_argument("--glob", action="append", help="glob against hf_path (repeatable)")
        p.add_argument("--max-bytes", type=int, help="skip files larger than this")
        p.add_argument("--dest", help="destination root (default: $TMOE_ROOT or the repo root)")
        p.add_argument("--jobs", type=int, default=8)
        p.add_argument("--force", action="store_true", help="re-download even if verified")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--list-limit", type=int, default=25)
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--yes", action="store_true", help="push only: actually act")
    args = ap.parse_args()

    root = os.path.abspath(args.dest) if args.dest else repo_root()
    rows = load_manifest(repo_root())          # manifest always read from the checkout
    if args.dest:                              # but destinations may be redirected
        for r in rows:
            r["dest"] = destination(root, r["hf_repo"], r["hf_path"])
    print(f"root: {root}")
    return {"pull": cmd_pull, "verify": cmd_verify, "push": cmd_push}[args.cmd](args, root, rows)


if __name__ == "__main__":
    sys.exit(main())
