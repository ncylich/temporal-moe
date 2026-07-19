#!/usr/bin/env python3
"""Decode-serving protocol runner (see PLAN.md Section 2 / Phase 1).

Ceiling (setup a): all experts resident, standard forward. Protocol: B=1; prefill
`--context` random token ids to fill the KV cache (untimed, eval + synchronize);
then time `--n` greedy (argmax) decode steps as one block per rep, using the
mlx-lm generate_step async_eval pipeline (the stock-engine fair ceiling). 1
untimed warmup rep + `--reps` timed reps; reps chain off the growing cache (no
re-prefill between reps). Reports per-rep tok/s, mean +/- std, peak memory.

CSV schema: phase,model,tier,setup,ubatch,context,prefill_ms,decode_tok_s,
peak_vram_mib,note,decode_tok_s_std,copied_bytes_per_token
"""
import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import load
from temporal import TemporalController

ROOT = Path(__file__).resolve().parent


def parse_setup(setup):
    """Map a --setup string to (temporal_mode, floor_N). ceiling -> (None, 0)."""
    if setup == "ceiling":
        return None, 0
    if setup == "noswap":
        return "noswap", 0
    if setup == "deploy_sync":
        return "deploy", 0
    if setup == "deploy_early":
        return "deploy", 0
    if setup == "deploy_overlap":
        return "deploy_overlap", 0
    if setup.startswith("floor_n="):
        return "floor", int(setup.split("=", 1)[1])
    raise SystemExit(f"unknown --setup {setup!r} "
                     "(ceiling|noswap|deploy_sync|deploy_early|deploy_overlap|floor_n=<N>)")


def prefill(model, cache, ids):
    """Untimed prefill of `ids` (shape [1, context]) -> first decode token."""
    logits = model(ids, cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y, *[c.state for c in cache])
    mx.synchronize()
    return y  # shape [1]


def decode_block(model, cache, y_start, n, pipeline=True):
    """Time n greedy decode steps continuing from `y_start`.

    pipeline=True: mlx-lm generate_step-style async_eval pipeline (build token
    t+1's graph while t executes). pipeline=False: mx.eval each step — frees the
    previous graph so hot-slot scatter updates can donate their buffer (in-place)
    instead of copying the whole tensor.

    Returns (elapsed_s, ids_list, last_token). last_token feeds the next rep."""

    def step(tok):  # tok shape [1]
        logits = model(tok[None], cache=cache)  # [1,1,V]
        return mx.argmax(logits[:, -1, :], axis=-1)  # [1]

    mx.synchronize()
    t0 = time.perf_counter()
    y = y_start
    produced = []
    if pipeline:
        mx.async_eval(y)
        for i in range(n):
            next_y = step(y)
            mx.async_eval(next_y)
            if i == 0:
                mx.eval(y)
            produced.append(y)
            y = next_y
        mx.eval(y)
    else:
        for _ in range(n):
            y = step(y)
            mx.eval(y)
            produced.append(y)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    ids = [int(t.item()) for t in produced]
    return elapsed, ids, y


def run(args):
    model_dir = Path(args.model_dir)
    tier = "fine" if "fine" in model_dir.name else "coarse"

    model, config = load(model_dir)
    vocab = config["vocab_size"]

    mode, floor_n = parse_setup(args.setup)
    ctrl = None
    if config.get("disk_experts"):
        # xl (bigger-than-RAM) model: experts on disk, no full-resident tier.
        if mode is None:  # ceiling
            raise SystemExit(
                "ceiling is impossible for the xl disk model: all-resident needs "
                f"{config['num_hidden_layers']} x {config['num_experts']} x 663552 B "
                "= 30.6 GB of expert weights in RAM (the whole point of xl). Use "
                "noswap | floor_n=<N> | deploy_sync.")
        if mode == "deploy_overlap":
            mode = "deploy"
        from xl import XLController
        ctrl = XLController(model, config, model_dir, mode, N=floor_n)
    elif mode is not None:
        stream = None
        if mode == "deploy_overlap":
            mode = "deploy"
            stream = mx.new_stream(mx.gpu)
        ctrl = TemporalController(model, mode, N=floor_n, copy_stream=stream)
        if args.setup == "deploy_early":
            ctrl.router_early = True

    n = 32 if args.smoke else args.n
    reps = 2 if args.smoke else args.reps

    # deterministic prefill ids
    rng = np.random.default_rng(args.seed)
    ids = mx.array(rng.integers(0, vocab, size=(1, args.context), dtype=np.int64))

    mx.reset_peak_memory()
    cache = model.make_cache()
    if ctrl is not None:
        ctrl.reset()

    t0 = time.perf_counter()
    y = prefill(model, cache, ids)
    prefill_ms = (time.perf_counter() - t0) * 1e3

    # warmup rep (untimed) then timed reps; cache keeps growing (no re-prefill).
    # Temporal residency + copies run on every decode token (warmup + timed),
    # so copied bytes accumulate over all decode tokens processed.
    all_ids = []
    for _ in range(args.warmup):
        _, wid, y = decode_block(model, cache, y, n, pipeline=not args.no_pipeline)
        all_ids += wid
        if args.cooldown:
            time.sleep(args.cooldown)

    tok_s = []
    for _ in range(reps):
        elapsed, rid, y = decode_block(model, cache, y, n, pipeline=not args.no_pipeline)
        tok_s.append(n / elapsed)
        all_ids += rid
        if args.cooldown:
            time.sleep(args.cooldown)

    decode_tokens = n * (args.warmup + reps)
    bytes_per_token = (ctrl.copied_bytes // decode_tokens) if ctrl else 0

    peak_mib = mx.get_peak_memory() / 1024**2
    final_offset = int(cache[0].offset)

    mean = statistics.mean(tok_s)
    std = statistics.pstdev(tok_s) if len(tok_s) > 1 else 0.0
    std_pct = 100 * std / mean if mean else 0.0
    unique = len(set(all_ids))

    result = dict(
        model=model_dir.name,
        tier=tier,
        setup=args.setup,
        context=args.context,
        n=n,
        reps=reps,
        prefill_ms=round(prefill_ms, 2),
        decode_tok_s=round(mean, 2),
        decode_tok_s_std=round(std, 3),
        decode_tok_s_std_pct=round(std_pct, 2),
        per_rep_tok_s=[round(t, 2) for t in tok_s],
        peak_vram_mib=round(peak_mib, 1),
        copied_bytes_per_token=bytes_per_token,
        final_cache_offset=final_offset,
        unique_decode_ids=unique,
        total_decode_ids=len(all_ids),
        smoke=args.smoke,
        seed=args.seed,
        mlx_version=mx.__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    print(json.dumps(result, indent=2))

    # raw per-rep dump
    raw_dir = ROOT / "results_raw"
    raw_dir.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    raw_path = raw_dir / f"{model_dir.name}_{args.setup}_{tag}_{int(time.time())}.json"
    with open(raw_path, "w") as f:
        json.dump(result, f, indent=2)

    # CSV row
    if args.csv:
        note = f"cache {args.context}->{final_offset}; uniq_ids={unique}/{len(all_ids)}"
        import os as _os
        if _os.environ.get("TEMPORAL_DISK_POOL"):
            note = (f"DISK_TIER pool>RAM QD={_os.environ.get('TEMPORAL_DISK_QD', '8')} " + note)
        if args.smoke:
            note = "SMOKE " + note
        row = dict(
            phase="decode",
            model=model_dir.name,
            tier=tier,
            setup=args.setup,
            ubatch=1,
            context=args.context,
            prefill_ms=round(prefill_ms, 2),
            decode_tok_s=round(mean, 2),
            peak_vram_mib=round(peak_mib, 1),
            note=note,
            decode_tok_s_std=round(std, 3),
            copied_bytes_per_token=bytes_per_token,
        )
        cols = list(row.keys())
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not exists:
                w.writeheader()
            w.writerow(row)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--setup", default="ceiling",
                    help="ceiling | noswap | deploy_sync | deploy_early | floor_n=<N>")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-pipeline", action="store_true",
                    help="mx.eval each decode step instead of the async_eval "
                         "pipeline (lets hot-slot scatters donate in-place)")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep between decode reps (default 0). Use "
                         "on a Mac to bleed off thermal/boost between reps so "
                         "per-rep tok/s deviation stays low; adds wall time only "
                         "between reps, never inside a timed block.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
