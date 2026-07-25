#!/usr/bin/env python3
"""Convert llama-bench CSV output into the FLAME-MoE serving-benchmark schema.

Schema matches llamacpp-bench/results/ablations/serving_benchmarks.csv and
mlx-bench/results/serving_benchmarks_mac.csv so Android rows land in the same analysis:

  phase,model,tier,setup,ubatch,context,prefill_ms,decode_tok_s,peak_vram_mib,note,
  decode_tok_s_std,copied_bytes_per_token

prefill_ms: lower is better. decode_tok_s: higher is better (mean over reps).
decode_tok_s_std: population stdev over reps; the noise floor for calling a gap real.
peak_vram_mib: left empty here -- Android PSS is not comparable to nvidia-smi global max
or to mx.get_peak_memory(); it is measured separately and labelled.
copied_bytes_per_token: always 0 -- the temporal swap kernel is CUDA-only, so Android
can only produce the all-resident ceiling.
"""
import csv
import math
import sys

raw_path, model, note, status = sys.argv[1:5]

with open(raw_path) as f:
    rows = list(csv.DictReader(f))
if not rows:
    sys.exit(f"FATAL: {raw_path} has no rows")

prefill_ms = decode_ts = decode_sd = None
ubatch = context = ""

for r in rows:
    avg_ns, avg_ts = float(r["avg_ns"]), float(r["avg_ts"])
    ubatch, context = r["n_ubatch"], r["n_depth"]
    if int(r["n_gen"]) == 0:          # the pp (prefill) row
        prefill_ms = avg_ns / 1e6
    else:                             # the tg (decode) row
        decode_ts, decode_sd = avg_ts, float(r["stddev_ts"])

# audit-1: a zero or non-finite metric must never be recorded as a success.
bad = [n for n, v in (("prefill_ms", prefill_ms),
                      ("decode_tok_s", decode_ts)) if v is None or not math.isfinite(v) or v <= 0]
if bad:
    status = "error"
    note += f";invalid={'+'.join(bad)}"

w = csv.writer(sys.stdout)
w.writerow(["decode", model, "android-cpu", "ceiling", ubatch, context,
            f"{prefill_ms:.3f}" if prefill_ms else "",
            f"{decode_ts:.4f}" if decode_ts else "",
            "",                                   # peak_vram_mib: measured separately
            f"[{status}] {note}",
            f"{decode_sd:.4f}" if decode_sd is not None else "",
            0])
