#!/usr/bin/env python3
"""Parse a Phase-0 train.log: final val loss, val loss at the 1e16 iter (iters/10),
NaN check, and last train loss. Prints a one-line summary + JSON.

Usage: parse_run.py <run_dir>
"""
import sys, re, json, os, math

# BPB = CE_nats / (ln2 * bytes_per_token). Divisor depends on the tokenizer of the run.
# Measured on held-out dclm: bpe-16k -> 2.7568, pythia-50k -> 2.9780.
BPB_DIVISOR = float(os.environ.get("BPB_DIVISOR", "2.7568"))
def bpb(ce):
    return round(ce / BPB_DIVISOR, 4) if ce is not None else None

def main():
    run = sys.argv[1]
    log = os.path.join(run, "train.log")
    meta = open(os.path.join(run, "run.meta")).read() if os.path.exists(os.path.join(run, "run.meta")) else ""
    m = re.search(r"iters=(\d+)", meta)
    total_iters = int(m.group(1)) if m else None
    iters_1e16 = round(total_iters / 10) if total_iters else None

    txt = open(log, errors="ignore").read()
    # validation lines: " validation loss at <prefix> | lm loss value: 4.93E+00 | ..."
    vals = []  # (prefix_iter_or_None, loss)
    for line in txt.splitlines():
        if "validation loss at" in line and "lm loss value:" in line:
            pm = re.search(r"validation loss at ([^|]+)\|", line)
            lm = re.search(r"lm loss value:\s*([0-9.Ee+\-]+)", line)
            if lm:
                prefix = pm.group(1).strip() if pm else ""
                it = None
                im = re.search(r"iteration\s+(\d+)", prefix)
                if im:
                    it = int(im.group(1))
                vals.append((it, float(lm.group(1)), prefix))
    nan = bool(re.search(r"(?i)\bnan\b", txt)) and ("loss value: nan" in txt.lower() or "found nan" in txt.lower())
    # last train loss
    tl = re.findall(r"lm loss:\s*([0-9.Ee+\-]+)", txt)
    last_train = float(tl[-1]) if tl else None

    final_val = vals[-1][1] if vals else None
    # val at 1e16 point: closest logged iter to iters_1e16
    val_1e16 = None
    if iters_1e16 and vals:
        cand = [(abs((it or 0) - iters_1e16), loss, it) for it, loss, _ in vals if it is not None]
        if cand:
            cand.sort()
            val_1e16 = {"iter": cand[0][2], "loss": cand[0][1]}

    out = dict(run=os.path.basename(run), total_iters=total_iters,
               iters_1e16=iters_1e16, final_val_loss=final_val,
               final_val_bpb=bpb(final_val),
               final_val_ppl=(round(math.exp(min(20, final_val)),1) if final_val else None),
               val_at_1e16=val_1e16,
               val_at_1e16_bpb=(bpb(val_1e16["loss"]) if val_1e16 else None),
               last_train_loss=last_train,
               nan=nan, n_val_evals=len(vals), bpb_divisor=BPB_DIVISOR)
    print(json.dumps(out))
    fv = f"{final_val:.4f} (BPB {bpb(final_val):.4f})" if final_val else "NA"
    v16 = f"{val_1e16['loss']:.4f} (BPB {bpb(val_1e16['loss']):.4f})@it{val_1e16['iter']}" if val_1e16 else "NA"
    print(f"SUMMARY {out['run']}: final_val_CE={fv}  val@iters/10={v16}  nan={nan}  evals={len(vals)}")

if __name__ == "__main__":
    main()
