#!/usr/bin/env python3
"""Append bits-per-byte (BPB) columns next to every cross-entropy (CE) column in
results/ablations/*.csv so all result tables share one comparable metric.

WHY
    Different tables report loss either as natural-log cross-entropy (nats/token)
    or as bits-per-byte (BPB). To compare them we convert every CE column that
    lacks a BPB counterpart into a new "<ce_col>_bpb" column.

CONVERSION FORMULA
    A CE in nats is a per-token surprisal in nats. To express it in bits per
    *byte* we (1) convert nats->bits by dividing by ln(2), and (2) convert
    per-token->per-byte by dividing by the corpus' bytes-per-token:

        BPB = CE_nats / (ln2 * bytes_per_token)

    The whole denominator (ln2 * bytes_per_token) is the run's "divisor" and
    depends only on tokenizer+corpus. Three canonical divisors are in play:

        DIVISOR   TOKENIZER / CORPUS
        2.7568    bpe-16k, original phase0 corpus
        2.7600    bpe-16k, G3-era corpus
        2.9780    pythia-50k corpus (all 1e18/1e19 flame*/t18/t19 runs)

DIVISOR SELECTION (verified, not guessed; per-file)
    a. If the file has a `bpb_divisor` column -> use it per-row.
    b. Else if the file has >=1 row with BOTH a CE value and an existing BPB
       value for the SAME quantity -> infer divisor = CE/BPB and require it to
       match a canonical divisor within 0.002; use the matched canonical value.
    c. Else use filename provenance: names starting with flame / t18_ / t19_ are
       pythia-50k -> 2.9780.
    d. Else: cannot determine -> DO NOT convert; the file is reported as skipped.

RULES ENFORCED
    * Never delete/replace/reorder existing columns, rows, or values. New columns
      are only APPENDED, named "<ce_col>_bpb".
    * Leading '#' comment lines are preserved byte-for-byte. Existing field bytes
      are preserved exactly: we append the new fields to the raw physical line
      rather than re-serializing untouched fields. New fields are plain numbers
      or empty, so they need no quoting.
    * BPB rounded to 4 decimals; blank left blank where the CE cell is blank/NA.
    * A CE column is SKIPPED if it already has a BPB counterpart: either a generic
      "<ce>_bpb" column (also gives idempotency), or a known differently-named
      counterpart listed in EXPLICIT_COUNTERPART_SKIP below.
    * Idempotent: re-running does not double-append (existing "<ce>_bpb" columns
      are detected and skipped).

CE column identification: a column counts as CE if its lowercased name contains
one of CE_NAME_SUBSTRINGS and does NOT contain 'bpb'. CE at these scales is
~3-11 nats; non-CE metrics (PPL, AUC, %, counts, swap rates, delta_bpb, ...) are
excluded by name.
"""
import csv
import glob
import math
import os
import sys

LN2 = math.log(2.0)
CANONICAL_DIVISORS = [2.7568, 2.7600, 2.9780]
INFER_TOL = 0.002  # rule 5b tolerance for matching an inferred divisor

# Substrings (lowercased) that mark a natural-log CE column. A name is CE if it
# contains one of these AND does not contain 'bpb'.
CE_NAME_SUBSTRINGS = ("val_ce", "test_ce", "ce_nats", "ce_test_final", "train_lm_loss")

# CE columns that already have a BPB counterpart under a NON-generic name
# (i.e. not "<ce>_bpb"), keyed by basename. These are skipped.
EXPLICIT_COUNTERPART_SKIP = {
    "flame192_leftflank_1e18.csv": {"test_CE_nats"},   # counterpart: BPB_pythia50k
    "phase0_isoflop_points.csv": {"ce_test_final"},    # counterpart: bpb_test_final
    "phase0_lr_tuning.csv": {"val_ce"},                # counterpart: val_bpb
    "seed_replicates.csv": {"val_ce", "test_ce"},      # counterparts: val_bpb / test_bpb
}

BLANK_TOKENS = {"", "na", "nan", "n/a", "none", "null"}


def is_ce_col(name):
    low = name.lower()
    if "bpb" in low:
        return False
    return any(s in low for s in CE_NAME_SUBSTRINGS)


def is_blank(v):
    return v.strip().lower() in BLANK_TOKENS


def determine_divisor(basename, header, data_rows):
    """Return (divisor_or_None_or_'percol', how_str).

    'percol' means a per-row bpb_divisor column is used (divisor varies).
    """
    # (a) explicit per-row divisor column
    if "bpb_divisor" in header:
        return ("percol", "per-row bpb_divisor column (rule 5a)")

    # (b) infer from a CE/BPB pair present in the same file
    ce_cols = [c for c in header if is_ce_col(c)]
    bpb_cols = [c for c in header if "bpb" in c.lower() and c != "bpb_divisor"]
    idx = {c: i for i, c in enumerate(header)}
    for ce in ce_cols:
        for bp in bpb_cols:
            ratios = []
            for r in data_rows:
                if len(r) <= max(idx[ce], idx[bp]):
                    continue
                cv, bv = r[idx[ce]], r[idx[bp]]
                if is_blank(cv) or is_blank(bv):
                    continue
                try:
                    c = float(cv)
                    b = float(bv)
                except ValueError:
                    continue
                if b > 0:
                    ratios.append(c / b)
            if len(ratios) >= 1:
                mean_ratio = sum(ratios) / len(ratios)
                for canon in CANONICAL_DIVISORS:
                    if abs(mean_ratio - canon) <= INFER_TOL:
                        return (canon, "inferred from %s/%s ratio~%.4f (rule 5b, matches %.4f)"
                                % (ce, bp, mean_ratio, canon))
    # (c) filename provenance
    if basename.startswith("flame") or basename.startswith("t18_") or basename.startswith("t19_"):
        return (2.9780, "filename provenance pythia-50k (rule 5c)")

    # (d) unknown
    return (None, "undetermined (rule 5d)")


def fmt_bpb(ce_str, divisor):
    if is_blank(ce_str):
        return ""
    try:
        ce = float(ce_str)
    except ValueError:
        return ""
    return "%.4f" % (ce / divisor)


def process_file(path):
    basename = os.path.basename(path)
    raw = open(path, "r", newline="").read()
    # Split keeping line endings so we can re-append the exact original bytes.
    lines = raw.splitlines(keepends=True)

    # Locate header (first non-comment physical line) and its index.
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            continue
        header_idx = i
        break
    if header_idx is None:
        return (basename, "no header", None, [], 0)

    def split_nl(ln):
        for nl in ("\r\n", "\n", "\r"):
            if ln.endswith(nl):
                return ln[:-len(nl)], nl
        return ln, ""

    header_body, _ = split_nl(lines[header_idx])
    header = next(csv.reader([header_body]))

    # Parse data rows (values only, for divisor inference + conversion).
    data_line_indices = [i for i in range(header_idx + 1, len(lines))
                         if lines[i].strip() != ""]
    data_rows = []
    for i in data_line_indices:
        body, _ = split_nl(lines[i])
        data_rows.append(next(csv.reader([body])))

    # Candidate CE columns.
    ce_cols = [c for c in header if is_ce_col(c)]

    # Decide which to convert (skip those with an existing counterpart).
    explicit_skip = EXPLICIT_COUNTERPART_SKIP.get(basename, set())
    to_add = []      # (ce_col, new_col_name)
    skipped = []     # (ce_col, reason)
    for ce in ce_cols:
        new_name = ce + "_bpb"
        if new_name in header:
            skipped.append((ce, "already has %s (idempotent)" % new_name))
            continue
        if ce in explicit_skip:
            skipped.append((ce, "existing BPB counterpart in file"))
            continue
        to_add.append((ce, new_name))

    if not to_add:
        return (basename, "nothing to add", None, ce_cols, len(data_rows), skipped, [])

    # Determine divisor now that we know we need it.
    divisor, how = determine_divisor(basename, header, data_rows)
    if divisor is None:
        return (basename, "DIVISOR-UNDETERMINED", how, ce_cols, len(data_rows), skipped, to_add)

    idx = {c: i for i, c in enumerate(header)}
    div_col_idx = idx.get("bpb_divisor")

    def row_divisor(row):
        if divisor == "percol":
            return float(row[div_col_idx])
        return divisor

    # Build new content: append fields to raw lines.
    out = list(lines)

    # Header: append new column names (plain identifiers, no quoting needed).
    hbody, hnl = split_nl(lines[header_idx])
    out[header_idx] = hbody + "," + ",".join(n for _, n in to_add) + hnl

    conv_counts = {n: 0 for _, n in to_add}
    for i in data_line_indices:
        body, nl = split_nl(lines[i])
        row = next(csv.reader([body]))
        d = row_divisor(row)
        new_fields = []
        for ce, new_name in to_add:
            cv = row[idx[ce]] if idx[ce] < len(row) else ""
            val = fmt_bpb(cv, d)
            if val != "":
                conv_counts[new_name] += 1
            new_fields.append(val)
        out[i] = body + "," + ",".join(new_fields) + nl

    with open(path, "w", newline="") as fh:
        fh.write("".join(out))

    return (basename, "OK", "%s -> divisor=%s" % (how, divisor), ce_cols,
            len(data_rows), skipped, [(n, conv_counts[n]) for _, n in to_add])


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "results", "ablations")
    root = os.path.normpath(root)
    files = sorted(glob.glob(os.path.join(root, "*.csv")))
    print("Scanning %d CSVs in %s\n" % (len(files), root))
    for path in files:
        base = os.path.basename(path)
        raw = open(path, "r", newline="").read()
        lines = raw.splitlines(keepends=True)
        header_idx = next((i for i, ln in enumerate(lines)
                          if not ln.lstrip().startswith("#")), None)
        if header_idx is None:
            continue
        header = next(csv.reader([lines[header_idx].rstrip("\r\n")]))
        if not any(is_ce_col(c) for c in header):
            continue  # no CE columns -> nothing to do, stay quiet

        res = process_file(path)
        base, status = res[0], res[1]
        how = res[2]
        ce_cols = res[3]
        nrows = res[4]
        skipped = res[5] if len(res) > 5 else []
        added = res[6] if len(res) > 6 else []
        print("== %s ==" % base)
        print("   CE cols found : %s" % ", ".join(ce_cols))
        print("   status        : %s" % status)
        if how:
            print("   divisor       : %s" % how)
        if skipped:
            for ce, reason in skipped:
                print("   skipped col   : %s (%s)" % (ce, reason))
        if status == "OK":
            for name, cnt in added:
                print("   added col     : %s (%d/%d rows converted)" % (name, cnt, nrows))
        elif status == "DIVISOR-UNDETERMINED":
            print("   NOT CONVERTED : could not determine divisor")
        print()


if __name__ == "__main__":
    main()
