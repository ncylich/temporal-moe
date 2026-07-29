#!/usr/bin/env python3
"""Fold results into two tidy CSVs, one per line of inquiry.

    results/ablations/ple_results.csv            per-layer embeddings
    results/ablations/layer_freeing_results.csv  per-layer residency relaxation

These are separate experiments and are kept apart deliberately. PLE ADDS a token-indexed lookup
while leaving the residency constraint intact; layer freeing REMOVES the constraint from chosen
layers and adds nothing. They share a base model, an eval slice and a set of published references,
and nothing else -- so mixing them in one table invites comparisons that are not like-for-like.

The shared references (base, impose, C, CE, F-prime, the 2 sigma bar, the divisor) are written into
BOTH files so each is self-contained and readable without the other.

The program produced eleven CSVs and six JSONs across seven kinds of measurement. That is hard to
read and harder to query. This emits a single long-format table instead:

    group, name, metric, value, note

Long rather than wide because the seven measurement kinds have irreconcilable schemas -- a trained
cell has a token budget and a BPB curve, an accounting row has bytes per token, a locus row has
AUCs. Forcing them into one wide table would be mostly empty cells; forcing them into separate
tables is what we are trying to get away from. Long format costs verbosity and buys one file that
any of pandas, csv or grep can slice by `group`.

Groups, and where each comes from:

    trained_cell   authoritative: the per-cell JSON the trainer writes at exit
    reference      published bake-off numbers, carried so comparisons are self-contained
    parity         flag-off vs the unmodified reference trainer, and the determinism floor
    accounting     parameters / flash fetch / training memory per rank
    coverage       audited-slice coverage by type, occurrence and eval loss
    trainfree      §9 closed-form tables scored with no training, and frequency buckets
    calibration    lambda*, reconstruction error and capture reference per calibrated table
    cal_stack      the fully training-free norm+PLE stack, both orderings
    layer_damage   BPB cost of constraining exactly one MoE layer
    free_set       BPB cost of leaving a SUBSET of layers unconstrained, vs the additive prediction
    locus          §8.1 token vs context AUC, with the no-PLE control
    row_norms      §2 diagnostic: table row norm bucketed by occurrence count
    heldout        the deliberately held-out token set used by the zero-property check

Run after any new cell; it is idempotent and rewrites the file from scratch.
"""

import csv, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                 # noqa: E402

BASE, IMPOSE, TWO_SIGMA = 0.6727, 2.7507, 0.012
OUT_PLE = os.path.join(ABLATIONS, "ple_results.csv")
OUT_LF = os.path.join(ABLATIONS, "layer_freeing_results.csv")
rows, lf_rows = [], []

# Cells with no PLE table at all, whose whole content is which layers were left unconstrained.
LF_CELLS = {"ce_free2", "ce_free_0_1_15", "ce_free_0_1_2"}
LF_GROUPS = {"layer_damage", "free_set"}


def add(group, name, metric, value, note=""):
    r = {"group": group, "name": name, "metric": metric, "value": value, "note": note}
    (lf_rows if (group in LF_GROUPS or name in LF_CELLS) else rows).append(r)


def rec(b):
    return round((1 - (b - BASE) / (IMPOSE - BASE)) * 100, 2)


# ---------------------------------------------------------------- trained cells (authoritative)
CELL_NOTE = {
    "ladder_full": "Phase 1 rank ladder, zero-init PLE",
    "ladder_r512": "Phase 1 rank ladder, zero-init PLE",
    "ladder_r128": "Phase 1 rank ladder, zero-init PLE",
    "ladder_r32":  "Phase 1 rank ladder, zero-init PLE",
    "seq_ple_512": "Phase 3 sequential: PLE introduced at 50M of 100M",
    "cal_r512":    "calibrated PLE init, base norms",
    "cal_full":    "calibrated PLE init, base norms",
    "cal_seq_512": "calibrated PLE init at 50M, delta captured vs the 50M surface",
    "calstack_full": "init from the PLE-then-norms training-free stack (53.13%)",
    "ce_ple_512":  "CE surface: PLE + router + norms + LoRA r32",
    "ce_ple_128":  "CE surface: PLE + router + norms + LoRA r32",
    "ce_free2":    "CE surface, NO PLE, MoE layers 0-1 unconstrained, +87.5% resident memory",
    "ce_free_0_1_15": "CE surface, NO PLE, MoE layers 0/1/15 unconstrained, +131.2% resident memory",
    "ce_free_0_1_2":  "CE surface, NO PLE, MoE layers 0/1/2 unconstrained, +131.2% resident memory",
}
for p in sorted(glob.glob(os.path.join(DATA_DIR, "ple_*.json"))):
    base = os.path.basename(p)
    if base.startswith(("ple_parity", "ple_heldout")):
        continue
    try:
        r = json.load(open(p))
    except Exception:
        continue
    if "final_bpb" not in r:
        continue
    t = r["tag"]
    note = CELL_NOTE.get(t, "")
    add("trained_cell", t, "final_bpb", round(r["final_bpb"], 6), note)
    add("trained_cell", t, "recovery_pct", rec(r["final_bpb"]), note)
    add("trained_cell", t, "train_tokens", r["train_tokens"])
    add("trained_cell", t, "rank", r["rank"])
    add("trained_cell", t, "lora_r", r.get("lora", 0))
    add("trained_cell", t, "free_set", r.get("free_set", "") or
        (f"first {r['free_layers']}" if r.get("free_layers") else ""))
    add("trained_cell", t, "ple_params", r.get("ple_params", 0))
    add("trained_cell", t, "mb_x_accum", f"{r['mb']}x{16 // r['mb']}")
    add("trained_cell", t, "table_wd", r.get("table_wd", ""))
    add("trained_cell", t, "calib_init", r.get("calib_init", False))
    add("trained_cell", t, "ple_start_tokens", r.get("ple_start", 0))
    add("trained_cell", t, "final_swap_rate", round(r["final_swap"], 6))
    add("trained_cell", t, "final_usage_entropy", round(r["final_entropy"], 6))
    add("trained_cell", t, "bpb_curve",
        "|".join(f"{h['tok'] // 10**6}M:{h['bpb']:.6f}" for h in r["curve"]))
    if r["curve"] and "train_lm" in r["curve"][0]:
        add("trained_cell", t, "train_bpb_curve",
            "|".join(f"{h['tok'] // 10**6}M:{h['train_bpb']:.6f}" for h in r["curve"]))

# ---------------------------------------------------------------- published references
for nm, b, what in (("base_free_routing", 0.6727, "unconstrained ceiling"),
                    ("impose_R8_untrained", 2.7507, "the gap being recovered"),
                    ("A_router_only", 1.2825, "bake-off arm A"),
                    ("C_router_norms_50M", 0.8791, "matched-budget comparator for 50M cells"),
                    ("C_router_norms_250M", 0.8505, "5x token-efficiency comparator"),
                    ("CE_router_norms_lora_50M", 0.8269, "LoRA as the third mechanism"),
                    ("Fprime_full_finetune_6.92B", 0.8106, "the constraint price")):
    for dest in (rows, lf_rows):
        dest.append({"group": "reference", "name": nm, "metric": "bpb", "value": b, "note": what})
        dest.append({"group": "reference", "name": nm, "metric": "recovery_pct",
                     "value": rec(b), "note": what})
for dest in (rows, lf_rows):
    dest.append({"group": "reference", "name": "two_sigma_bar", "metric": "bpb", "value": TWO_SIGMA,
                 "note": "differences below this are noise (PLE_PLAN.md §3)"})
    dest.append({"group": "reference", "name": "divisor_D", "metric": "value",
                 "value": 3.1089070924799973, "note": "ln2 x bytes_per_token, audited slice"})

# ---------------------------------------------------------------- simple CSV passthroughs
def take(fname, group, key, metrics, notecol=None):
    path = os.path.join(ABLATIONS, fname)
    if not os.path.exists(path):
        return 0
    n = 0
    for r in csv.DictReader(open(path)):
        nm = r[key]
        note = r.get(notecol, "") if notecol else ""
        for m in metrics:
            if m in r and r[m] != "":
                add(group, nm, m, r[m], note)
                n += 1
    return n


take("ple_parity.csv", "parity", "arm", ["bpb", "swap", "entropy", "tokens"], "what")
take("ple_accounting.csv", "accounting", "rank",
     ["total_params", "flash_fetch_bytes_per_token", "resident_basis_bytes", "expert_swap_bytes",
      "fetch_vs_expert_swap", "train_param_plus_grad_GiB", "train_adam8bit_GiB",
      "train_total_adam8bit_GiB"], "note")
take("layer_damage.csv", "layer_damage", "layer", ["bpb", "damage_bpb"])
take("joint_free.csv", "free_set", "free_set",
     ["bpb", "damage_bpb", "additive_prediction", "interaction", "resident_slots",
      "memory_vs_full_residency_pct"])
take("ple_locus.csv", "locus", "tag",
     ["token_AUC_median", "context_AUC_median", "context_minus_token_median", "n_experts_probed"])
take("ple_cal_stack.csv", "cal_stack", "stage", ["bpb", "recovery_pct"], "trained")
take("ple_row_norms.csv", "row_norms", "occurrence_bucket",
     ["n_rows", "mean_row_norm", "mean_contrib_rms", "frac_rows_exactly_zero"], "tag")

# coverage is a single wide row
cov = os.path.join(ABLATIONS, "ple_coverage.csv")
if os.path.exists(cov):
    for r in csv.DictReader(open(cov)):
        for k, v in r.items():
            if v != "":
                add("coverage", "audited_slice", k, v)

# training-free evals + frequency buckets
tf = os.path.join(ABLATIONS, "ple_trainfree_and_buckets.csv")
if os.path.exists(tf):
    for r in csv.DictReader(open(tf)):
        nm = r["tag"] if r["bucket"] == "ALL" else f"{r['tag']}/occ_{r['bucket']}"
        add("trainfree", nm, "bpb", r["bpb"], r["table"])
        add("trainfree", nm, "recovery_pct", r["recovery_pct"], r["table"])
        add("trainfree", nm, "n_eval_tokens", r["n_eval_tokens"])

# calibration metadata, one group per capture
for p in sorted(glob.glob(os.path.join(ABLATIONS, "ple_calib_meta*.json"))):
    tag = os.path.basename(p).replace("ple_calib_meta", "").replace(".json", "").lstrip("_") or "untrained_base"
    if tag in ("dry", "drychk"):
        continue                     # smoke-test captures, not results
    m = json.load(open(p))
    for k in ("lambda_star", "within", "between", "tokens_seen", "tokens",
              "rel_recon_err_r512", "reference", "free_side", "delta"):
        if k in m:
            v = round(m[k], 6) if isinstance(m[k], float) else m[k]
            add("calibration", tag, k, v)

# held-out set summary only; the 160 ids are regenerable with heldout.py
hp = os.path.join(ABLATIONS, "ple_heldout.csv")
if os.path.exists(hp):
    hr = list(csv.DictReader(open(hp)))
    add("heldout", "zero_property_set", "n_tokens", len(hr),
        "stratified across occurrence deciles; ids regenerable with heldout.py")
    add("heldout", "zero_property_set", "eval_loss_share",
        round(sum(float(x["eval_loss_share"]) for x in hr), 8))
    add("heldout", "zero_property_set", "corpus_count_min",
        min(int(x["corpus_count_in_cell"]) for x in hr))
    add("heldout", "zero_property_set", "corpus_count_max",
        max(int(x["corpus_count_in_cell"]) for x in hr))

from collections import Counter
for path, rr, label in ((OUT_PLE, rows, "per-layer embeddings"),
                        (OUT_LF, lf_rows, "per-layer residency relaxation")):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "name", "metric", "value", "note"])
        w.writeheader()
        w.writerows(rr)
    print(f"wrote {os.path.basename(path)}  ({len(rr)} rows, {os.path.getsize(path)} bytes)  {label}")
    for g, n in sorted(Counter(x["group"] for x in rr).items(), key=lambda kv: -kv[1]):
        print(f"    {g:14s} {n:4d}")
