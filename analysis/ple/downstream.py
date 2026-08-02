#!/usr/bin/env python3
"""Downstream 10-task 0-shot evaluation of a trained free-set surface (PLE_PLAN.md §8.2).

`ple_RESULTS.md` §7 records this as not runnable: "lm_eval is incompatible with transformers 5.12.1
(`AutoModelForVision2Seq` removed)". That is true of an unguarded import and false of this program --
`scripts/adaptation/lmeval_downstream.py` had already worked around it two months earlier by stubbing
the two vision model modules before importing anything from lm_eval, and its 706 KB log is the
published `olmoe_adapt_downstream.csv`. The blocker was solved in a sibling script and the solution
was not carried across. Same three lines are at the top of this file.

What it does: rebuild the CE surface (router + RMSNorm gains + LoRA r32) from a `csurf_*.pt`, relax
residency on the chosen layers, and score the same ten tasks with the same harness and the same
primary-metric convention as the published table, so the new cell drops into it as another column.

**The surface is verified before it is scored.** Loading masters into the wrong parameters is silent
-- the shapes line up, the model runs, and the accuracies are simply of a different model. So BPB on
the audited slice is recomputed and matched against what the training cell recorded, and a mismatch
aborts rather than reports. This is the check `merge_ce.py` made for the same reason.

    downstream.py --csurf csurf_ce_free_0_1_2_at200M.pt --free-set 0,1,2 --tag ce_free_0_1_2_200M

Writes/extends results/ablations/layer_freeing_downstream.csv. Reference columns (base free routing,
residency imposed untrained, CE-adapted at full residency) are read from the published
olmoe_adapt_downstream.csv rather than recomputed: same harness, same venv, same tasks, and
recomputing them would cost an hour to reproduce numbers that are already committed.
"""
import sys, types

# Must precede any lm_eval import. lm_eval's model registry imports every backend eagerly, and the
# two VLM backends reference transformers classes this version removed. Nothing here uses them.
for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)

import argparse, csv, json, os                                       # noqa: E402
import torch                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                              # noqa: E402
from olmoe_paths import DATA_DIR                                     # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

from lm_eval.models.huggingface import HFLM                          # noqa: E402
from lm_eval import simple_evaluate                                  # noqa: E402

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "sciq", "boolq", "lambada_openai", "copa"]
REF = os.path.join(ABLATIONS, "olmoe_adapt_downstream.csv")
OUT = os.path.join(ABLATIONS, "layer_freeing_downstream.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--csurf", required=True, help="csurf_*.pt written by train_ple.py, in the data dir")
ap.add_argument("--free-set", required=True, help="layers left unconstrained, e.g. 0,1,2")
ap.add_argument("--tag", required=True, help="column label for this cell")
ap.add_argument("--expect-bpb", type=float, default=None,
                help="audited-slice BPB this surface is known to reach. Defaults to the value in "
                     "ple_<cell>.json. The rebuilt model must match it or the run aborts.")
ap.add_argument("--tol", type=float, default=2e-4,
                help="how far the rebuilt model's BPB may sit from the recorded one. Eval is "
                     "deterministic given the same weights, so this covers reload rounding, not "
                     "sampling: a real load error moves BPB by whole tenths, not by 1e-4.")
ap.add_argument("--batch-size", type=int, default=16)
A = ap.parse_args()

FREE = [int(x) for x in A.free_set.split(",") if x.strip() != ""]
CK = A.csurf if os.path.isabs(A.csurf) else os.path.join(DATA_DIR, A.csurf)
D = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))["divisor_D"]

# ---- rebuild the surface ----------------------------------------------------------------------
model, tok = RES.load_model()
RES.enable_residency(R=8)
RES.set_free_layers(FREE)
_slots = (16 - len(FREE)) * 8 + len(FREE) * 64
print(f"[ds] layers {FREE} UNCONSTRAINED, rest R=8; resident slots {_slots} vs 128 "
      f"(+{_slots / 128 * 100 - 100:.1f}% memory)", flush=True)

RES.freeze_all_but_router(model)
train_params = RES.router_params(model) + RES.norm_params(model) + RES.add_lora(model, r=32, alpha=64)
ck = torch.load(CK, map_location="cuda")
if len(ck["masters"]) != len(train_params):
    sys.exit(f"[abort] {os.path.basename(CK)} holds {len(ck['masters'])} tensors, this surface has "
             f"{len(train_params)}. The checkpoint was written by a different recipe -- check --lora.")
with torch.no_grad():
    for p, m in zip(train_params, ck["masters"]):
        if tuple(p.shape) != tuple(m.shape):
            sys.exit(f"[abort] shape mismatch loading masters: {tuple(m.shape)} into {tuple(p.shape)}")
        p.data.copy_(m.to("cuda").to(p.dtype))
print(f"[ds] loaded {os.path.basename(CK)} (seen={ck['seen'] / 1e6:.0f}M, lora_r={ck.get('lora')})",
      flush=True)

# ---- verify it is the model we think it is ------------------------------------------------------
expect = A.expect_bpb
if expect is None:
    cell = os.path.basename(CK)[len("csurf_"):].rsplit("_at", 1)[0]
    jpath = os.path.join(DATA_DIR, f"ple_{cell}.json")
    if os.path.exists(jpath):
        j = json.load(open(jpath))
        want_tok = ck["seen"]
        hit = [h for h in j["curve"] if abs(h["tok"] - want_tok) < 10**6]
        expect = hit[-1]["bpb"] if hit else (j["final_bpb"] if j["train_tokens"] == want_tok else None)
    if expect is None:
        sys.exit(f"[abort] no recorded BPB for {os.path.basename(CK)} at {ck['seen'] / 1e6:.0f}M "
                 f"tokens; pass --expect-bpb explicitly rather than scoring an unverified surface.")

bpb_ids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt")
eval_sub = bpb_ids[torch.linspace(0, bpb_ids.shape[0] - 1, 256).long()].to("cuda").long()
model.eval()
tot = n = 0
with torch.no_grad():
    for i in range(eval_sub.shape[0]):
        x = eval_sub[i:i + 1]
        out = model(x).logits.float()
        tot += torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                                 x[:, 1:].reshape(-1), reduction="sum").item()
        n += x[:, 1:].numel()
got = (tot / n) / D
if abs(got - expect) > A.tol:
    sys.exit(f"[abort] rebuilt surface scores BPB {got:.6f}, training recorded {expect:.6f} "
             f"(delta {got - expect:+.6f} > tol {A.tol}). The weights loaded are not the weights "
             f"trained; downstream numbers from this model would be of some other model.")
print(f"[ds] identity check OK: BPB {got:.6f} vs recorded {expect:.6f} ({got - expect:+.2e})", flush=True)

# ---- score ---------------------------------------------------------------------------------------
RES.enable_residency(R=8)
RES.set_free_layers(FREE)
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=A.batch_size)
print(f"[ds] scoring {len(TASKS)} tasks 0-shot ...", flush=True)
res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, bootstrap_iters=1000)["results"]


def get(d, metric):
    vk = next((k for k in d if k == metric or k.startswith(metric + ",")), None)
    sk = next((k for k in d if k.startswith(metric + "_stderr")), None)
    return (float(d[vk]) if vk else None, float(d[sk]) if sk else None)


# ---- join onto the published reference columns ----------------------------------------------------
ref = {}
with open(REF) as f:
    ref_rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
hdr, ref_rows = ref_rows[0], ref_rows[1:]
for r in ref_rows:
    ref[(r[hdr.index("task")], r[hdr.index("metric")])] = r
missing = [t for t in TASKS if not any(k[0] == t for k in ref)]
if missing:
    sys.exit(f"[abort] {REF} has no rows for {missing}; the reference columns would be blank for "
             f"tasks this cell scored, which is not a comparison.")

rows = []
for t in TASKS:
    for m in ("acc", "acc_norm"):
        v, se = get(res.get(t, {}), m)
        r = ref.get((t, m))
        if v is None or r is None:
            continue
        base = float(r[hdr.index("base_free")])
        imp = float(r[hdr.index("impose_R8")])
        ce = float(r[hdr.index("CE_adapt_R8")]) if r[hdr.index("CE_adapt_R8")] else None
        # fraction of the damage the constraint did that this cell has undone. 1.0 = back to free
        # routing, 0.0 = no better than the untrained mask. Undefined where the constraint did no
        # damage, which is why it is blank rather than a divide-by-near-zero.
        closed = (v - imp) / (base - imp) if abs(base - imp) > 1e-6 else None
        ce_closed = (ce - imp) / (base - imp) if (ce is not None and abs(base - imp) > 1e-6) else None
        rows.append([t, m, f"{base:.4f}", f"{imp:.4f}", f"{ce:.4f}" if ce is not None else "",
                     f"{v:.4f}", f"{se:.4f}" if se else "",
                     f"{v - base:+.4f}", f"{v - ce:+.4f}" if ce is not None else "",
                     f"{closed:.4f}" if closed is not None else "",
                     f"{ce_closed:.4f}" if ce_closed is not None else "",
                     A.tag, A.free_set, str(ck["seen"])])
        print(f"  {t:16}{m:9} base={base:.4f} impose={imp:.4f} "
              f"CE={ce:.4f} {A.tag}={v:.4f} closed={closed:.3f}" if ce is not None else
              f"  {t:16}{m:9} {A.tag}={v:.4f}", flush=True)

HEADER = ["task", "metric", "base_free", "impose_R8", "CE_adapt_R8", "cell_acc", "cell_se",
          "cell_minus_base", "cell_minus_CE", "cell_gap_closed", "CE_gap_closed",
          "cell", "free_set", "train_tokens"]
NOTE = ("# Downstream 10-task 0-shot for free-set cells. Harness, tasks and metric convention match "
        "olmoe_adapt_downstream.csv, whose base_free / impose_R8 / CE_adapt_R8 columns are reused "
        "verbatim rather than recomputed. cell_gap_closed = (cell - impose)/(base_free - impose): "
        "1.0 is free-routing quality, 0.0 is the untrained mask. Higher is better.")

prior = []
if os.path.exists(OUT):
    with open(OUT) as f:
        pr = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    if pr and pr[0] == HEADER:
        prior = [r for r in pr[1:] if r[HEADER.index("cell")] != A.tag]   # replace this cell only
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([NOTE])
    w.writerow(HEADER)
    w.writerows(prior + rows)
print(f"[ds] wrote {OUT}: {len(rows)} new rows for {A.tag}, {len(prior)} kept", flush=True)

accs = [(float(r[HEADER.index("cell_acc")]), float(r[HEADER.index("base_free")]),
         float(r[HEADER.index("impose_R8")]), float(r[HEADER.index("CE_adapt_R8")] or "nan"))
        for r in rows if r[1] == "acc"]
mc = sum(a for a, _, _, _ in accs) / len(accs)
mb = sum(b for _, b, _, _ in accs) / len(accs)
mi = sum(i for _, _, i, _ in accs) / len(accs)
mce = sum(c for _, _, _, c in accs) / len(accs)
print(f"[ds] mean acc over {len(accs)} tasks -- base-free {mb:.4f} | impose-R8 {mi:.4f} | "
      f"CE-adapt-R8 {mce:.4f} | {A.tag} {mc:.4f}", flush=True)
print(f"[ds] mean gap closed -- CE-adapt {(mce - mi) / (mb - mi):.3f} | {A.tag} "
      f"{(mc - mi) / (mb - mi):.3f}", flush=True)
