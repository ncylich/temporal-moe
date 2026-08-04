#!/usr/bin/env python3
"""What does NAIVELY imposing residency cost on real tasks, not just on BPB?

BPB is a held-out likelihood and it is the right instrument for small effects, but it is not what a
deployment cares about. This scores the same ten 0-shot tasks, with the same harness and the same
primary-metric convention as `olmoe_adapt_downstream.csv`, so the Qwen columns sit beside the OLMoE
ones without re-deriving anything.

"Naive" is the point: the constraint is switched on at evaluation time with **no adaptation at all**.
On OLMoE that is catastrophic -- accuracy collapses to near chance on most tasks (arc_easy 0.7715 ->
0.2799) -- and the whole adaptation programme exists to repair it. If Qwen's ~70x cheaper BPB damage
also shows up as tasks surviving untrained imposition, then large-expert-count models can take the
serving win without a training run at all, which is a materially different claim from "cheap in BPB".

Arms, per model:

    free        no constraint. The model as published, and the ceiling everything else is read against.
    R=8         top-k resident. Same absolute rule as the OLMoE runs.
    R=32        12.5% resident on a 256-expert model: matched to OLMoE's FRACTION rather than its R.
                Skipped where E=128 makes R=32 a 25% budget that has no OLMoE counterpart.

BPB on the audited slice is recomputed per arm alongside the tasks, so the two instruments are
reported from the same forward configuration and a mismatch cannot hide.

    qwen_downstream.py --family qwen3 --model /workspace/qwen3moe-adapt/model ...
"""
import sys, types

# Must precede any lm_eval import: its model registry imports every backend eagerly, and the two VLM
# backends reference transformers classes this version removed. Nothing here uses them.
for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)

import argparse, csv, json, os, time                                 # noqa: E402
import torch                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residency as RES                                              # noqa: E402
import residency_qwen as RQ                                          # noqa: E402
from qwen_sweep import score                                         # noqa: E402

from lm_eval.models.huggingface import HFLM                          # noqa: E402
from lm_eval import simple_evaluate                                  # noqa: E402

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "sciq", "boolq", "lambada_openai", "copa"]


def get(d, metric):
    vk = next((k for k in d if k == metric or k.startswith(metric + ",")), None)
    sk = next((k for k in d if k.startswith(metric + "_stderr")), None)
    return (float(d[vk]) if vk else None, float(d[sk]) if sk else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="qwen3_5", choices=("qwen3_5", "qwen3"))
    ap.add_argument("--model", default="/workspace/qwen35-adapt/model")
    ap.add_argument("--data", default="/workspace/qwen35-adapt/data")
    ap.add_argument("--slice-name", default="qwen")
    ap.add_argument("--out", default="/workspace/qwen35-adapt/results")
    ap.add_argument("--tag", default="qwen35")
    ap.add_argument("--arms", default="free,R8,R32")
    ap.add_argument("--bpb-seq", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    meta = json.load(open(f"{A.data}/bpb_slice_meta_{A.slice_name}.json"))
    D = meta["divisor_D"]
    model, tok = RQ.load_model(path=A.model, family=A.family)
    L, E = model.config.num_hidden_layers, model.config.num_experts
    ALL = list(range(L))
    ids = torch.load(f"{A.data}/bpb_slice_ids_{A.slice_name}.pt", weights_only=False)
    bl = [ids[i:i + 1].long() for i in range(A.bpb_seq)]

    ARMS = {"free": (ALL, 8), "R8": (None, 8), "R32": (None, 32)}
    want = [a for a in A.arms.split(",") if a in ARMS]
    results, bpbs = {}, {}
    for arm in want:
        fs, R = ARMS[arm]
        RES._CFG.update(on=True, R=R, evict="min_logit", collect_telem=False)
        RES.set_free_layers(fs)
        bpbs[arm] = score(model, bl, D, fs, R)[0]
        print(f"\n[ds] arm={arm}  R={R}  resident={'100%' if fs else f'{100*R/E:.2f}%'}  "
              f"BPB={bpbs[arm]:.6f}   scoring {len(TASKS)} tasks 0-shot ...", flush=True)
        t0 = time.time()
        # HFLM wraps the already-configured model; the residency hook is global state on the module,
        # so the arm in force during generation is the one set immediately above.
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=A.batch_size)
        results[arm] = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                                       bootstrap_iters=1000)["results"]
        print(f"[ds] arm={arm} done in {(time.time()-t0)/60:.1f} min", flush=True)

    path = os.path.join(A.out, f"{A.tag}_downstream_naive.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# {A.tag} ({A.family}, E={E}, {L} layers): downstream cost of imposing rolling "
                    f"residency with NO adaptation. Ten 0-shot tasks, same harness and metric "
                    f"convention as olmoe_adapt_downstream.csv. 'free' is the published model; R8/R32 "
                    f"switch the constraint on at eval time only. BPB on the audited slice is "
                    f"recomputed per arm from the same configuration. Producer: "
                    f"analysis/ple/qwen_downstream.py"])
        w.writerow(["task", "metric"] + [f"{a}" for a in want] +
                   [f"{a}_stderr" for a in want] +
                   [f"delta_{a}_vs_free" for a in want if a != "free"])
        for t in TASKS:
            for m in ("acc", "acc_norm"):
                vals, errs = [], []
                for a in want:
                    v, s = get(results[a].get(t, {}), m)
                    vals.append(v); errs.append(s)
                if all(v is None for v in vals):
                    continue
                base = vals[want.index("free")] if "free" in want else None
                deltas = [(vals[i] - base) if (base is not None and vals[i] is not None) else None
                          for i, a in enumerate(want) if a != "free"]
                w.writerow([t, m] +
                           [f"{v:.4f}" if v is not None else "" for v in vals] +
                           [f"{s:.4f}" if s is not None else "" for s in errs] +
                           [f"{d:+.4f}" if d is not None else "" for d in deltas])
        w.writerow([])
        w.writerow(["bpb_audited_slice", ""] + [f"{bpbs[a]:.6f}" for a in want])
    print(f"\n[write] {path}", flush=True)
    print("  BPB per arm: " + "  ".join(f"{a}={bpbs[a]:.6f}" for a in want), flush=True)
    print("=== DOWNSTREAM COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
