#!/usr/bin/env python3
"""OLMoE downstream under naive residency, with the gate-mass artifact removed.

The published OLMoE imposition row (mean accuracy 0.3116, at the random-guessing floor) was produced
with `gate_mass="renorm"`: masking non-resident experts to -inf and taking the softmax over the R
residents. OLMoE sets `norm_topk_prob=False`, so its gate weights are the raw softmax-over-64
probabilities -- summing to ~0.40 over the top-8 unmasked and to exactly 1.0 after masking. That
scales every Mixture-of-Experts block output by ~2.5x across 16 layers, on top of the routing change
residency is meant to test, and re-measurement showed it accounts for 91.6% of the BPB damage
(+2.0014 as published vs +0.1690 corrected).

So the task collapse in that row is very likely an activation blow-up rather than a routing effect,
but "very likely" is inference from BPB. This measures it directly, same ten tasks and harness as
the era table olmoe_adapt_downstream.csv (archived: results/archive/olmoe_wrong_renorm),
so the corrected row drops into the published table.

Two arms only, which is what the comparison needs:

    free   every layer exempt -- the router hook returns the shipped forward untouched, so this arm
           is bit-identical to the published model and is unaffected by the artifact either way.
    R8     residency on all 16 layers at R=k=8 (12.5% resident), gate mass PRESERVED: the top-k is
           selected from the masked distribution but weighted from the unmasked one, so residency
           changes which experts serve and not how much they contribute.

    olmoe_downstream_naive.py
"""
import sys, types

for _m in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
    sys.modules[_m] = types.ModuleType(_m)

import argparse, csv, json, os, time                                 # noqa: E402
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


def get(d, m):
    vk = next((k for k in d if k == m or k.startswith(m + ",")), None)
    sk = next((k for k in d if k.startswith(m + "_stderr")), None)
    return (float(d[vk]) if vk else None, float(d[sk]) if sk else None)


@torch.no_grad()
def bpb(model, ids, divisor, n):
    tot = ntok = 0
    for i in range(n):
        b = ids[i:i + 1].to("cuda").long()
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        for i0 in range(0, lg.shape[1], 512):
            sl = lg[:, i0:i0 + 512].float()
            tot += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tg[:, i0:i0 + 512].reshape(-1), reduction="sum"))
            del sl
        ntok += tg.numel(); del lg
    return (tot / ntok) / divisor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bpb-seq", type=int, default=16)
    ap.add_argument("--gate-mass", default="preserve", choices=("preserve", "renorm"))
    A = ap.parse_args()

    meta = json.load(open(f"{DATA_DIR}/bpb_slice_meta.json"))
    D = meta["divisor_D"]
    model, tok = RES.load_model()
    L = model.config.num_hidden_layers
    ids = torch.load(f"{DATA_DIR}/bpb_slice_ids.pt", weights_only=False)[: A.bpb_seq]
    print(f"  OLMoE E={model.config.num_experts} k={model.config.num_experts_per_tok} layers={L} "
          f"norm_topk_prob={model.config.norm_topk_prob} gate_mass={A.gate_mass}", flush=True)

    ARMS = {"free": list(range(L)), "R8": None}
    res, bpbs = {}, {}
    for arm, fs in ARMS.items():
        RES._CFG.update(on=True, R=8, evict="min_logit", collect_telem=False, gate_mass=A.gate_mass)
        RES.set_free_layers(fs)
        bpbs[arm] = bpb(model, ids, D, A.bpb_seq)
        torch.cuda.empty_cache()
        print(f"\n[ds] arm={arm} resident={'100%' if fs else '12.5%'} BPB={bpbs[arm]:.6f} "
              f"scoring {len(TASKS)} tasks ...", flush=True)
        t0 = time.time()
        bs = int(A.batch_size) if A.batch_size.isdigit() else A.batch_size
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
        res[arm] = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0,
                                   limit=(A.limit or None), bootstrap_iters=1000)["results"]
        print(f"[ds] arm={arm} done in {(time.time()-t0)/60:.1f} min", flush=True)

    path = os.path.join(ABLATIONS, f"olmoe_downstream_naive_{A.gate_mass}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# OLMoE-1B-7B: downstream cost of imposing rolling residency at R=8 of 64 "
                    f"(12.5% resident) with NO adaptation, gate_mass={A.gate_mass}. 'preserve' "
                    f"selects the top-k from the masked distribution but takes weights from the "
                    f"unmasked one, so residency changes which experts serve and not how much they "
                    f"contribute; 'renorm' is the published behaviour, which on this "
                    f"norm_topk_prob=False model also rescales gate mass ~0.40 -> 1.0 and accounts "
                    f"for 91.6% of the measured BPB damage. Ten 0-shot tasks, same harness as "
                    f"the archived era table (results/archive/olmoe_wrong_renorm). "
                    f"Producer: analysis/residency/olmoe_downstream_naive.py"])
        w.writerow(["task", "metric", "free", "R8", "free_stderr", "R8_stderr", "delta_R8_vs_free"])
        for t in TASKS:
            for m in ("acc", "acc_norm"):
                fv, fs_ = get(res["free"].get(t, {}), m)
                rv, rs_ = get(res["R8"].get(t, {}), m)
                if fv is None and rv is None:
                    continue
                w.writerow([t, m, f"{fv:.4f}" if fv is not None else "",
                            f"{rv:.4f}" if rv is not None else "",
                            f"{fs_:.4f}" if fs_ is not None else "",
                            f"{rs_:.4f}" if rs_ is not None else "",
                            f"{rv-fv:+.4f}" if (fv is not None and rv is not None) else ""])
        w.writerow([])
        w.writerow(["bpb_audited_slice", "", f"{bpbs['free']:.6f}", f"{bpbs['R8']:.6f}"])
    print(f"\n[write] {path}", flush=True)
    print(f"  BPB: free={bpbs['free']:.6f}  R8={bpbs['R8']:.6f}  damage={bpbs['R8']-bpbs['free']:+.6f}",
          flush=True)
    print("=== OLMOE DOWNSTREAM COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
