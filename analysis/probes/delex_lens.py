#!/usr/bin/env python3
"""delex-1e19 Part 1 (d): data-weighted logit lens -> mechinterp_lens_1e19.csv.

For each routed expert (layers 2-4), take its routed-token-averaged output v_e (from the capture),
project through the final RMSNorm + unembedding U, softmax -> p_e, effective vocab = exp(H(p_e)).
 - variant 'weighted': v_e = mean expert output over routed tokens (data-conditioned).
 - variant 'static'  : v_e = uniform mean of the expert's fc2 weight columns (no data) -> the
                       no-signal reference (should read ~vocab size, 50k here).
Schema: label,run,layer,expert,variant,n_tokens,eff_vocab,dispersion   (dispersion = H(p)/ln V).
"""
import os, sys, csv, re, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_read

ROOT = "/workspace/FLAME-MoE"; RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/mechinterp_lens_1e19.csv")
LAYERS = [2, 3, 4]
CELLS = [("moe_coarse_1e19", "moe_coarse_1e19"),
         ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
         ("temporal_fine_1e19", "temporal_fine_g3_1e19")]
EPS = 1e-6


def rmsnorm(v, g):
    return v / torch.sqrt((v * v).mean(-1, keepdim=True) + EPS) * g


def eff_vocab(v, gain, U):
    """v [E,H] -> eff_vocab[E], dispersion[E] via softmax(U @ rmsnorm(v))."""
    x = rmsnorm(v.double(), gain.double())              # [E,H]
    logits = x @ U.double().T                            # [E, V]
    logits = logits - logits.max(-1, keepdim=True).values
    p = torch.softmax(logits, -1)
    H = -(p * p.clamp(min=1e-20).log()).sum(-1)          # [E]
    V = U.shape[0]
    return torch.exp(H), H / np.log(V)


def main():
    rows = []
    for label, run in CELLS:
        cap = os.path.join(RUNS, run, "delex_capture.pt")
        if not os.path.exists(cap):
            print(f"[skip] {label}: no capture"); continue
        d = torch.load(cap, map_location="cpu", weights_only=False)
        ip = ckpt_read.iter_dir(os.path.join(RUNS, run, "ckpt"))
        # unembedding + final norm + per-layer expert fc2
        need = ["output_layer.weight", "decoder.final_layernorm.weight"]
        fc2 = {L: f"decoder.layers.{L}.mlp.experts.experts.linear_fc2.weight" for L in LAYERS}
        w = ckpt_read.load(ip, need + list(fc2.values()))
        U = w["output_layer.weight"].float(); gain = w["decoder.final_layernorm.weight"].float()
        for L in LAYERS:
            if L not in d["layers"]:
                continue
            Ld = d["layers"][L]
            v_w = (Ld["out_sum"] / Ld["out_cnt"].clamp(min=1).unsqueeze(1)).float()   # [E,H] weighted
            cnt = Ld["out_cnt"]
            ev_w, disp_w = eff_vocab(v_w, gain, U)
            # static: uniform mean of fc2 columns [H, ffn] -> [E,H]
            f2 = w[fc2[L]].float()                                                     # [E,H,ffn]
            v_s = f2.mean(-1)                                                          # [E,H]
            ev_s, disp_s = eff_vocab(v_s, gain, U)
            for e in range(v_w.shape[0]):
                rows.append([label, run, L, e, "weighted", int(cnt[e]),
                             round(float(ev_w[e]), 1), round(float(disp_w[e]), 4)])
                rows.append([label, run, L, e, "static", int(cnt[e]),
                             round(float(ev_s[e]), 1), round(float(disp_s[e]), 4)])
        wv = [r for r in rows if r[0] == label and r[4] == "weighted"]
        evs = sorted(r[6] for r in wv)
        med = evs[len(evs) // 2]; dec = evs[max(0, len(evs) // 10)]
        st = [r[6] for r in rows if r[0] == label and r[4] == "static"]
        print(f"[ok] {label}: weighted median eff_vocab={med:.0f} sharpest_decile={dec:.0f} "
              f"static(no-signal ref) median={sorted(st)[len(st)//2]:.0f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["label", "run", "layer", "expert", "variant", "n_tokens",
                                         "eff_vocab", "dispersion"]); wr.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
