#!/usr/bin/env python3
"""Tier-1 probe-replay experiments (E1-E8) — offline analysis of router-probe logs, ZERO training.

All numbers/figures come from the already-saved per-token gating logs
(`results/phase0/runs/<run>/router_log.pt`, loaded via plot_probe.load). Everything here is a CPU
replay of a *selection policy* over the logged demand; the trained weights are never touched.

Experiments (see tmp/additional-experiments.md and docs/research/probe-replay-results.md):
  E1 swap-rate telemetry + re-reference / victim-cache
  E2 streamed-diversity attribution (union size, residency, effective-experts, token-service)
  E3 mass-weighted consistency (A3) and hit-rate vs set-based
  E4 trigger-margin (tau) hysteresis replay -> swap-rate vs retained-mass tradeoff
  E5 Belady / discounted-oracle / LRU eviction bound (policy headroom)
  E6 per-layer ranking (hit-rate / swap-rate / lifetime)
  E7 EMA-logit smoothing replay (slow-feature routing preview; beta=1 == baseline identity)
  E8 document-boundary attribution (EOD cold-fill contamination)

Run: .venv/bin/python scripts/phase0/probe_replay.py   (regenerates every number + figure)

Convention (matches scripts/phase0/plot_probe.py rolling()/overlap()):
  "hit-rate" / "coverage" = fraction of a token's unconstrained top-k demand that is ALREADY
  resident on entry to the token (i.e. before that token's own <=1 swap). This is the cacheability
  metric behind the shipped B/A3 figures (temporal s2 == 36.2%, full MoE == 17.7% at K=k).
  A swap "fires" at a token iff the entering resident set != the token's global top-k (== >=1
  demanded expert is non-resident); the shipped policy then swaps exactly one expert in.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_probe import load, topk_ids, OUT, rolling, overlap, sweep, PAIRS, G3  # reuse helpers/paths

CACHE = "/workspace/FLAME-MoE/results/phase0/probe_batch_cache"
FIGDATA = "/workspace/FLAME-MoE/results/phase0/figure_data"   # small CSVs behind every figure
RAM_RATIO = 32.0                              # r_ram / r (SSD->RAM bandwidth ratio) for s_max budget


def _csv(name, header, rows):
    """Write one tidy CSV of the exact series behind a figure (small; committed to the repo)."""
    import csv
    os.makedirs(FIGDATA, exist_ok=True)
    with open(f"{FIGDATA}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", f"{FIGDATA}/{name}")

# ---- run registry (active non-embedding params in millions; matched full-MoE pair; grain) ----
META = {
    "tmoe_minlogit_sh1_s0_1e16": dict(N=1.36,  moe="v16k_d_s0_1e16",     grain=1, batch="16k"),
    "tmoe_minlogit_sh1_s2_1e17": dict(N=8.12,  moe="v16k_sweep_s2_1e17", grain=1, batch="16k"),
    "tmoe_minlogit_sh1_s3_1e17": dict(N=14.77, moe="v16k_sweep_s3_1e17", grain=1, batch="16k"),
    "g3_tmoe_s1_1e17":           dict(N=3.91,  moe=None,                 grain=3, batch="16k"),
    "flame38m_temporal_minlogit":dict(N=38.0,  moe=None,                 grain=1, batch="50k"),
}
ALL_TEMPORAL = ["tmoe_minlogit_sh1_s0_1e16", "tmoe_minlogit_sh1_s2_1e17",
                "tmoe_minlogit_sh1_s3_1e17", "g3_tmoe_s1_1e17", "flame38m_temporal_minlogit"]
HEADLINERS   = ["tmoe_minlogit_sh1_s2_1e17", "g3_tmoe_s1_1e17", "flame38m_temporal_minlogit"]


def label(run, withN=True):
    """De-jargoned figure label: active-param count + expert-grain description."""
    m = META[run]; N = m["N"]
    Ns = f"{N:.1f}M active" if N < 100 else f"{N:.0f}M active"
    grain = "fine-grained (18 of 192 experts)" if m["grain"] == 3 else "coarse (6 of 64 experts)"
    return f"{Ns}, {grain}" if withN else grain


# =====================================================================================
#  Replay engine — vectorised over the batch dimension B (mirrors compute_resident_mask).
#  Operates on one MoE layer's logits [S, B, E]. K = k, at most one swap per token.
# =====================================================================================
def _prep(lg, k):
    """Per-token demand top-k mask and softmax gate weights (softmax over the selected k logits)."""
    S, B, E = lg.shape
    order = np.argsort(-lg, axis=2)
    topk = order[:, :, :k]
    dm = np.zeros((S, B, E), bool); np.put_along_axis(dm, topk, True, 2)
    tl = np.take_along_axis(lg, topk, 2)
    tl = tl - tl.max(2, keepdims=True); ew = np.exp(tl); w = ew / ew.sum(2, keepdims=True)
    wfull = np.zeros((S, B, E), np.float32); np.put_along_axis(wfull, topk, w.astype(np.float32), 2)
    return order, dm, wfull


def replay(lg, k, evict="min_logit", tau=0.0, prefetch=0, gamma=None, record_swaps=False):
    """Roll the shipped K=k, cap-1 residency policy over logged logits [S,B,E].

    evict: 'min_logit' (shipped) | 'lru' | 'belady' (offline-optimal: evict farthest next demand) |
           'discounted' (score = discounted future selection mass y_t(e), gamma set).
    tau:      hysteresis margin (logit space): swap iff best_nonresident > worst_resident + tau.
    prefetch: h>0 -> nominate for demand h tokens in the FUTURE (prescient prefetch bound).
    gamma:    discount for evict='discounted'.
    Returns dict: setcov[S,B], masscov[S,B] (both PRE-swap), swaprate[S,B] (bool);
    if record_swaps also nominee[S,B], evicted[S,B] expert indices (-1 == no swap).
    """
    S, B, E = lg.shape
    order, dm, w = _prep(lg, k)
    NEG, POS = -np.inf, np.inf

    fut = Y = None
    if evict == "belady":
        fut = np.empty((S, B, E), np.float32); nextpos = np.full((B, E), POS, np.float32)
        for t in range(S - 1, -1, -1):
            fut[t] = nextpos
            sel = order[t, :, :k]; np.put_along_axis(nextpos, sel, np.float32(t), 1)
    if evict == "discounted":
        Y = np.zeros((S, B, E), np.float32)
        for t in range(S - 2, -1, -1):
            Y[t] = dm[t + 1].astype(np.float32) + gamma * Y[t + 1]

    res = np.zeros((B, E), bool)
    top0 = order[0, :, :k]; np.put_along_axis(res, top0, True, 1)
    refresh = np.full((B, E), NEG, np.float32)
    rank = np.arange(k - 1, -1, -1, dtype=np.float32)[None].repeat(B, 0)
    np.put_along_axis(refresh, top0, rank, 1)

    bidx = np.arange(B)
    setcov = np.empty((S, B), np.float32); masscov = np.empty((S, B), np.float32)
    swaps = np.zeros((S, B), bool)
    setcov[0] = 1.0; masscov[0] = 1.0
    nom_rec = evc_rec = None
    if record_swaps:
        nom_rec = -np.ones((S, B), np.int32); evc_rec = -np.ones((S, B), np.int32)

    for t in range(1, S):
        lt = lg[t]
        # coverage measured on ENTRY (pre-swap): demand[t] vs current resident
        setcov[t] = (dm[t] & res).sum(1) / k
        masscov[t] = (w[t] * res).sum(1)
        if evict == "discounted":
            yt = Y[t]
            nom_i = np.where(res, NEG, yt).argmax(1)
            nom_key = yt[bidx, nom_i]
            worst_i = np.where(res, yt, POS).argmin(1)
            do_swap = nom_key > yt[bidx, worst_i] + tau
            evict_i = worst_i
        else:
            src = lg[min(t + prefetch, S - 1)] if prefetch else lt
            nom_i = np.where(res, NEG, src).argmax(1)
            nom_val = src[bidx, nom_i]
            worst_val = np.where(res, src, POS).min(1)
            do_swap = nom_val > worst_val + tau
            if evict == "lru":
                ekey = refresh
            elif evict == "belady":
                ekey = -fut[t]            # evict farthest next demand -> smallest -fut
            else:
                ekey = src                # min_logit (shipped)
            evict_i = np.where(res, ekey, POS).argmin(1)
        swaps[t] = do_swap
        sb = bidx[do_swap]
        if len(sb):
            res[sb, evict_i[do_swap]] = False
            res[sb, nom_i[do_swap]] = True
            refresh[sb, nom_i[do_swap]] = t
            if record_swaps:
                nom_rec[t, sb] = nom_i[do_swap]; evc_rec[t, sb] = evict_i[do_swap]
    out = dict(setcov=setcov, masscov=masscov, swaps=swaps)
    if record_swaps:
        out["nominee"] = nom_rec; out["evicted"] = evc_rec
    return out


def replay_run(run, **kw):
    """Run replay over all MoE layers of a run; return per-layer results + k, E."""
    r = load(run); per = {}
    k = E = None
    for ln, rec in r["layers"].items():
        k = rec["k"]; E = rec["logits"].shape[-1]
        per[ln] = replay(rec["logits"], k, **kw)
    return per, k, E


def agg(per, key, skip0=True):
    """Mean of a per-layer [S,B] metric over layers, tokens (t>=1), batch."""
    vals = [p[key][1:].mean() if skip0 else p[key].mean() for p in per.values()]
    return float(np.mean(vals))


# =====================================================================================
#  E1 — swap-rate telemetry (feasibility margin)
# =====================================================================================
def s_max(k):
    """Bandwidth budget: swaps/token hideable behind the same layer's resident compute."""
    return (k - 1) / RAM_RATIO


def e1():
    print("\n=== E1  swap-rate telemetry ===")
    rows = []            # (run, layer, swaprate, p95burst)
    victim = {}          # run -> (sizes, hitrates)
    for run in ALL_TEMPORAL:
        per, k, E = replay_run(run, record_swaps=True)
        budget = s_max(k)
        # per-layer swap rate + burst run-lengths
        all_incoming = []   # (relative re-reference distance) across layer/batch
        for ln, p in per.items():
            sr = float(p["swaps"][1:].mean())
            # burst lengths: consecutive swapping tokens (per batch element)
            bursts = []
            sw = p["swaps"]
            for b in range(sw.shape[1]):
                run_len = 0
                for t in range(sw.shape[0]):
                    if sw[t, b]:
                        run_len += 1
                    elif run_len:
                        bursts.append(run_len); run_len = 0
                if run_len:
                    bursts.append(run_len)
            p95 = float(np.percentile(bursts, 95)) if bursts else 0.0
            rows.append((run, ln, sr, p95))
            # re-reference distance: when an expert is swapped IN, how long since it was evicted
            nom = p["nominee"]; evc = p["evicted"]; S, B = nom.shape
            for b in range(B):
                last_evict = {}
                for t in range(S):
                    e_in = nom[t, b]
                    if e_in >= 0:
                        d = t - last_evict[e_in] if e_in in last_evict else np.inf
                        all_incoming.append(d)
                    e_out = evc[t, b]
                    if e_out >= 0:
                        last_evict[e_out] = t
        # victim cache: hit iff the swapped-in expert was among the last C distinct evicted experts
        inc = np.array(all_incoming, float)
        sizes = np.array([0, 1, 2, 4, 8, 16, 32])
        # a size-C victim cache holds the C most-recently evicted DISTINCT experts; an incoming
        # expert is a hit iff its re-reference distance places it within those C (approx by counting
        # distinct evictions since it was last evicted -> we recompute exactly per (b))
        hitrate = _victim_hitrate(per, sizes)
        victim[run] = (sizes, hitrate)
        mean_sr = np.mean([sr for r_, l_, sr, _ in rows if r_ == run])
        print(f"  {label(run):40s} k={k:2d}  mean swaps/tok={mean_sr:.3f}  budget s_max={budget:.3f}"
              f"  finite re-ref frac={np.isfinite(inc).mean():.2f}")

    _fig_e1_swaprate(rows)
    _fig_e1_victim(victim)
    return rows, victim


def _victim_hitrate(per, sizes):
    """Exact victim-cache hit-rate: incoming expert present among last-C distinct evicted."""
    hits = np.zeros(len(sizes)); tot = 0
    for p in per.values():
        nom = p["nominee"]; evc = p["evicted"]; S, B = nom.shape
        for b in range(B):
            recent = []           # MRU-ordered distinct recently-evicted experts
            for t in range(S):
                e_in = nom[t, b]
                if e_in >= 0:
                    tot += 1
                    for j, C in enumerate(sizes):
                        if C > 0 and e_in in recent[:C]:
                            hits[j] += 1
                e_out = evc[t, b]
                if e_out >= 0:
                    if e_out in recent:
                        recent.remove(e_out)
                    recent.insert(0, e_out)
    return hits / max(tot, 1)


# =====================================================================================
#  E2 — streamed-diversity attribution (from the LOGGED resident sets)
# =====================================================================================
def e2():
    print("\n=== E2  streamed-diversity attribution ===")
    summary = {}          # run -> dict
    resid_dists = {}      # run -> per-expert residency fractions (deepest layer) for the plot
    for run in ALL_TEMPORAL:
        r = load(run)
        unions, effN, pinned, maxres = [], [], [], []
        deepest = max(r["layers"]); pinned_ids = []
        for ln, rec in r["layers"].items():
            m = rec["mask"]; S, B, E = m.shape          # logged resident set actually used
            # union size per sequence (distinct experts resident at any t)
            u = m.any(0).sum(1)                          # [B]
            unions.append(u.astype(float))
            # per-expert residency fraction, averaged over batch
            resid = m.mean((0, 1))                       # [E] fraction of tokens resident
            effN.append(np.exp(-(resid / resid.sum() * np.log(resid / resid.sum() + 1e-12)).sum()))
            pinned.append(int((resid > 0.8).sum()))
            maxres.append(float(resid.max()))
            if ln == deepest:
                resid_dists[run] = np.sort(m.mean(0).mean(0))[::-1]  # sorted residency, deepest layer
                pinned_ids = np.where(resid > 0.8)[0]
        uni = np.concatenate(unions)
        summary[run] = dict(E=E, union_mean=float(uni.mean()), union_std=float(uni.std()),
                            union_frac=float(uni.mean() / E), effN=float(np.mean(effN)),
                            pinned=float(np.mean(pinned)), maxres=float(np.max(maxres)),
                            pinned_deep=pinned_ids.tolist())
        s = summary[run]
        print(f"  {label(run):40s} E={E:3d}  union={s['union_mean']:5.1f} "
              f"({s['union_frac']*100:4.1f}% of E)  eff-experts={s['effN']:5.1f}  "
              f"max-residency={s['maxres']*100:4.1f}%  >0.8-resident/layer={s['pinned']:.1f}")

    # token-service concentration: temporal (resident==served since K=k) vs matched full MoE (top-k)
    conc = {}
    for run in ["tmoe_minlogit_sh1_s2_1e17"]:
        moe = META[run]["moe"]
        t_serv = _service_counts(run, use_mask=True)
        m_serv = _service_counts(moe, use_mask=False)
        conc[run] = (t_serv, m_serv, moe)
        print(f"  token-service Gini  temporal={_gini(t_serv):.3f}  full-MoE={_gini(m_serv):.3f} "
              f"({label(run)})")

    _fig_e2_union(summary)
    _fig_e2_residency(resid_dists)
    return summary, conc


def _service_counts(run, use_mask):
    r = load(run); counts = []
    for rec in r["layers"].values():
        if use_mask and rec["mask"] is not None:
            counts.append(rec["mask"].sum((0, 1)))
        else:
            counts.append(topk_ids(rec["logits"], rec["k"]).sum((0, 1)))
    return np.concatenate([c / c.sum() for c in counts])   # normalised service share


def _gini(x):
    x = np.sort(x); n = len(x); i = np.arange(1, n + 1)
    return float((2 * (i * x).sum() / (n * x.sum())) - (n + 1) / n)


# =====================================================================================
#  E3 — mass-weighted consistency (A3) and hit-rate vs set-based
# =====================================================================================
def e3():
    print("\n=== E3  mass-weighted consistency & coverage ===")
    rows = []
    for run in ALL_TEMPORAL:
        per, k, E = replay_run(run, evict="min_logit")
        set_hit = agg(per, "setcov"); mass_hit = agg(per, "masscov")
        # A3 (vs previous active set) == our pre-swap coverage under the shipped policy.
        rows.append((run, "temporal", set_hit, mass_hit))
        moe = META[run]["moe"]
        if moe:
            sm, mm = _moe_mass_a3(moe)
            rows.append((run, "full MoE", sm, mm))
            print(f"  {label(run):40s} temporal set={set_hit*100:4.1f}% mass={mass_hit*100:4.1f}%"
                  f"   MoE set={sm*100:4.1f}% mass={mm*100:4.1f}%")
        else:
            print(f"  {label(run):40s} temporal set={set_hit*100:4.1f}% mass={mass_hit*100:4.1f}%")
    _fig_e3(rows)
    return rows


def _moe_mass_a3(moe):
    """Full-MoE self-consistency: overlap of top-k(t) with top-k(t-1), set- and mass-weighted."""
    r = load(moe); sset, smass = [], []
    for rec in r["layers"].values():
        lg = rec["logits"]; k = rec["k"]
        _, dm, w = _prep(lg, k)
        prev = dm[:-1]                                    # previous top-k set
        sset.append((dm[1:] & prev).sum(2).mean() / k)
        smass.append((w[1:] * prev).sum(2).mean())
    return float(np.mean(sset)), float(np.mean(smass))


# =====================================================================================
#  E4 — trigger-margin (tau) hysteresis replay
# =====================================================================================
def e4():
    print("\n=== E4  trigger-margin (tau) replay ===")
    taus = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    curves = {}; taustar = {}
    for run in HEADLINERS:
        sr, rm = [], []
        for tau in taus:
            per, k, E = replay_run(run, evict="min_logit", tau=tau)
            sr.append(agg(per, "swaps")); rm.append(agg(per, "masscov"))
        sr = np.array(sr); rm = np.array(rm)
        curves[run] = (np.array(taus), sr, rm)
        # tau*: max retained mass with swap-rate <= 0.5 (working target for G3)
        ok = np.where(sr <= 0.5)[0]
        ts = taus[ok[0]] if len(ok) else taus[-1]
        # tau that halves the baseline swap rate
        half = sr[0] / 2
        thalf = np.interp(-half, -sr, taus)              # monotone-ish; interpolate on decreasing sr
        taustar[run] = dict(tau_budget=ts, sr_at_budget=float(sr[min(ok[0] if len(ok) else -1, len(sr)-1)]),
                            tau_half=float(thalf), baseline_sr=float(sr[0]),
                            baseline_mass=float(rm[0]))
        print(f"  {label(run):40s} baseline swap={sr[0]:.3f} mass={rm[0]*100:4.1f}%  "
              f"tau*(s<=0.5)={ts}  swap@tau*={taustar[run]['sr_at_budget']:.3f}  "
              f"tau(halve swap)={thalf:.2f}")
    _fig_e4(curves)
    return curves, taustar


# =====================================================================================
#  E5 — Belady / discounted-oracle / LRU eviction bound
# =====================================================================================
def e5(taustar):
    print("\n=== E5  eviction-policy headroom (Belady bound) ===")
    table = {}
    for run in HEADLINERS:
        res = {}
        base = replay_run(run, evict="min_logit")[0]
        res["min_logit"] = (agg(base, "setcov"), agg(base, "masscov"))
        lru = replay_run(run, evict="lru")[0]
        res["LRU"] = (agg(lru, "setcov"), agg(lru, "masscov"))
        ts = taustar[run]["tau_budget"]
        tau = replay_run(run, evict="min_logit", tau=ts)[0]
        res[f"min_logit+tau*({ts})"] = (agg(tau, "setcov"), agg(tau, "masscov"))
        for g in (0.5, 0.9, 0.95):
            d = replay_run(run, evict="discounted", gamma=g)[0]
            res[f"discounted-oracle(g={g})"] = (agg(d, "setcov"), agg(d, "masscov"))
        bel = replay_run(run, evict="belady")[0]
        res["Belady"] = (agg(bel, "setcov"), agg(bel, "masscov"))
        for h in (1, 4, 16):
            bp = replay_run(run, evict="belady", prefetch=h)[0]
            res[f"Belady+prefetch(h={h})"] = (agg(bp, "setcov"), agg(bp, "masscov"))
        table[run] = res
        ml = res["min_logit"][0]; be = res["Belady"][0]
        assert be >= ml - 1e-6, f"Belady {be} < min_logit {ml} (offline-optimal cannot be worse)"
        print(f"  {label(run)}")
        for name, (sc, mc) in res.items():
            print(f"      {name:28s} set={sc*100:5.1f}%  mass={mc*100:5.1f}%")
    _fig_e5(table)
    return table


# =====================================================================================
#  E6 — per-layer ranking
# =====================================================================================
def e6():
    print("\n=== E6  per-layer ranking ===")
    perlayer = {}
    for run in HEADLINERS:
        per, k, E = replay_run(run, evict="min_logit", record_swaps=False)
        lifes = _lifetimes(run)
        d = {}
        for ln, p in per.items():
            d[ln] = dict(hit=float(p["setcov"][1:].mean()), swap=float(p["swaps"][1:].mean()),
                         life=lifes[ln])
        perlayer[run] = d
        order = sorted(d, key=lambda l: d[l]["hit"])
        print(f"  {label(run)}")
        for ln in sorted(d):
            print(f"      layer {ln}: hit={d[ln]['hit']*100:4.1f}%  swap={d[ln]['swap']:.3f}  "
                  f"lifetime={d[ln]['life']:.2f}")
    _fig_e6(perlayer)
    return perlayer


def _lifetimes(run):
    r = load(run); out = {}
    for ln, rec in r["layers"].items():
        lg = rec["logits"]; k = rec["k"]
        ls = [rolling(lg[:, b], k, k)[1] for b in range(min(8, lg.shape[1]))]
        out[ln] = float(np.mean(ls))
    return out


# =====================================================================================
#  E7 — EMA-logit smoothing replay (slow-feature routing preview)
# =====================================================================================
def _ema(lg, beta):
    if beta >= 1.0:
        return lg
    out = np.empty_like(lg); out[0] = lg[0]
    for t in range(1, lg.shape[0]):
        out[t] = (1 - beta) * out[t - 1] + beta * lg[t]
    return out


def e7():
    print("\n=== E7  EMA-logit smoothing replay ===")
    betas = [1.0, 0.5, 0.25, 0.1]
    curves = {}
    identity_ok = True
    for run in HEADLINERS:
        r = load(run)
        sr, sc, mc = [], [], []
        # baseline (no smoothing) for identity check
        base = replay_run(run, evict="min_logit")[0]
        base_sr, base_sc = agg(base, "swaps"), agg(base, "setcov")
        for beta in betas:
            per = {}
            for ln, rec in r["layers"].items():
                k = rec["k"]; lg = _ema(rec["logits"], beta)
                per[ln] = replay(lg, k, evict="min_logit")
            sr.append(agg(per, "swaps")); sc.append(agg(per, "setcov")); mc.append(agg(per, "masscov"))
        curves[run] = (np.array(betas), np.array(sr), np.array(sc), np.array(mc))
        ok = abs(sr[0] - base_sr) < 1e-9 and abs(sc[0] - base_sc) < 1e-9
        identity_ok &= ok
        print(f"  {label(run):40s} beta=1 swap={sr[0]:.3f} set={sc[0]*100:4.1f}%  "
              f"beta=0.1 swap={sr[-1]:.3f} set={sc[-1]*100:4.1f}%  identity(b=1)={'OK' if ok else 'FAIL'}")
    _fig_e7(curves)
    return curves, identity_ok


# =====================================================================================
#  E8 — document-boundary attribution
# =====================================================================================
def e8():
    print("\n=== E8  document-boundary attribution ===")
    out = {}
    for run in HEADLINERS:
        batch = META[run]["batch"]
        eodfile = f"{CACHE}/eod_{batch}.npy"
        if not os.path.exists(eodfile):
            print(f"  [skip] {run}: EOD cache {eodfile} missing"); continue
        eod = np.load(eodfile)                          # [B, S] bool
        per, k, E = replay_run(run, evict="min_logit")
        S = eod.shape[1]
        windows = [4, 16, 64]
        res = {}
        # swaps saturate near 1.0 everywhere, so we use the GRADED hit-rate (setcov) deficit:
        # how much lower is coverage in the w tokens after a boundary vs mid-document.
        for w in windows:
            ha = _split_rate(per, eod, S, w, "setcov")
            res[w] = dict(hit_after=ha[0], hit_within=ha[1], frac_tokens_after=ha[2],
                          deficit=ha[1] - ha[0])
        # within-document-only headline numbers (exclude tokens within 16 of a boundary)
        wd = _within_doc_headline(per, eod, S, 16)
        out[run] = dict(windows=res, within_doc=wd, batch=batch)
        r4, r16 = res[4], res[16]
        print(f"  {label(run):40s} [{batch} batch]  hit-rate after-EOD: "
              f"w=4 {r4['hit_after']*100:.1f}% vs within {r4['hit_within']*100:.1f}% "
              f"(deficit {r4['deficit']*100:+.1f}pt) | w=16 {r16['hit_after']*100:.1f}% vs "
              f"{r16['hit_within']*100:.1f}%  ({r16['frac_tokens_after']*100:.1f}% of tokens)")
        print(f"      within-document-only hit-rate={wd['set']*100:.1f}%  (all-token {wd['set_all']*100:.1f}%)")
    _fig_e8(out)
    return out


def _split_rate(per, eod, S, w, key):
    after_n = after_d = within_n = within_d = 0.0
    for p in per.values():
        arr = {"miss": (p["setcov"] < 1.0), "setcov": p["setcov"], "swaps": p["swaps"]}[key]
        B = arr.shape[1]
        for b in range(min(B, eod.shape[0])):
            after = np.zeros(S, bool)
            for pp in np.where(eod[b])[0]:
                after[pp + 1:min(pp + 1 + w, S)] = True
            after[0] = False
            wd = ~after; wd[0] = False
            after_n += arr[after, b].sum(); after_d += after.sum()
            within_n += arr[wd, b].sum(); within_d += wd.sum()
    return (after_n / max(after_d, 1), within_n / max(within_d, 1), after_d / (after_d + within_d))


def _within_doc_headline(per, eod, S, w):
    a_set = a_swap = a_n = t_set = t_swap = t_n = 0.0
    for p in per.values():
        sc = p["setcov"]; sw = p["swaps"]; B = sc.shape[1]
        for b in range(min(B, eod.shape[0])):
            after = np.zeros(S, bool)
            for pp in np.where(eod[b])[0]:
                after[pp + 1:min(pp + 1 + w, S)] = True
            keep = ~after; keep[0] = False
            a_set += sc[keep, b].sum(); a_swap += sw[keep, b].sum(); a_n += keep.sum()
            allk = np.ones(S, bool); allk[0] = False
            t_set += sc[allk, b].sum(); t_swap += sw[allk, b].sum(); t_n += allk.sum()
    return dict(set=a_set / a_n, swap=a_swap / a_n, set_all=t_set / t_n, swap_all=t_swap / t_n)


# =====================================================================================
#  Figures (house style: descriptive names, de-jargoned labels, bottom caption)
# =====================================================================================
def _cap(fig, text):
    fig.text(0.5, 0.01, text, ha="center", fontsize=8, wrap=True)


def _save(fig, name):
    fig.tight_layout(rect=[0, 0.09, 1, 1]); path = f"{OUT}/{name}"
    fig.savefig(path, dpi=140); plt.close(fig); print("wrote", path)


def _fig_e1_swaprate(rows):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    runs = ALL_TEMPORAL
    xs = np.arange(len(runs))
    for i, run in enumerate(runs):
        srs = [sr for r_, l_, sr, _ in rows if r_ == run]
        ax.scatter([i] * len(srs), np.array(srs), c="C2", s=45, zorder=3,
                   label="per-layer realized swap rate" if i == 0 else None)
        ax.scatter([i], [np.mean(srs)], c="black", marker="_", s=800, zorder=4,
                   label="mean over layers" if i == 0 else None)
    for k, lab, c in [(6, "coarse k=6 budget", "C0"), (18, "fine-grained k=18 budget", "C3"),
                      (32, "k=32 budget", "C1")]:
        ax.axhline(s_max(k), ls="--", c=c, lw=1.3, label=f"{lab} = {s_max(k):.2f}")
        ax.axhline(2 * s_max(k), ls=":", c=c, lw=1.0)
    ax.set_xticks(xs); ax.set_xticklabels([label(r).replace(", ", "\n") for r in runs], fontsize=8)
    ax.set_ylabel("realized swaps per token  (lower = cheaper to stream)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Realized expert-swap rate is near 1 per token — far above the SSD->RAM bandwidth budget")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=7, loc="center right")
    _cap(fig, "Realized swaps per token under the shipped rolling-residency policy (keep top-k experts "
              "resident, swap <=1 per token, evict lowest-logit), per Mixture-of-Experts layer (green) "
              "and layer-mean (black). Dashed lines = s_max = (k-1)/32, the swaps/token that can be "
              "hidden behind the same layer's resident-expert compute (SSD->RAM bandwidth ratio ~32); "
              "dotted = the ~2x budget when the router is computed early (before attention). Lower is "
              "cheaper; points far above the budget mean streaming cannot be fully hidden as-is.")
    _save(fig, "swap_rate_vs_bandwidth_budget.png")


def _fig_e1_victim(victim):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for run in ALL_TEMPORAL:
        sizes, hr = victim[run]
        ax.plot(sizes, hr * 100, "o-", label=label(run))
    ax.set_xlabel("victim-cache size  (number of recently-evicted experts kept in RAM)")
    ax.set_ylabel("share of swap-ins served from the victim cache  (%)  — higher = less SSD traffic")
    ax.set_title("A small victim cache of recently-evicted experts absorbs a large share of expert re-loads")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
    _cap(fig, "For each expert swapped in under the shipped policy, whether it was among the last C "
              "distinct experts evicted (a 'victim cache' of size C, in experts). y = fraction of "
              "swap-ins that hit this cache (would avoid an SSD re-load); higher is better. "
              "'temporal' = rolling residency (keep top-k resident, swap 1 per token).")
    _save(fig, "victim_cache_hitrate_vs_size.png")


def _fig_e2_union(summary):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    runs = ALL_TEMPORAL; xs = np.arange(len(runs))
    um = [summary[r]["union_mean"] for r in runs]; us = [summary[r]["union_std"] for r in runs]
    Es = [summary[r]["E"] for r in runs]; en = [summary[r]["effN"] for r in runs]
    ax.bar(xs - 0.2, um, 0.4, yerr=us, color="C2", label="distinct experts used over the sequence (union)")
    ax.bar(xs + 0.2, en, 0.4, color="C4", label="effective experts (exp of residency entropy)")
    ax.plot(xs, Es, "k_", ms=28, mew=2, label="total experts E (RAM would need all of these)")
    ax.set_xticks(xs); ax.set_xticklabels([label(r).replace(", ", "\n") for r in runs], fontsize=8)
    ax.set_ylabel("number of experts")
    ax.set_title("Temporal MoE streams far more experts over a sequence than fit in its resident set")
    ax.grid(True, ls=":", alpha=0.3, axis="y"); ax.legend(fontsize=8)
    _cap(fig, "Per 2048-token sequence, distinct experts that are resident at some point (green, +/-1 SD "
              "over sequences) and the effective expert count = exp(entropy of the residency "
              "distribution) (purple), vs the total expert pool E (black). Union near E => streamed "
              "diversity is real (the model uses far more experts over time than the top-k resident); "
              "union small => it collapsed to a static small pool. 'temporal' = rolling residency.")
    _save(fig, "streamed_expert_diversity_per_sequence.png")


def _fig_e2_residency(resid_dists):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for run in ALL_TEMPORAL:
        d = resid_dists[run]
        ax.plot(np.arange(len(d)) / (len(d) - 1), d, label=label(run))
    ax.axhline(0.8, ls="--", c="gray", lw=1, label="pinned threshold (resident >80% of tokens)")
    ax.set_xlabel("experts ranked by residency (0 = most resident, 1 = least), deepest MoE layer")
    ax.set_ylabel("fraction of tokens the expert is resident  (0-1)")
    ax.set_title("A few experts are de-facto pinned (resident almost always); most are streamed briefly")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
    _cap(fig, "Per-expert residency fraction (share of the 2048 tokens for which an expert is in the "
              "resident set), sorted descending, at the deepest Mixture-of-Experts layer. Experts above "
              "the dashed line (>0.8) are de-facto pinned and are candidates for an explicit pinned "
              "slot. 'temporal' = rolling residency (keep top-k resident, swap 1 per token).")
    _save(fig, "expert_residency_distribution.png")


def _fig_e3(rows):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    temp = [(label(r), s, m) for r, kind, s, m in rows if kind == "temporal"]
    xs = np.arange(len(temp))
    ax.bar(xs - 0.2, [t[1] * 100 for t in temp], 0.4, color="C2",
           label="set self-consistency (share of demanded experts already resident)")
    ax.bar(xs + 0.2, [t[2] * 100 for t in temp], 0.4, color="C1",
           label="gate-mass self-consistency (share of demanded gate weight already resident)")
    ax.set_xticks(xs); ax.set_xticklabels([t[0].replace(", ", "\n") for t in temp], fontsize=8)
    ax.set_ylabel("self-consistency  (%)  — higher = more of what the token wants is already cached")
    ax.set_title("Gate mass is far more self-consistent than raw set membership: the heavy top experts stay resident")
    ax.grid(True, ls=":", alpha=0.3, axis="y"); ax.legend(fontsize=8)
    _cap(fig, "For the temporal (rolling-residency) models: at each token, share of its top-k demand "
              "already resident on entry (before this token's swap), counted by set membership (green) "
              "vs weighted by softmax gate mass (orange). Mass >> set means the high-weight top-1/2 "
              "experts are resident far more often than the tail; higher is better.")
    _save(fig, "gate_mass_vs_set_self_consistency.png")


def _fig_e4(curves):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for run, c in zip(HEADLINERS, ["C2", "C3", "C4"]):
        taus, sr, rm = curves[run]
        ax.plot(sr, rm * 100, "o-", color=c, label=label(run))
        for tv, x, y in zip(taus, sr, rm):
            ax.annotate(f"{tv:g}", (x, y * 100), fontsize=6, alpha=0.6)
    ax.axvline(0.5, ls="--", c="gray", lw=1, label="deployable swap budget (s=0.5)")
    ax.set_xlabel("realized swaps per token  (lower = cheaper)")
    ax.set_ylabel("retained gate mass  (%)  — higher = more demand still served")
    ax.set_title("Dropping marginal swaps via a hysteresis margin is nearly free: quality-per-swap is concave")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
    _cap(fig, "Deploy-time hysteresis knob tau (annotated, logit units): swap only if the best "
              "non-resident expert beats the worst resident by more than tau. x = realized swaps/token "
              "(lower cheaper), y = retained gate mass (share of top-k gate weight still resident, "
              "higher better). A near-flat top-left arm means many swaps can be cut with little mass "
              "loss. 'temporal' = rolling residency.")
    _save(fig, "swap_rate_vs_retained_mass_tradeoff.png")


def _fig_e5(table):
    order = ["LRU", "min_logit", None, "discounted-oracle(g=0.9)", "Belady", "Belady+prefetch(h=16)"]
    fig, axes = plt.subplots(1, len(HEADLINERS), figsize=(14, 5.2), sharey=True)
    for ax, run in zip(axes, HEADLINERS):
        res = table[run]
        names = list(res.keys()); vals = [res[n][0] * 100 for n in names]
        ml = res["min_logit"][0] * 100; be = res["Belady"][0] * 100
        colors = ["C1" if "Belady" in n else "gray" if n == "LRU" else
                  "C3" if "discounted" in n else "C0" if "tau" in n else "C2" for n in names]
        ax.barh(np.arange(len(names)), vals, color=colors)
        ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.axvline(ml, ls="--", c="C2", lw=1); ax.axvline(be, ls="--", c="C1", lw=1)
        ax.set_title(label(run), fontsize=9); ax.set_xlabel("set hit-rate (%)")
        ax.invert_yaxis(); ax.grid(True, ls=":", alpha=0.3, axis="x")
    fig.suptitle("Offline-optimal (Belady) eviction barely beats the shipped least-logit policy: eviction "
                 "learning has little headroom at K=k", fontsize=11)
    _cap(fig, "Pre-swap set hit-rate (share of a token's top-k demand already resident; higher better) "
              "under eviction policies replayed on the same logged demand, K=k, one swap/token. "
              "min_logit (green dashed) = shipped; Belady (orange dashed) = offline-optimal upper bound "
              "(evict the resident whose next demand is farthest ahead); discounted-oracle = exact "
              "learned-lookahead bound; +prefetch(h) = allow the swap to fire h tokens early. A small "
              "Belady-minus-min_logit gap means a smarter/learned eviction policy can buy little.")
    fig.tight_layout(rect=[0, 0.11, 1, 0.95]); path = f"{OUT}/eviction_policy_headroom_belady_bound.png"
    fig.savefig(path, dpi=140); plt.close(fig); print("wrote", path)


def _fig_e6(perlayer):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for run, c in zip(HEADLINERS, ["C2", "C3", "C4"]):
        d = perlayer[run]; lns = sorted(d)
        ax.plot(lns, [d[l]["hit"] * 100 for l in lns], "o-", color=c, label=label(run))
    ax.set_xlabel("Mixture-of-Experts layer number (shallow -> deep)")
    ax.set_ylabel("pre-swap hit-rate  (%)  — higher = more locally consistent routing")
    ax.set_title("Routing locality grows with depth: shallow MoE layers are the least cacheable")
    ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)
    _cap(fig, "Per-layer pre-swap hit-rate (share of each token's top-k demand already resident on "
              "entry) under the shipped rolling-residency policy; higher = more locally consistent. "
              "Deeper layers cache better, so any non-uniform budget (per-layer resident size, "
              "pinning, or margin) should spend most on the shallow layers.")
    _save(fig, "per_layer_routing_locality_ranking.png")


def _fig_e7(curves):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    for run, c in zip(HEADLINERS, ["C2", "C3", "C4"]):
        betas, sr, sc, mc = curves[run]
        ax1.plot(betas, sr, "o-", color=c, label=label(run))
        ax2.plot(sr, sc * 100, "o-", color=c, label=label(run))
        for b, x, y in zip(betas, sr, sc):
            ax2.annotate(f"b={b:g}", (x, y * 100), fontsize=6, alpha=0.6)
    ax1.set_xlabel("EMA weight on the current token, beta (1 = no smoothing)")
    ax1.set_ylabel("realized swaps per token")
    ax1.set_title("Smoothing the router lowers swap rate"); ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(fontsize=7); ax1.invert_xaxis()
    ax2.set_xlabel("realized swaps per token (lower = cheaper)")
    ax2.set_ylabel("pre-swap hit-rate (%) — higher = better")
    ax2.set_title("...but hit-rate falls with it (selection replayed on an un-smoothed-trained model)")
    ax2.grid(True, ls=":", alpha=0.4); ax2.legend(fontsize=7)
    fig.suptitle("Exponential-moving-average routing preview: fewer swaps, but coverage drops in lockstep "
                 "(indicative only)", fontsize=11)
    _cap(fig, "Selection replayed on smoothed logits logits'_t=(1-beta)logits'_{t-1}+beta*logits_t "
              "(linear router => equals smoothing hidden states). Left: swaps/token vs beta. Right: "
              "hit-rate vs swaps/token trace. beta=1 reproduces the baseline exactly (harness identity "
              "check). Caveat: the model was trained WITHOUT smoothing, so gains are indicative, not a "
              "quality measurement.")
    fig.tight_layout(rect=[0, 0.1, 1, 0.94]); path = f"{OUT}/demand_smoothing_swap_vs_coverage.png"
    fig.savefig(path, dpi=140); plt.close(fig); print("wrote", path)


def _fig_e8(out):
    runs = [r for r in HEADLINERS if r in out]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    xs = np.arange(len(runs))
    ax.bar(xs - 0.2, [out[r]["windows"][4]["hit_after"] * 100 for r in runs], 0.4, color="C3",
           label="hit-rate <=4 tokens after a document boundary")
    ax.bar(xs + 0.2, [out[r]["windows"][4]["hit_within"] * 100 for r in runs], 0.4, color="C0",
           label="hit-rate mid-document")
    for i, r in enumerate(runs):
        d = out[r]["windows"][4]["deficit"] * 100
        ax.annotate(f"{-d:+.1f}pt", (i, out[r]["windows"][4]["hit_after"] * 100 + 1), ha="center",
                    fontsize=8, color="C3")
    ax.set_xticks(xs); ax.set_xticklabels([label(r).replace(", ", "\n") for r in runs], fontsize=8)
    ax.set_ylabel("pre-swap hit-rate  (%)  — share of demanded experts already resident")
    ax.set_title("Document-boundary cold-fill costs only a few points of hit-rate (a probe packing artifact)")
    ax.grid(True, ls=":", alpha=0.3, axis="y"); ax.legend(fontsize=8); ax.set_ylim(0, 55)
    _cap(fig, "The probe packs several documents per 2048-token sequence; residency only cold-fills at "
              "t=0, so packed-document boundaries (end-of-document token) inject topic shifts absent at "
              "deployment (each request starts fresh). Bars: pre-swap hit-rate (share of a token's top-k "
              "demand already resident; higher better) in the 4 tokens after a boundary (red) vs "
              "mid-document (blue); annotation = the boundary penalty. A small deficit => boundaries are "
              "not a major contaminant of the reported locality statistics.")
    _save(fig, "document_boundary_churn.png")


# =====================================================================================
def _export_figure_data(e1_rows, e1_victim, e2_summary, e3_rows, e4_curves,
                        e5_table, e6_perlayer, e7_curves, e8_out):
    """Write the small, tidy CSV behind every figure to results/phase0/figure_data/.
    These are the concise, committed stand-in for the (large) raw router_log.pt tensors."""
    _csv("e1_swap_rate_by_layer.csv", ["model", "layer", "mean_swap_rate", "p95_burst_len"],
         [[label(r), ln, f"{sr:.5f}", f"{p95:.2f}"] for (r, ln, sr, p95) in e1_rows])
    _csv("e1_victim_cache_hitrate.csv", ["model", "cache_size_experts", "reload_hitrate"],
         [[label(r), int(c), f"{h:.5f}"] for r, (sizes, hit) in e1_victim.items()
          for c, h in zip(sizes, hit)])
    _csv("e2_streamed_diversity.csv",
         ["model", "num_experts", "union_mean", "union_std", "union_frac_of_E",
          "effective_experts", "mean_experts_over_0.8_resident_per_layer", "max_residency_frac"],
         [[label(r), s["E"], f"{s['union_mean']:.2f}", f"{s['union_std']:.2f}",
           f"{s['union_frac']:.4f}", f"{s['effN']:.2f}", f"{s['pinned']:.2f}", f"{s['maxres']:.4f}"]
          for r, s in e2_summary.items()])
    _csv("e3_mass_vs_set_consistency.csv", ["model", "routing", "set_consistency", "mass_consistency"],
         [[label(r), kind, f"{sh:.5f}", f"{mh:.5f}"] for (r, kind, sh, mh) in e3_rows])
    _csv("e4_swap_vs_retained_mass.csv", ["model", "tau", "swap_rate", "retained_mass"],
         [[label(r), f"{tt:g}", f"{ss:.5f}", f"{mm:.5f}"]
          for r, (taus, sr, rm) in e4_curves.items() for tt, ss, mm in zip(taus, sr, rm)])
    _csv("e5_eviction_policy_headroom.csv", ["model", "policy", "set_coverage", "mass_coverage"],
         [[label(r), pol, f"{sc:.5f}", f"{mc:.5f}"]
          for r, res in e5_table.items() for pol, (sc, mc) in res.items()])
    _csv("e6_per_layer_ranking.csv", ["model", "layer", "hit_rate", "swap_rate", "lifetime_tokens"],
         [[label(r), ln, f"{d[ln]['hit']:.5f}", f"{d[ln]['swap']:.5f}", f"{d[ln]['life']:.3f}"]
          for r, d in e6_perlayer.items() for ln in sorted(d)])
    _csv("e7_demand_smoothing.csv", ["model", "ema_beta", "swap_rate", "set_coverage", "mass_coverage"],
         [[label(r), f"{bb:g}", f"{ss:.5f}", f"{cc:.5f}", f"{mm:.5f}"]
          for r, (betas, sr, sc, mc) in e7_curves.items() for bb, ss, cc, mm in zip(betas, sr, sc, mc)])
    _csv("e8_document_boundary.csv",
         ["model", "batch", "window_tokens", "hit_after_eod", "hit_within_doc",
          "frac_tokens_after", "deficit"],
         [[label(r), o["batch"], w, f"{dd['hit_after']:.5f}", f"{dd['hit_within']:.5f}",
           f"{dd['frac_tokens_after']:.5f}", f"{dd['deficit']:.5f}"]
          for r, o in e8_out.items() for w, dd in o["windows"].items()])
    # plot_probe.py headline figures (learned-locality-vs-scale + rolling coverage/lifetime vs K)
    rows = []
    for tag, N, tr, mr in list(PAIRS) + [G3]:
        rec = next(iter(load(tr)["layers"].values())); k, E = rec["k"], rec["logits"].shape[-1]
        rows.append([label(tr), f"{N:g}", f"{overlap(tr)*100:.2f}",
                     (f"{overlap(mr)*100:.2f}" if mr else ""), f"{100.0*k/E:.2f}"])
    _csv("learned_locality_vs_scale.csv",
         ["model", "active_params_M", "temporal_overlap_pct", "full_moe_overlap_pct", "random_pct"], rows)
    rows = []
    for r in ALL_TEMPORAL + ["v16k_sweep_s2_1e17"]:
        Ks, cov, life, k = sweep(r)
        for K, c, l in zip(Ks, cov, life):
            rows.append([label(r) if r in META else "full MoE, 8.1M active",
                         f"{K/k:.3f}", f"{c:.5f}", f"{l:.3f}"])
    _csv("rolling_coverage_lifetime_vs_K.csv",
         ["model", "resident_cache_K_over_k", "hit_rate", "mean_lifetime_tokens"], rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    e1_rows, e1_victim = e1()
    e2_summary, _ = e2()
    e3_rows = e3()
    e4_curves, taustar = e4()
    e5_table = e5(taustar)
    e6_perlayer = e6()
    e7_curves, ident = e7()
    print(f"\n[E7 identity check] beta=1.0 reproduces baseline exactly: {'PASS' if ident else 'FAIL'}")
    e8_out = e8()
    _export_figure_data(e1_rows, e1_victim, e2_summary, e3_rows, e4_curves,
                        e5_table, e6_perlayer, e7_curves, e8_out)
    print("\nprobe_replay complete.")


if __name__ == "__main__":
    main()
