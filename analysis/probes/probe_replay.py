#!/usr/bin/env python3
"""Tier-1 probe-replay experiments (E1-E8) — offline analysis of router-probe logs, ZERO training.

All numbers/figures come from the already-saved per-token gating logs
(`results/phase0/runs/<run>/router_log.pt`, loaded via plot_probe.load). Everything here is a CPU
replay of a *selection policy* over the logged demand; the trained weights are never touched.

Experiments (see docs/research/mechanism/probe-replay-e1-e8.md):
  E1 swap-rate telemetry + re-reference / victim-cache
  E2 streamed-diversity attribution (union size, residency, effective-experts, token-service)
  E3 mass-weighted consistency (A3) and hit-rate vs set-based
  E4 trigger-margin (tau) hysteresis replay -> swap-rate vs retained-mass tradeoff
  E5 Belady / discounted-oracle / LRU eviction bound (policy headroom)
  E6 per-layer ranking (hit-rate / swap-rate / lifetime)
  E7 EMA-logit smoothing replay (slow-feature routing preview; beta=1 == baseline identity)
  E8 document-boundary attribution (EOD cold-fill contamination)

Run: $PY analysis/probes/probe_replay.py   (regenerates every number + figure)

Convention (matches analysis/plots/plot_probe.py rolling()/overlap()):
  "hit-rate" / "coverage" = fraction of a token's unconstrained top-k demand that is ALREADY
  resident on entry to the token (i.e. before that token's own <=1 swap). This is the cacheability
  metric behind the shipped B/A3 figures (temporal s2 == 36.2%, full MoE == 17.7% at K=k).
  A swap "fires" at a token iff the entering resident set != the token's global top-k (== >=1
  demanded expert is non-resident); the shipped policy then swaps exactly one expert in.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"))  # analysis/plots
from plot_probe import load, topk_ids, OUT, rolling, overlap, sweep, PAIRS, G3  # reuse helpers/paths

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CACHE, ABLATIONS
RAM_RATIO = 32.0                              # r_ram / r (SSD->RAM bandwidth ratio) for s_max budget


def _csv(name, header, rows):
    """Write one tidy CSV of the exact series behind a figure (small; committed to the repo)."""
    import csv
    os.makedirs(ABLATIONS, exist_ok=True)
    with open(f"{ABLATIONS}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", f"{ABLATIONS}/{name}")

# ---- run registry: from MANIFEST.csv + run.meta, never a hardcoded list ----
# The lists this replaces named five runs (tmoe_minlogit_sh1_*, g3_tmoe_s1_1e17,
# flame38m_temporal_minlogit), and NONE of them is in MANIFEST.csv: the router logs behind the
# published e1-e8 numbers were not preserved. The 22 logs that were preserved are a different
# population, so this sweep does not reproduce those numbers, it replaces them. See
# results/ablations/README.md.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry                                                    # noqa: E402

_RUNS = registry.runs(router_log=True, on_disk=True)
_BY_NAME = {r.name: r for r in _RUNS}
ALL_RUNS = [r.name for r in _RUNS]
ALL_TEMPORAL = [r.name for r in _RUNS if r.temporal]
# B10 / C4: unconstrained runs. Replaying the residency policy over an unconstrained model's demand
# is the baseline arm every replay metric was missing; the engine already supports it, the old run
# list simply had no non-temporal entry.
BASELINES = [r.name for r in _RUNS if not r.temporal]


def _headliners(runs):
    """One run per (budget, granularity) cell: the deepest log available, tie-broken toward the
    plainest recipe name so a trigger-shaping variant never stands in for its cell."""
    best = {}
    for r in runs:
        key = (r.budget, r.grain, r.regime)
        cur = best.get(key)
        if cur is None or (len(r.name), r.name) < (len(cur.name), cur.name):
            best[key] = r
    return [r.name for r in sorted(best.values(), key=lambda r: (r.budget, r.grain_label))]


HEADLINERS = _headliners(_RUNS)


def label(run, withN=True):
    """Self-describing figure label. The published labels led with an active-parameter count, which
    run.meta does not record; regime + granularity + budget identifies the cell without a decoder
    ring, and every CSV additionally carries the raw run name and budget (schema convention 3)."""
    r = _BY_NAME.get(run) or registry.get(run)
    grain = ("fine-grained (18 of 192 experts)" if r.grain == 3
             else "coarse (6 of 64 experts)" if r.grain == 1 else "unknown granularity")
    if not withN:
        return grain
    return f"{r.regime} @{r.budget}, {grain}"


def _active_params_m(run):
    """Active non-embedding params in millions, from analysis/shapes.py rather than a hardcoded table.

    The x-axis of the learned-locality figure is model scale, and the column carrying it was lost when
    the opaque `model` label was replaced by run/budget/regime/grain. Deriving it from the shapes table
    is better than the constant it replaces: shapes.py is what run.sh itself prices the FLOP budget
    with, so the number is the same one the run was configured against.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import shapes
    r = _BY_NAME.get(run) or registry.get(run)
    sh = registry.shape_of(run)
    if sh not in shapes.SHAPES:
        return float("nan")
    return shapes.active_nonembed(**shapes.SHAPES[sh], grain=r.grain or 1) / 1e6


def meta_cols(run):
    """The (run, budget, regime, grain) prefix every per-run CSV row carries."""
    r = _BY_NAME.get(run) or registry.get(run)
    return [run, r.budget, r.regime, r.grain_label]


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


def replay(lg, k, evict="min_logit", tau=0.0, prefetch=0, gamma=None, record_swaps=False,
           eval_lg=None, record_resident=False):
    """Roll the shipped K=k, cap-1 residency policy over logged logits [S,B,E].

    evict: 'min_logit' (shipped) | 'lru' | 'belady' (offline-optimal: evict farthest next demand) |
           'discounted' (score = discounted future selection mass y_t(e), gamma set).
    tau:      hysteresis margin (logit space): swap iff best_nonresident > worst_resident + tau.
    prefetch: h>0 -> nominate for demand h tokens in the FUTURE (prescient prefetch bound).
    gamma:    discount for evict='discounted'.
    eval_lg:  if given, coverage (setcov/masscov) is scored against THIS stream's demand/gates
              while the trigger runs on `lg` — use for shaped triggers (EMA/momentum) so coverage
              measures service of the RAW demand, not the shaped stream's own (self-referential).
    Returns dict: setcov[S,B], masscov[S,B] (both PRE-swap), swaprate[S,B] (bool);
    if record_swaps also nominee[S,B], evicted[S,B] expert indices (-1 == no swap);
    if record_resident also resident[S,B,E] bool, the entering resident set at each token.
    """
    S, B, E = lg.shape
    order, dm, w = _prep(lg, k)
    dm_e, w_e = dm, w                  # coverage-scoring stream (defaults to the trigger stream)
    if eval_lg is not None:
        _, dm_e, w_e = _prep(eval_lg, k)   # score coverage vs raw demand; policy stays on lg
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
    nom_rec = evc_rec = res_rec = None
    if record_swaps:
        nom_rec = -np.ones((S, B), np.int32); evc_rec = -np.ones((S, B), np.int32)
    if record_resident:
        res_rec = np.zeros((S, B, E), bool); res_rec[0] = res

    for t in range(1, S):
        lt = lg[t]
        if record_resident:
            res_rec[t] = res                     # entering resident set, before this token's swap
        # coverage measured on ENTRY (pre-swap): demand[t] vs current resident
        setcov[t] = (dm_e[t] & res).sum(1) / k
        masscov[t] = (w_e[t] * res).sum(1)
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
    if record_resident:
        out["resident"] = res_rec
    return out


def replay_run(run, **kw):
    """Run replay over all MoE layers of a run; return per-layer results + k, E."""
    r = load(run); per = {}
    k = E = None
    for ln, rec in r["layers"].items():
        k = rec["k"]; E = rec["logits"].shape[-1]
        per[ln] = replay(rec["logits"], k, **kw)
    return per, k, E


def per_layer(per, skip0=True):
    """{layer: (setcov, masscov)} plus an 'all' key holding the layer-pooled pair.

    B6/B8 ask for these metrics per layer; the pooled value is kept under 'all' so the figures and
    the Belady sanity check keep reading one number without re-deriving it.
    """
    sl = slice(1, None) if skip0 else slice(None)
    out = {ln: (float(p["setcov"][sl].mean()), float(p["masscov"][sl].mean()))
           for ln, p in per.items()}
    out["all"] = (float(np.mean([v[0] for v in out.values()])),
                  float(np.mean([v[1] for v in out.values()])))
    return out


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
    for run in ALL_RUNS:
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
    for run in ALL_RUNS:
        r = load(run)
        unions, effN, pinned, maxres = [], [], [], []
        deepest = max(r["layers"]); pinned_ids = []
        counterfactual = not _BY_NAME[run].temporal
        for ln, rec in r["layers"].items():
            m = rec["mask"]
            if m is None:
                # B10: an unconstrained run logs no resident set because it never had one. Replaying
                # the residency policy over its own demand yields the set it *would* have held --
                # the baseline arm every residency metric was missing.
                m = replay(rec["logits"], rec["k"], evict="min_logit",
                           record_resident=True)["resident"]
            S, B, E = m.shape
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
        summary[run] = dict(E=E, counterfactual=counterfactual,
                            union_mean=float(uni.mean()), union_std=float(uni.std()),
                            union_frac=float(uni.mean() / E), effN=float(np.mean(effN)),
                            pinned=float(np.mean(pinned)), maxres=float(np.max(maxres)),
                            pinned_deep=pinned_ids.tolist())
        s = summary[run]
        print(f"  {label(run):46s} E={E:3d}  union={s['union_mean']:5.1f} "
              f"({s['union_frac']*100:4.1f}% of E)  eff-experts={s['effN']:5.1f}  "
              f"max-residency={s['maxres']*100:4.1f}%  >0.8-resident/layer={s['pinned']:.1f}"
              f"{'   [counterfactual replay: no logged mask]' if counterfactual else ''}")

    # token-service concentration: temporal (resident==served since K=k) vs matched full MoE (top-k)
    conc = {}
    pair = {(r.budget, r.grain): r.name for r in _RUNS if not r.temporal}
    for run in [r.name for r in _RUNS if r.temporal and (r.budget, r.grain) in pair]:
        moe = pair[(_BY_NAME[run].budget, _BY_NAME[run].grain)]
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
    pair = {(r.budget, r.grain): r.name for r in _RUNS if not r.temporal}
    for run in ALL_RUNS:
        per, k, E = replay_run(run, evict="min_logit")
        set_hit = agg(per, "setcov"); mass_hit = agg(per, "masscov")
        # A3 (vs previous active set) == our pre-swap coverage under the shipped policy.
        rows.append((run, "temporal", set_hit, mass_hit))
        rr = _BY_NAME[run]
        moe = pair.get((rr.budget, rr.grain)) if rr.temporal else None
        if moe == run:
            moe = None
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
    for run in ALL_RUNS:
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
    print("\n=== E5  eviction-policy headroom (Belady bound), per layer ===")
    table = {}
    for run in ALL_RUNS:
        res = {}
        res["min_logit"] = per_layer(replay_run(run, evict="min_logit")[0])
        res["LRU"] = per_layer(replay_run(run, evict="lru")[0])
        ts = taustar[run]["tau_budget"]
        res[f"min_logit+tau*({ts})"] = per_layer(replay_run(run, evict="min_logit", tau=ts)[0])
        for g in (0.5, 0.9, 0.95):
            res[f"discounted-oracle(g={g})"] = per_layer(
                replay_run(run, evict="discounted", gamma=g)[0])
        res["Belady"] = per_layer(replay_run(run, evict="belady")[0])
        for h in (1, 4, 16):
            res[f"Belady+prefetch(h={h})"] = per_layer(
                replay_run(run, evict="belady", prefetch=h)[0])
        table[run] = res
        ml = res["min_logit"]["all"][0]; be = res["Belady"]["all"][0]
        assert be >= ml - 1e-6, f"Belady {be} < min_logit {ml} (offline-optimal cannot be worse)"
        print(f"  {label(run)}")
        for name, per in res.items():
            sc, mc = per["all"]
            lay = " ".join(f"L{ln}:{per[ln][0]*100:.0f}" for ln in sorted(k for k in per if k != "all"))
            print(f"      {name:28s} set={sc*100:5.1f}%  mass={mc*100:5.1f}%   {lay}")
    _fig_e5(table)
    return table


# =====================================================================================
#  E6 — per-layer ranking
# =====================================================================================
def e6():
    print("\n=== E6  per-layer ranking ===")
    perlayer = {}
    for run in ALL_RUNS:
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
    print("\n=== E7  EMA-logit smoothing replay, per layer ===")
    betas = [1.0, 0.5, 0.25, 0.1]
    curves = {}          # run -> {beta: {layer|'all': (swaprate, setcov, masscov)}}
    identity_ok = True
    for run in ALL_RUNS:
        r = load(run)
        base = replay_run(run, evict="min_logit")[0]
        base_sr, base_sc = agg(base, "swaps"), agg(base, "setcov")
        byb = {}
        for beta in betas:
            per = {}
            for ln, rec in r["layers"].items():
                k = rec["k"]; lg = _ema(rec["logits"], beta)
                per[ln] = replay(lg, k, evict="min_logit")
            byl = {ln: (float(p["swaps"][1:].mean()), float(p["setcov"][1:].mean()),
                        float(p["masscov"][1:].mean())) for ln, p in per.items()}
            byl["all"] = (agg(per, "swaps"), agg(per, "setcov"), agg(per, "masscov"))
            byb[beta] = byl
        curves[run] = byb
        s1, c1, _ = byb[1.0]["all"]
        ok = abs(s1 - base_sr) < 1e-9 and abs(c1 - base_sc) < 1e-9
        identity_ok &= ok
        s2, c2, _ = byb[0.1]["all"]
        print(f"  {label(run):46s} beta=1 swap={s1:.3f} set={c1*100:4.1f}%  "
              f"beta=0.1 swap={s2:.3f} set={c2*100:4.1f}%  identity(b=1)={'OK' if ok else 'FAIL'}")
    _fig_e7(curves)
    return curves, identity_ok


# =====================================================================================
#  E8 — document-boundary attribution
# =====================================================================================
def e8():
    print("\n=== E8  document-boundary attribution ===")
    out = {}
    for run in ALL_RUNS:
        # Pick the mask by which corpus the run was trained on. Keying off run.meta["tok"] alone
        # silently mis-selects: every 1e19 run has that field empty and would fall to the 16k mask,
        # though 1e19 trains on dclm_tokenized (50k). Only 1e16/1e17 use tok16k_full. Both masks are
        # (64, 2048) with eod_id 0, so a shape check cannot catch the mistake -- e8 would just score
        # boundary churn against boundaries that are not there.
        _tok = _BY_NAME[run].meta.get("tok") or ""
        if "pythia" in _tok:
            batch = "50k"
        elif _BY_NAME[run].budget in ("1e16", "1e17"):
            batch = "16k"
        else:
            batch = "50k"
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


# =====================================================================================
#  A1 (mechanisms plan Tier A) — combined tau-margin x EMA-smoothing trigger, per grain.
#  Composes E4 (tau) and E7 (EMA) exactly as shipped: smooth the trigger's logits with _ema,
#  then replay with a hysteresis margin tau on the smoothed stream. beta=1.0 tau=0 == baseline.
# =====================================================================================
def a1_tau_ema(taus=(0.0, 1.0, 2.0, 4.0), betas=(1.0, 0.5, 0.25, 0.1)):
    print("\n=== A1  combined tau x EMA trigger-shaping replay ===")
    rows = []   # (run, tau, beta, swaprate, setcov, masscov, s_max_budget)
    for run in HEADLINERS:
        r = load(run)
        budget = None
        for beta in betas:
            for tau in taus:
                per = {}
                for ln, rec in r["layers"].items():
                    k = rec["k"]; budget = s_max(k)
                    raw = rec["logits"]
                    per[ln] = replay(_ema(raw, beta), k, evict="min_logit", tau=tau, eval_lg=raw)
                sr, sc, mc = agg(per, "swaps"), agg(per, "setcov"), agg(per, "masscov")
                rows.append((run, tau, beta, sr, sc, mc, budget))
        feasible = [x for x in rows if x[0] == run and x[3] <= budget]
        best = max(feasible, key=lambda x: x[5]) if feasible else None
        if best:
            print(f"  {label(run):40s} budget={budget:.3f}  best-feasible: "
                  f"tau={best[1]:.0f} beta={best[2]:.2f} -> swap={best[3]:.3f} mass={best[5]*100:.1f}%")
        else:
            print(f"  {label(run):40s} budget={budget:.3f}  NO (tau,beta) in this grid reaches budget")
    _csv("a1_tau_ema_grid.csv",
         ["run", "tau", "beta", "swaprate", "setcov", "masscov", "s_max_budget"], rows)
    return rows


# =====================================================================================
#  A2 (mechanisms plan Tier A) — additive momentum trigger score vs pure EMA (A1's winner).
#  score_t = p_t + beta_m * M_{t-1},  M_t = (1-gamma_m) M_{t-1} + gamma_m p_t   (p = softmax(logits))
#  Causal (M_{t-1} only), cold-start M_{-1} = p_0 (no bonus at t=0). Feed the shaped score into
#  the SAME replay() used everywhere else -- only the input array differs from _ema's output.
# =====================================================================================
def _softmax_full(lg):
    z = lg - lg.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _momentum_scores(lg, beta_m, gamma_m, alpha_m=0.0, gamma_q=0.015625, mode="add"):
    # alpha_m>0 = Karen's full double-momentum (slow-EMA penalty Q; anti-pinning). mode="logratio"
    # = popularity-normalized momentum on RAW-LOGIT base (LG1). Mirrors
    # temporal_router.momentum_shaped_scores; keep the two in sync (cross-framework test).
    p = _softmax_full(lg)
    M = np.empty_like(p)
    M[0] = p[0]
    for t in range(1, p.shape[0]):
        M[t] = (1 - gamma_m) * M[t - 1] + gamma_m * p[t]
    if mode == "logratio" or alpha_m > 0:
        Q = np.empty_like(p)
        Q[0] = p[0]
        for t in range(1, p.shape[0]):
            Q[t] = (1 - gamma_q) * Q[t - 1] + gamma_q * p[t]
    if mode == "logratio":
        eps = 1.0 / lg.shape[-1]
        score = lg.astype(np.float32).copy()
        score[1:] = lg[1:] + beta_m * np.log((M[:-1] + eps) / (Q[:-1] + eps))
        return score
    score = p.copy()
    score[1:] = p[1:] + beta_m * M[:-1]
    if alpha_m > 0:
        score[1:] -= alpha_m * Q[:-1]
    return score


def a2_beta_m(gammas=(1 / 8, 1 / 16, 1 / 32), betas=(0.5, 1.0, 2.0)):
    print("\n=== A2  additive-momentum trigger vs pure EMA (A1 winner) ===")
    rows = []   # (run, gamma_m, beta_m, swaprate, setcov, masscov)
    ema_ref = {}  # run -> best EMA-only (tau=0) point at comparable swap rate, from A1's beta grid
    for run in HEADLINERS:
        r = load(run)
        # EMA-only reference curve (tau=0) at the same betas A1 swept, for an apples-to-apples read
        ema_pts = []
        for beta in (1.0, 0.5, 0.25, 0.1):
            per = {}
            for ln, rec in r["layers"].items():
                k = rec["k"]; raw = rec["logits"]
                per[ln] = replay(_ema(raw, beta), k, evict="min_logit", eval_lg=raw)
            ema_pts.append((agg(per, "swaps"), agg(per, "masscov")))
        best_mom = None
        for gm in gammas:
            for bm in betas:
                per = {}
                for ln, rec in r["layers"].items():
                    k = rec["k"]
                    raw = rec["logits"]
                    per[ln] = replay(_momentum_scores(raw, bm, gm), k, evict="min_logit", eval_lg=raw)
                sr, sc, mc = agg(per, "swaps"), agg(per, "setcov"), agg(per, "masscov")
                rows.append((run, gm, bm, sr, sc, mc))
                if best_mom is None or mc > best_mom[3]:
                    best_mom = (gm, bm, sr, mc)
        # compare best momentum point vs the closest-swap-rate EMA-only point
        closest_ema = min(ema_pts, key=lambda x: abs(x[0] - best_mom[2]))
        beats = best_mom[3] > closest_ema[1]
        print(f"  {label(run):40s} momentum best: gamma_m={best_mom[0]:.4f} beta_m={best_mom[1]:.1f} "
              f"swap={best_mom[2]:.3f} mass={best_mom[3]*100:.1f}%  |  "
              f"closest EMA-only swap={closest_ema[0]:.3f} mass={closest_ema[1]*100:.1f}%  "
              f"-> momentum {'WINS' if beats else 'loses'} ({(best_mom[3]-closest_ema[1])*100:+.1f}pt)")
    _csv("a2_momentum_grid.csv",
         ["run", "gamma_m", "beta_m", "swaprate", "setcov", "masscov"], rows)
    return rows


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
    runs = HEADLINERS
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
    for run in HEADLINERS:
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
    runs = HEADLINERS; xs = np.arange(len(runs))
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
    for run in HEADLINERS:
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
    for run, c in zip(HEADLINERS, [f"C{i}" for i in range(len(HEADLINERS))]):
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
        names = list(res.keys()); vals = [res[n]["all"][0] * 100 for n in names]
        ml = res["min_logit"]["all"][0] * 100; be = res["Belady"]["all"][0] * 100
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
    for run, c in zip(HEADLINERS, [f"C{i}" for i in range(len(HEADLINERS))]):
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
    for run, c in zip(HEADLINERS, [f"C{i}" for i in range(len(HEADLINERS))]):
        byb = curves[run]
        betas = np.array(sorted(byb))
        sr = np.array([byb[b]["all"][0] for b in betas])
        sc = np.array([byb[b]["all"][1] for b in betas])
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
def a11_freerider():
    """A11 -- distinct experts per sequence and tokens per expert, over every preserved router log.

    The committed file had four rows carrying labels that decode only by reading a source file, for
    runs that are not in MANIFEST.csv. Regenerated from the registry with run/budget/regime/grain.

    No per-layer breakout, and that is a property of the architecture rather than a shortcut: top-k
    routing assigns exactly k/E of each batch's tokens to each expert on average however those
    assignments are distributed in time, so tokens-per-expert is pinned by (k, E) at every layer. The
    measured column is here to show that it holds, which is what rules out the gradient-noise
    explanation in Appendix A of delexicalization.md. Distinct-experts-per-sequence does vary by layer
    and is reported per layer in e2_streamed_diversity.csv; this file keeps the per-model summary the
    appendix quotes.
    """
    rows = []
    for run in ALL_RUNS:
        r = _BY_NAME[run]
        rec = load(run)
        per_seq, tpe = [], []
        for ln, d in rec["layers"].items():
            lg = d["logits"]; k = d["k"]; E = lg.shape[-1]
            # The served set is top-k of the ROUTED logits: for a temporal run the router selects
            # among residents, so it is top-k of the masked logits, not the intersection of the
            # resident set with the unconstrained demand. That intersection is the cache hit set and
            # is much smaller -- using it made tokens-per-expert read 447 where the architecture pins
            # it at k/E of the batch.
            if d["mask"] is not None:
                routed = np.where(d["mask"], lg, -np.inf)
            else:
                routed = lg
            served = topk_ids(routed, k)               # [S,B,E], exactly k True per token
            per_seq.append(served.any(0).sum(1).astype(float))     # distinct experts per sequence
            tpe.append(served.sum(0).sum(0).astype(float))         # tokens per expert, over the batch
        u = np.concatenate(per_seq); t = np.concatenate(tpe)
        rows.append(meta_cols(run) + [E, k, f"{u.mean():.2f}", f"{u.mean()/E:.4f}",
                                      f"{np.median(t):.1f}", f"{t.mean():.1f}"])
        print(f"  {label(run):46s} E={E:3d} distinct/seq={u.mean():6.2f} "
              f"({u.mean()/E*100:4.1f}% of E)  tokens/expert mean={t.mean():.0f}")
    _csv("mechinterp_freerider.csv",
         ["run", "budget", "regime", "grain", "E", "k", "distinct_experts_per_seq",
          "distinct_frac_of_E", "tokens_per_expert_median", "tokens_per_expert_mean"], rows)
    return rows


def locality_csvs():
    """The two plot_probe.py figures' data. Separated from _export_figure_data so it can be
    regenerated on its own (`--locality-only`) without re-running e1-e8, which costs an hour."""
    # plot_probe.py headline figures (learned-locality-vs-scale + rolling coverage/lifetime vs K).
    # PAIRS/G3 in plot_probe.py name runs whose router logs were never preserved, so the pairing is
    # rebuilt from the registry: each temporal run against the unconstrained run in its own
    # (budget, granularity) cell, blank where no such baseline was preserved.
    pair = {(r.budget, r.grain): r.name for r in _RUNS if not r.temporal}
    rows = []
    for run in HEADLINERS:
        r = _BY_NAME[run]
        rec = next(iter(load(run)["layers"].values())); k, E = rec["k"], rec["logits"].shape[-1]
        mr = pair.get((r.budget, r.grain))
        rows.append(meta_cols(run) + [f"{_active_params_m(run):.2f}",
                                      f"{overlap(run)*100:.2f}",
                                      (f"{overlap(mr)*100:.2f}" if mr and mr != run else ""),
                                      mr or "", f"{100.0*k/E:.2f}"])
    _csv("learned_locality_vs_scale.csv",
         ["run", "budget", "regime", "grain", "active_params_M", "temporal_overlap_pct",
          "full_moe_overlap_pct", "full_moe_run", "random_pct"], rows)
    rows = []
    for run in ALL_RUNS:
        Ks, cov, life, k = sweep(run)
        for K, c, l in zip(Ks, cov, life):
            rows.append(meta_cols(run) + [f"{K/k:.3f}", f"{c:.5f}", f"{l:.3f}"])
    _csv("rolling_coverage_lifetime_vs_K.csv",
         ["run", "budget", "regime", "grain", "resident_cache_K_over_k", "hit_rate",
          "mean_lifetime_tokens"], rows)

def _export_figure_data(e1_rows, e1_victim, e2_summary, e3_rows, e4_curves,
                        e5_table, e6_perlayer, e7_curves, e8_out):
    """Write the small, tidy CSV behind every figure to results/ablations/.
    These are the concise, committed stand-in for the (large) raw router_log.pt tensors."""
    M = meta_cols
    # Schema convention 3: every row carries the raw run name and its budget/regime/granularity, not
    # only a display label. Convention 1: the layer key is written, never pooled away.
    _csv("e1_swap_rate_by_layer.csv",
         ["run", "budget", "regime", "grain", "layer", "mean_swap_rate", "p95_burst_len"],
         [M(r) + [ln, f"{sr:.5f}", f"{p95:.2f}"] for (r, ln, sr, p95) in e1_rows])
    _csv("e1_victim_cache_hitrate.csv",
         ["run", "budget", "regime", "grain", "cache_size_experts", "reload_hitrate"],
         [M(r) + [int(c), f"{h:.5f}"] for r, (sizes, hit) in e1_victim.items()
          for c, h in zip(sizes, hit)])
    _csv("e2_streamed_diversity.csv",
         ["run", "budget", "regime", "grain", "num_experts", "counterfactual_replay",
          "union_mean", "union_std",
          "union_frac_of_E", "effective_experts", "mean_experts_over_0.8_resident_per_layer",
          "max_residency_frac"],
         [M(r) + [s["E"], "yes" if s["counterfactual"] else "no", f"{s['union_mean']:.2f}", f"{s['union_std']:.2f}",
                  f"{s['union_frac']:.4f}", f"{s['effN']:.2f}", f"{s['pinned']:.2f}",
                  f"{s['maxres']:.4f}"] for r, s in e2_summary.items()])
    _csv("e3_mass_vs_set_consistency.csv",
         ["run", "budget", "regime", "grain", "routing", "set_consistency", "mass_consistency"],
         [M(r) + [kind, f"{sh:.5f}", f"{mh:.5f}"] for (r, kind, sh, mh) in e3_rows])
    _csv("e4_swap_vs_retained_mass.csv",
         ["run", "budget", "regime", "grain", "tau", "swap_rate", "retained_mass"],
         [M(r) + [f"{tt:g}", f"{ss:.5f}", f"{mm:.5f}"]
          for r, (taus, sr, rm) in e4_curves.items() for tt, ss, mm in zip(taus, sr, rm)])
    _csv("e5_eviction_policy_headroom.csv",
         ["run", "budget", "regime", "grain", "policy", "layer", "set_coverage", "mass_coverage"],
         [M(r) + [pol, ln, f"{sc:.5f}", f"{mc:.5f}"]
          for r, res in e5_table.items() for pol, per in res.items()
          for ln, (sc, mc) in per.items()])
    _csv("e6_per_layer_ranking.csv",
         ["run", "budget", "regime", "grain", "layer", "hit_rate", "swap_rate", "lifetime_tokens"],
         [M(r) + [ln, f"{d[ln]['hit']:.5f}", f"{d[ln]['swap']:.5f}", f"{d[ln]['life']:.3f}"]
          for r, d in e6_perlayer.items() for ln in sorted(d)])
    _csv("e7_demand_smoothing.csv",
         ["run", "budget", "regime", "grain", "ema_beta", "layer", "swap_rate", "set_coverage",
          "mass_coverage"],
         [M(r) + [f"{bb:g}", ln, f"{ss:.5f}", f"{cc:.5f}", f"{mm:.5f}"]
          for r, per in e7_curves.items() for bb, byl in per.items()
          for ln, (ss, cc, mm) in byl.items()])
    _csv("e8_document_boundary.csv",
         ["run", "budget", "regime", "grain", "batch", "window_tokens", "hit_after_eod",
          "hit_within_doc", "frac_tokens_after", "deficit"],
         [M(r) + [o["batch"], w, f"{dd['hit_after']:.5f}", f"{dd['hit_within']:.5f}",
                  f"{dd['frac_tokens_after']:.5f}", f"{dd['deficit']:.5f}"]
          for r, o in e8_out.items() for w, dd in o["windows"].items()])
    locality_csvs()



def main():
    os.makedirs(OUT, exist_ok=True)
    if "--locality-only" in sys.argv:
        locality_csvs()
        return
    if "--a11-only" in sys.argv:
        print("\n=== A11  free-rider / tokens-per-expert ===")
        a11_freerider()
        return
    if "--a1" in sys.argv:
        a1_tau_ema()
        return
    if "--a2" in sys.argv:
        a2_beta_m()
        return
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
