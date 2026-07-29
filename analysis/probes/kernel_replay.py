#!/usr/bin/env python3
"""Eviction-as-temporal-filter replay (K1-K4) — offline, numpy-only, committed data.

Companion measurements for docs/research/mechanism/lru-as-convolution.md, which reads the
rolling-residency policy as a temporal filter over each expert's demand history:

    eviction key_t(e) = (h * d_e)(t)          d_e(t) = 1[e in the token's unconstrained top-k]

Under that reading the two shipped policies are two fixed kernels: `min_logit` is a width-1
(instantaneous) kernel on the score, and `lru` -- which refreshes only on ADMISSION, never on use --
is a box kernel of width k on the admission stream, i.e. a FIFO queue. This script measures the
kernel that the shipped policy actually realises, and sweeps kernel shape/width to locate the
optimum.

Unlike probe_replay.py (E1-E8) this needs NO router_log.pt: it runs off the committed
results/ablations/expert_selection_per_token.csv -- 220 tokens x the deepest MoE layer x sequence 0
for three models -- which records, per token, the unconstrained top-k demand and the resident set
the shipped min_logit policy actually used. That is a thin slice, so every number here is
INDICATIVE (promote-only, per the repo's epistemology: free replays promote ideas, never falsify
them). Block-bootstrap CIs are reported so the thin slice cannot be over-read.

Experiments:
  K1  effective kernel of the shipped policy -- age-at-eviction (in admissions) and its survival
      curve, against the box_k that `lru` would impose.
  K2  demand renewal statistics -- inter-demand interval CDF, and the box-kernel coverage
      prediction cov = F(W) it implies for a residency window of W tokens.
  K3  eviction-kernel replay -- same demand, same nomination stream, different eviction kernel.
  K4  admission-stream autocorrelation -- how much of the demand signal survives the cap-1 sampler.

Run: $PY analysis/probes/kernel_replay.py
"""
import os
import sys
import csv
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                       # noqa: E402

SRC = os.path.join(ABLATIONS, "expert_selection_per_token.csv")
DEMAND = "temporal (unconstrained preference)"                    # unconstrained top-k per token
RESIDENT = "temporal (resident set used)"                         # what the shipped policy served
BLOCK = 32                                                        # moving-block bootstrap width
NBOOT = 2000
RNG = np.random.default_rng(1234)


# ---------------------------------------------------------------------------- data
def load():
    """-> {model_M: dict(D=[set]*T, R=[set]*T, k, E, layer)} from the committed raster CSV."""
    panels = collections.defaultdict(dict)
    meta = {}
    with open(SRC) as f:
        for r in csv.DictReader(f):
            m = int(float(r["active_params_M"]))
            panels[(m, r["panel"].strip())].setdefault(int(r["token"]), set()).add(int(r["expert"]))
            meta[m] = (int(r["num_experts_E"]), int(r["topk_k"]), r["moe_layer"])
    out = {}
    for m, (E, k, layer) in sorted(meta.items()):
        d, res = panels[(m, DEMAND)], panels[(m, RESIDENT)]
        out[m] = dict(D=[d[t] for t in sorted(d)], R=[res[t] for t in sorted(res)],
                      k=k, E=E, layer=layer)
    return out


def label(m):
    return f"{m}M active, coarse (6 of 64 experts)"


def admissions(R):
    """Logged admission/eviction trace: adm[t], evi[t] (or None). Asserts the cap-1 invariant."""
    adm, evi = [None], [None]
    for t in range(1, len(R)):
        a, e = R[t] - R[t - 1], R[t - 1] - R[t]
        assert len(a) == len(e) <= 1, f"cap-1 violated at t={t}: +{a} -{e}"
        adm.append(next(iter(a)) if a else None)
        evi.append(next(iter(e)) if e else None)
    return adm, evi


def block_ci(x, stat=np.mean, nboot=NBOOT, block=BLOCK):
    """Moving-block bootstrap 95% CI — the per-token series is strongly autocorrelated, so an
    iid bootstrap would understate the interval by several-fold."""
    x = np.asarray(x, float)
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    vals = np.empty(nboot)
    for i in range(nboot):
        idx = RNG.choice(starts, nb)
        vals[i] = stat(np.concatenate([x[s:s + block] for s in idx])[:n])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------- K1
def k1_effective_kernel(data):
    """Age-at-eviction of the shipped min_logit policy, in ADMISSION events (the FIFO clock).

    `lru` (= FIFO) evicts at age exactly k by construction, so its survival curve is the box
    1[a < k]. Everything the shipped policy does differently is visible as departure from that.
    """
    print("\n=== K1  effective residency kernel of the shipped min_logit policy ===")
    rows, surv_rows = [], []
    for m, d in data.items():
        R, k = d["R"], d["k"]
        adm, evi = admissions(R)
        clock = 0
        born = {e: 0 for e in R[0]}                  # cold fill = k pseudo-admissions at clock 0
        born_tok = {e: 0 for e in R[0]}
        # The raster CSV records the cold-fill SET but not its logit order, so all k cold-fill
        # experts are dated 0 (they are ~3% of the evictions in this window).
        ages, ages_tok = [], []
        for t in range(1, len(R)):
            if adm[t] is not None:
                clock += 1                       # eviction and admission are the same event, so
            if evi[t] is not None:               # the clock ticks first: FIFO then evicts at
                ages.append(clock - born.get(evi[t], 0))   # age exactly k (test_temporal_router).
                ages_tok.append(t - born_tok.get(evi[t], 0))
            if adm[t] is not None:
                born[adm[t]] = clock
                born_tok[adm[t]] = t
        # age composition of the live resident set: FIFO holds exactly one slot of each age
        # 0..k-1 at all times, so any departure from a flat profile is the kernel allocating
        # residency time unequally across slots.
        clock2 = 0
        born2 = {e: 0 for e in R[0]}
        live_old = []
        for t in range(1, len(R)):
            if adm[t] is not None:
                clock2 += 1
                born2[adm[t]] = clock2
            live_old.append(sum(1 for e in R[t] if clock2 - born2.get(e, 0) >= k))
        old_frac = float(np.mean(live_old)) / k

        ages = np.array(ages)
        at = np.array(ages_tok)
        lo, hi = block_ci(ages)
        print(f"  {label(m):34s} evictions={len(ages)}  age-at-evict (admissions): "
              f"mean={ages.mean():.2f} [{lo:.2f},{hi:.2f}] median={np.median(ages):.0f} "
              f"p90={np.percentile(ages, 90):.0f} max={ages.max()}  "
              f"P(age<2)={np.mean(ages < 2):.2f}  P(age==k)={np.mean(ages == k):.3f}  "
              f"slots older than k admissions={old_frac * 100:.0f}% (FIFO: 0%)")
        rows.append([label(m), len(ages), f"{ages.mean():.4f}", f"{lo:.4f}", f"{hi:.4f}",
                     f"{np.median(ages):.1f}", f"{np.percentile(ages, 90):.1f}", int(ages.max()),
                     f"{np.mean(ages < 2):.4f}", f"{np.mean(ages == k):.4f}",
                     f"{at.mean():.4f}", f"{np.median(at):.1f}", int(at.max()),
                     f"{old_frac:.4f}"])
        for a in range(0, 33):
            surv_rows.append([label(m), a, f"{np.mean(ages > a):.4f}", 1 if a < k else 0])
    _csv("k1_effective_kernel.csv",
         ["model", "evictions", "mean_age_admissions", "ci_lo", "ci_hi", "median_age",
          "p90_age", "max_age", "frac_age_lt_2", "frac_age_eq_k",
          "mean_age_tokens", "median_age_tokens", "max_age_tokens",
          "frac_slots_older_than_k_admissions"], rows)
    _csv("k1_kernel_survival.csv",
         ["model", "age_admissions", "shipped_survival", "fifo_box_survival"], surv_rows)
    return rows


# ---------------------------------------------------------------------------- K2
def k2_renewal(data):
    """Inter-demand intervals, and the box-kernel coverage they predict.

    A resident expert admitted on demand survives W tokens under a width-W box kernel, so its next
    demand is a hit iff the inter-demand interval is <= W:  cov_box(W) = F(W). This turns a
    residency budget into a coverage prediction with no policy simulation at all.
    """
    print("\n=== K2  demand renewal intervals and the box-kernel coverage law ===")
    rows = []
    for m, d in data.items():
        D, k, E = d["D"], d["k"], d["E"]
        last = {}
        gaps = []
        for t, s in enumerate(D):
            for e in s:
                if e in last:
                    gaps.append(t - last[e])
                last[e] = t
        gaps = np.array(gaps)
        F = lambda w: float(np.mean(gaps <= w))                              # noqa: E731
        print(f"  {label(m):34s} intervals={len(gaps)} mean={gaps.mean():.1f} "
              f"median={np.median(gaps):.0f}  F(1)={F(1):.3f} F(k={k})={F(k):.3f} "
              f"F(16)={F(16):.3f} F(64)={F(64):.3f}")
        for w in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96):
            rows.append([label(m), w, f"{F(w):.4f}"])
        rows.append([label(m), "mean_interval", f"{gaps.mean():.4f}"])
        rows.append([label(m), "median_interval", f"{np.median(gaps):.4f}"])
    _csv("k2_demand_renewal.csv", ["model", "window_tokens_or_stat", "value"], rows)
    return rows


# ---------------------------------------------------------------------------- K3
def _nominate_selfnom(res, Dt, score):
    """Self-consistent nomination without logits: among this token's demanded non-residents, take
    the one with the largest recent-demand EMA (ties -> lowest index). Used to check that K3's
    ranking is not an artifact of replaying min_logit's own admission trace."""
    cand = sorted(Dt - res)
    if not cand:
        return None
    return max(cand, key=lambda e: (score[e], -e))


def replay(d, evict, W=8, gamma=0.125, gamma_slow=0.03125, w_slow=1.0,
           protocol="trace", ema_gamma=0.125):
    """Roll one eviction kernel over the logged demand.

    protocol="trace"   -> admit exactly what the shipped router admitted (identical input signal,
                          different kernel: the controlled comparison). If that expert is already
                          resident in this arm, no swap fires.
    protocol="selfnom" -> nominate from this arm's own state (see _nominate_selfnom).

    evict:
      "fifo"   oldest admission            -- the shipped `lru`; box_k on the admission stream
      "lrd"    oldest LAST DEMAND          -- textbook LRU (refresh on use, not on admission)
      "boxW"   fewest demands in last W    -- box kernel of width W on the demand stream
      "ema"    smallest demand EMA(gamma)  -- exponential kernel, time constant ~1/gamma
      "mix"    fast EMA + w_slow * slow EMA -- two-scale kernel bank (the multi-scale prescription)
      "belady" farthest next demand        -- offline-optimal eviction (oracle)
    All non-oracle keys are causal and include the CURRENT token's demand, matching the information
    the shipped min_logit trigger has (it reads the current logits). Ties break on admission age,
    so every arm degenerates to `fifo` when its kernel is uninformative.
    """
    D, R, k, E = d["D"], d["R"], d["k"], d["E"]
    T = len(D)
    adm, _ = admissions(R)
    nxt = None
    if evict == "belady":
        nxt = [None] * T
        seen = {}
        for t in range(T - 1, -1, -1):
            nxt[t] = dict(seen)
            for e in D[t]:
                seen[e] = t

    res = set(R[0])
    age = {e: i for i, e in enumerate(sorted(res))}
    clock = len(res)
    last_dem = {e: 0 for e in res}
    ema = np.zeros(E)
    slow = np.zeros(E)
    score = np.zeros(E)                                    # nomination EMA (selfnom only)
    hist = collections.deque(maxlen=max(W, 1))
    cov, swaps = [], 0
    resid = collections.Counter()

    for t in range(1, T):
        cov.append(len(D[t] & res) / k)                    # pre-swap coverage (entry state)
        # causal state update: the current token's demand is available to the decision
        hist.append(D[t])
        for e in D[t]:
            last_dem[e] = t
        ema *= (1.0 - gamma)
        slow *= (1.0 - gamma_slow)
        for e in D[t]:
            ema[e] += gamma
            slow[e] += gamma_slow
        score *= (1.0 - ema_gamma)
        for e in D[t]:
            score[e] += ema_gamma

        a = adm[t] if protocol == "trace" else _nominate_selfnom(res, D[t], score)
        if a is not None and a not in res:
            swaps += 1
            if evict == "fifo":
                victim = min(res, key=lambda e: age[e])
            elif evict == "lrd":
                victim = min(res, key=lambda e: (last_dem[e], age[e]))
            elif evict == "boxW":
                victim = min(res, key=lambda e: (sum(1 for s in hist if e in s), age[e]))
            elif evict == "ema":
                victim = min(res, key=lambda e: (ema[e], age[e]))
            elif evict == "mix":
                victim = min(res, key=lambda e: (ema[e] + w_slow * slow[e], age[e]))
            elif evict == "belady":
                victim = max(res, key=lambda e: nxt[t].get(e, 10 ** 9))
            else:
                raise ValueError(evict)
            res.discard(victim)
            res.add(a)
            clock += 1
            age[a] = clock
        resid.update(res)

    v = np.array([resid[e] for e in range(E)], float)
    v /= v.sum()
    nz = v[v > 0]
    lo, hi = block_ci(cov)
    return dict(cov=float(np.mean(cov)), lo=lo, hi=hi, swaps=swaps / (T - 1),
                eff=float(np.exp(-(nz * np.log(nz)).sum())), union=int((v > 0).sum()),
                maxres=max(resid.values()) / (T - 1))


def k3_kernel_sweep(data):
    print("\n=== K3  eviction-kernel replay (same demand, same nominations, different kernel) ===")
    arms = ([("lru = FIFO (box on admissions)", "fifo", {}),
             ("LRD (refresh on demand)", "lrd", {})]
            + [(f"box-W demand count, W={W}", "boxW", dict(W=W)) for W in (1, 2, 4, 8, 16, 32, 64)]
            + [(f"EMA demand, gamma={g:g} (tau~{1 / g:.0f} tok)", "ema", dict(gamma=g))
               for g in (0.5, 0.25, 0.125, 0.0625, 0.03125)]
            + [(f"two-scale EMA, fast 0.5 + {w:g}*slow 0.03125", "mix",
                dict(gamma=0.5, gamma_slow=0.03125, w_slow=w)) for w in (0.5, 1.0, 2.0, 4.0)]
            + [("Belady (oracle eviction)", "belady", {})])
    rows = []
    for m, d in data.items():
        R, D, k = d["R"], d["D"], d["k"]
        shipped = [len(D[t] & R[t - 1]) / k for t in range(1, len(R))]
        slo, shi = block_ci(shipped)
        v = np.array([sum(e in R[t] for t in range(1, len(R))) for e in range(d["E"])], float)
        v /= v.sum()
        nz = v[v > 0]
        eff0 = float(np.exp(-(nz * np.log(nz)).sum()))
        print(f"\n  {label(m)}   [logged shipped min_logit: cov={np.mean(shipped) * 100:.1f}% "
              f"[{slo * 100:.1f},{shi * 100:.1f}]  eff={eff0:.1f}]")
        rows.append([label(m), "trace", "min_logit (shipped, logged)", f"{np.mean(shipped):.4f}",
                     f"{slo:.4f}", f"{shi:.4f}", "1.0000", f"{eff0:.2f}", int((v > 0).sum()),
                     f"{v.max():.4f}"])
        for protocol in ("trace", "selfnom"):
            for name, ev, kw in arms:
                if protocol == "selfnom" and ev == "belady":
                    continue                              # oracle only needed once, on the trace
                r = replay(d, ev, protocol=protocol, **kw)
                rows.append([label(m), protocol, name, f"{r['cov']:.4f}", f"{r['lo']:.4f}",
                             f"{r['hi']:.4f}", f"{r['swaps']:.4f}", f"{r['eff']:.2f}",
                             r["union"], f"{r['maxres']:.4f}"])
                tag = "trace " if protocol == "trace" else "selfnom"
                print(f"      [{tag}] {name:38s} cov={r['cov'] * 100:5.1f}% "
                      f"[{r['lo'] * 100:4.1f},{r['hi'] * 100:4.1f}]  swaps={r['swaps']:.3f}  "
                      f"eff={r['eff']:5.1f}  maxres={r['maxres'] * 100:5.1f}%")
    _csv("k3_eviction_kernel_replay.csv",
         ["model", "nomination_protocol", "eviction_kernel", "set_coverage", "ci_lo", "ci_hi",
          "swap_rate", "eff_experts", "union", "max_residency"], rows)
    return rows


# ---------------------------------------------------------------------------- K4
def k4_signal_bandwidth(data):
    """How much of the demand signal survives the cap-1 residency sampler.

    Per expert, correlate the demand indicator d_e(t) with the residency indicator r_e(t) at lag 0,
    and report the autocorrelation time of each. A residency process whose autocorrelation time is
    far shorter than demand's is under-integrating; far longer means it is over-smoothing.
    """
    print("\n=== K4  demand vs residency signal bandwidth ===")
    rows = []
    for m, d in data.items():
        D, R, E, T = d["D"], d["R"], d["E"], len(d["D"])
        dm = np.zeros((T, E))
        rm = np.zeros((T, E))
        for t in range(T):
            for e in D[t]:
                dm[t, e] = 1
            for e in R[t]:
                rm[t, e] = 1

        def acorr_time(X):
            """Sum of positive autocorrelation lags (integrated autocorrelation time), per expert."""
            out = []
            for e in range(E):
                x = X[:, e] - X[:, e].mean()
                if x.std() < 1e-9:
                    continue
                c = np.correlate(x, x, "full")[T - 1:] / (x @ x)
                s = 1.0
                for lag in range(1, min(64, T)):
                    if c[lag] <= 0:
                        break
                    s += 2 * c[lag]
                out.append(s)
            return float(np.mean(out))

        td, tr = acorr_time(dm), acorr_time(rm)
        num = ((dm - dm.mean(0)) * (rm - rm.mean(0))).sum()
        den = np.sqrt(((dm - dm.mean(0)) ** 2).sum() * ((rm - rm.mean(0)) ** 2).sum())
        print(f"  {label(m):34s} tau_demand={td:5.2f} tok  tau_residency={tr:5.2f} tok  "
              f"ratio={tr / td:4.2f}  corr(d,r)={num / den:.3f}")
        rows.append([label(m), f"{td:.4f}", f"{tr:.4f}", f"{tr / td:.4f}", f"{num / den:.4f}"])
    _csv("k4_signal_bandwidth.csv",
         ["model", "tau_demand_tokens", "tau_residency_tokens", "ratio", "corr_demand_residency"],
         rows)
    return rows


# ---------------------------------------------------------------------------- io
def _csv(name, header, rows):
    os.makedirs(ABLATIONS, exist_ok=True)
    with open(os.path.join(ABLATIONS, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", os.path.join(ABLATIONS, name))


def main():
    data = load()
    m0 = next(iter(data.values()))
    print(f"source: {SRC}\n"
          f"slice:  {len(data)} models x {len(m0['D'])} tokens, deepest MoE layer, sequence 0, "
          f"k={m0['k']} of E={m0['E']}")
    k1_effective_kernel(data)
    k2_renewal(data)
    k3_kernel_sweep(data)
    k4_signal_bandwidth(data)
    print("\nkernel_replay complete.")


if __name__ == "__main__":
    main()
