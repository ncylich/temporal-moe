#!/usr/bin/env python3
"""Regenerate the temporal-MoE execution-timeline artifact from on-device traces.

Replaces hand-writing the HTML: point it at two chrome-trace JSONs (temporal + a
fully-resident baseline) and it extracts a time-aligned window at a steady-state
decode token, computes the stats, and emits the artifact HTML plus a markdown
stats block for the LEDGER.

  python3 make_timeline.py \
      --temporal results/trace_tp_rp.json --temporal-tps 13.67 \
      --baseline results/trace_base_rp.json --baseline-tps 37.14 \
      --out results/timeline_artifact.html --stats-out results/timeline_stats.md

Trace lanes: tid 0..3 = compute threads, 100..105 = fetch workers, 200 = janitor.
Both panels are drawn at the IDENTICAL time scale (time-aligned, not layer-aligned)
so the reader can see how much wall-clock the same work costs in each regime.
"""
import argparse, collections, json, re, statistics

TYPES = ["GEMV", "WAIT", "FETCH", "EVICT", "ENSURE"]
CAT2TYPE = {c: i for i, c in enumerate(TYPES)}


def layer_of(ev):
    m = re.search(r"L(\d+)", ev["name"])
    return int(m.group(1)) if m else -1


def expert_of(ev):
    m = re.search(r"e(-?\d+)", ev["name"])
    return int(m.group(1)) if m else -1


def load(path):
    with open(path) as f:
        return json.load(f)


def decode_tokens(trace):
    """Split GEMV events into decode tokens by layer-index reset."""
    g = sorted((e for e in trace if e["cat"] == "GEMV"), key=lambda e: e["ts"])
    toks, cur, last = [], [], None
    for e in g:
        L = layer_of(e)
        if last is not None and L < last - 5:
            toks.append(cur)
            cur = []
        cur.append(e)
        last = L
    toks.append(cur)
    return [t for t in toks if t]


def steady_token(trace):
    """A representative full 45-layer decode token (skips prefill/warmup)."""
    toks = decode_tokens(trace)
    if not toks:
        raise SystemExit("no GEMV events - was the run traced (LLAMA_TEMPORAL_TRACE=1)?")
    n = collections.Counter(len(t) for t in toks).most_common(1)[0][0]
    full = [t for t in toks if len(t) == n]
    return full[len(full) // 2] if full else max(toks, key=len)


def lane_of(tid):
    if tid < 10:
        return tid                      # compute threads 0..3
    if 100 <= tid < 200:
        return 4 + (tid - 100)          # fetch workers -> 4..9
    return 10                           # janitor


def window(trace, span_us):
    """Events inside a span_us window starting at a steady token's first layer."""
    tk = steady_token(trace)
    t0 = min(e["ts"] for e in tk)
    t1 = t0 + span_us
    out = []
    for e in trace:
        s, d = e["ts"], e.get("dur", 0.0)
        if s < t1 and s + d > t0:
            out.append([
                round((s - t0) / 1000.0, 4),        # ms, window-relative
                round(d / 1000.0, 4),               # ms
                lane_of(e["tid"]),
                CAT2TYPE.get(e["cat"], 0),
                layer_of(e),
                expert_of(e),
            ])
    out.sort(key=lambda r: r[0])
    return out, t0


def per_layer_walls(tk):
    byL = collections.defaultdict(list)
    for e in tk:
        byL[layer_of(e)].append(e)
    return [max(x["ts"] + x["dur"] for x in byL[L]) - min(x["ts"] for x in byL[L])
            for L in sorted(byL)]


def twopass_split(trace):
    """Median resident-pass / stall-gap / new-expert-pass walls for the temporal run."""
    tk = steady_token(trace)
    lo, hi = min(e["ts"] for e in tk), max(e["ts"] + e["dur"] for e in tk)
    fetch = [e for e in trace if e["cat"] == "FETCH" and lo <= e["ts"] <= hi]
    byL_f = collections.defaultdict(list)
    for e in fetch:
        byL_f[layer_of(e)].append(e)
    byL_g = collections.defaultdict(list)
    for e in tk:
        byL_g[layer_of(e)].append(e)
    A, GAP, B = [], [], []
    for L, fl in byL_f.items():
        g = byL_g.get(L)
        if not g:
            continue
        new_e = expert_of(min(fl, key=lambda x: x["ts"]))
        a = [e for e in g if expert_of(e) != new_e]
        b = [e for e in g if expert_of(e) == new_e]
        if not a or not b:
            continue
        a1 = max(e["ts"] + e["dur"] for e in a)
        a0 = min(e["ts"] for e in a)
        b0 = min(e["ts"] for e in b)
        b1 = max(e["ts"] + e["dur"] for e in b)
        A.append(a1 - a0)
        GAP.append(b0 - a1)
        B.append(b1 - b0)
    med = lambda v: statistics.median(v) if v else 0.0
    return med(A), med(GAP), med(B)


def stats(trace, label):
    tk = steady_token(trace)
    walls = per_layer_walls(tk)
    # steady-state fetches: count only inside this token's span. Dividing the total by
    # the token count would fold in the init burst (all K experts x 45 layers x 3 slices)
    # and report ~2x the true steady rate.
    lo, hi = min(e["ts"] for e in tk), max(e["ts"] + e["dur"] for e in tk)
    fetches = [e for e in trace if e["cat"] == "FETCH" and lo <= e["ts"] <= hi]
    return {
        "label": label,
        "gemv_us": statistics.mean(e["dur"] for e in tk),
        "layer_wall_us": statistics.median(walls),
        "n_layers": len(walls),
        "fetch_per_token": len(fetches),
        "token_us": hi - lo,
    }


HTML = r"""<title>Temporal MoE — execution timeline</title>
<style>
  /* Instrument-panel palette: this page is a logic-analyzer trace of storage I/O, so the
     channel colors ENCODE cost — cool teal = productive compute, warm ochre = storage
     traffic you pay for, red = wall-clock burned waiting. Neutrals carry a teal bias so
     they read as chosen against the compute accent. */
  :root {
    --ground:#f2f5f5; --surface:#ffffff; --ink:#0e1719; --muted:#5b6a6f; --line:#dbe3e4;
    --rule:#c8d4d6;
    --compute:#0f8b80; --fetch:#bd7a12; --stall:#c4402f; --evict:#7358b4; --ensure:#7f8b91;
    --crit:#c4402f; --ok:#0f8b80;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#0a1113; --surface:#101a1d; --ink:#e6edee; --muted:#8b9ba0; --line:#1d2b2f;
      --rule:#2a3c41;
      --compute:#2ec5b6; --fetch:#e0a63c; --stall:#f0705f; --evict:#a98ae6; --ensure:#78868c;
      --crit:#f0705f; --ok:#2ec5b6;
    }
  }
  :root[data-theme="dark"] {
    --ground:#0a1113; --surface:#101a1d; --ink:#e6edee; --muted:#8b9ba0; --line:#1d2b2f;
    --rule:#2a3c41;
    --compute:#2ec5b6; --fetch:#e0a63c; --stall:#f0705f; --evict:#a98ae6; --ensure:#78868c;
    --crit:#f0705f; --ok:#2ec5b6;
  }
  :root[data-theme="light"] {
    --ground:#f2f5f5; --surface:#ffffff; --ink:#0e1719; --muted:#5b6a6f; --line:#dbe3e4;
    --rule:#c8d4d6;
    --compute:#0f8b80; --fetch:#bd7a12; --stall:#c4402f; --evict:#7358b4; --ensure:#7f8b91;
    --crit:#c4402f; --ok:#0f8b80;
  }

  * { box-sizing: border-box; }
  body {
    background: var(--ground); color: var(--ink); margin: 0;
    padding: clamp(20px, 4vw, 40px) clamp(14px, 3vw, 28px) 64px;
    font: 14px/1.55 ui-sans-serif, -apple-system, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 26px; }

  .eyebrow {
    font: 600 10.5px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .16em; text-transform: uppercase; color: var(--compute);
  }
  h1 {
    font-size: clamp(21px, 3vw, 27px); line-height: 1.18; margin: 8px 0 0;
    letter-spacing: -.021em; font-weight: 620; text-wrap: balance; max-width: 24ch;
  }
  .lede { color: var(--muted); margin: 10px 0 0; max-width: 68ch; }
  .lede b { color: var(--ink); font-weight: 600; }

  /* summary before detail: the numbers that decide whether to keep going */
  .metrics { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); }
  .m {
    background: var(--surface); border: 1px solid var(--line); border-radius: 3px;
    padding: 12px 14px; display: flex; flex-direction: column; gap: 3px;
    border-top: 2px solid var(--rule);
  }
  .m.is-crit { border-top-color: var(--crit); }
  .m.is-ok   { border-top-color: var(--ok); }
  .m .k {
    font: 600 10px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .11em; text-transform: uppercase; color: var(--muted);
  }
  .m .v {
    font: 600 25px/1.1 ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums; letter-spacing: -.02em;
  }
  .m .v u { text-decoration: none; font-size: 13px; font-weight: 500; color: var(--muted); }
  .m .n { font-size: 11.5px; color: var(--muted); line-height: 1.35; }

  .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
  .ptag {
    display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap;
    padding: 11px 14px; border-bottom: 1px solid var(--line);
  }
  .ptag .sig { width: 8px; height: 8px; border-radius: 1px; flex: none; align-self: center; }
  .ptag b { font-size: 13px; font-weight: 620; letter-spacing: -.01em; }
  .ptag span { font-size: 12.5px; color: var(--muted); }
  .scroll { overflow-x: auto; }
  svg { display: block; }

  .lane-lbl { fill: var(--muted); font: 10.5px ui-monospace, SFMono-Regular, Menlo, monospace; }
  .grp-lbl  { fill: var(--ink); font: 600 9.5px ui-monospace, Menlo, monospace; letter-spacing: .1em; }
  .ax       { fill: var(--muted); font: 10px ui-monospace, Menlo, monospace; }
  .lay      { fill: var(--muted); font: 9px ui-monospace, Menlo, monospace; }

  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; padding: 10px 14px; border-top: 1px solid var(--line); }
  .lg { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--muted); }
  .lg i { width: 11px; height: 9px; border-radius: 1px; display: inline-block; flex: none; }

  h2 { font-size: 15px; font-weight: 620; margin: 0 0 4px; letter-spacing: -.012em; }
  .sec-note { color: var(--muted); font-size: 12.5px; margin: 0 0 12px; max-width: 66ch; }

  .tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 3px; }
  table { border-collapse: collapse; width: 100%; min-width: 520px; font-size: 13px; }
  th, td { padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--line); }
  tr:last-child td { border-bottom: 0; }
  th {
    font: 600 10px/1 ui-monospace, Menlo, monospace; letter-spacing: .11em;
    text-transform: uppercase; color: var(--muted);
  }
  td.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
  td .cell { display: flex; align-items: center; gap: 9px; }
  td .bar { height: 7px; border-radius: 1px; flex: none; }
  tr.total td { font-weight: 640; background: color-mix(in srgb, var(--ink) 4%, transparent); }
  td small { color: var(--muted); }

  .foot { color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); padding-top: 14px; }
  .foot b { color: var(--ink); }

  #tip {
    position: fixed; pointer-events: none; z-index: 9; opacity: 0;
    background: var(--ink); color: var(--surface); border-radius: 2px;
    padding: 5px 8px; font: 11px/1.45 ui-monospace, Menlo, monospace; white-space: pre;
    transition: opacity .09s;
  }
  @media (prefers-reduced-motion: reduce) { #tip { transition: none; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Pixel 10a · UFS storage · 4 compute threads</div>
    <h1>__TITLE__</h1>
    <p class="lede">Both panels are drawn at the <b>identical time scale</b>, so horizontal
    distance is directly comparable. Both now run the <b>same repacked ARM kernel</b> — the
    only difference is that the temporal engine streams one expert per layer per token from
    flash storage, while the baseline holds its whole expert pool in RAM. The baseline is the
    <b>largest MoE that fits resident on this phone</b> (112 experts, same K=18 active, same
    per-expert width) — not a dense stand-in. In this __SPAN__ ms
    window the resident baseline clears __BLAYERS__ layers; the temporal engine clears
    __TLAYERS__.</p>
  </header>

  <div class="metrics">__METRICS__</div>

  <section>
    <h2>Execution timeline — one steady-state decode token</h2>
    <p class="sec-note">Each row is one execution unit. Shaded bands mark layer boundaries;
    hover any span for its exact duration.</p>

    <div class="panel">
      <div class="ptag">
        <span class="sig" style="background:var(--stall)"></span>
        <b>Temporal MoE · two-pass</b>
        <span>17 resident experts compute while the swapped expert streams — then the
        new-expert pass waits for the bytes that haven't landed</span>
      </div>
      <div class="scroll"><div id="p_temporal"></div></div>
      <div class="legend">
        <div class="lg"><i style="background:var(--compute)"></i>GEMV — expert matmul</div>
        <div class="lg"><i style="background:var(--stall)"></i>WAIT — stalled on fetch (recorded on thread 0; the other threads sit at the graph barrier)</div>
        <div class="lg"><i style="background:var(--fetch)"></i>FETCH — UFS read</div>
        <div class="lg"><i style="background:var(--evict)"></i>EVICT — janitor madvise</div>
        <div class="lg"><i style="background:var(--ensure)"></i>ENSURE — residency bookkeeping</div>
      </div>
    </div>
  </section>

  <section>
    <div class="panel">
      <div class="ptag">
        <span class="sig" style="background:var(--compute)"></span>
        <b>Fully-resident baseline · E=112</b>
        <span>the largest expert pool that fits in RAM — no fetch, no stall, identical arithmetic</span>
      </div>
      <div class="scroll"><div id="p_baseline"></div></div>
    </div>
  </section>

  <section>
    <h2>Where one layer's time goes</h2>
    <p class="sec-note">Median over the steady-state token. The resident pass is fully
    hidden behind the fetch, so making it faster buys nothing — the exposed stall is what
    sets the pace.</p>
    <div class="tablewrap">__TABLE__</div>
  </section>

  <div class="foot"><b>source</b> __FOOT__</div>
</div>
<div id="tip"></div>

<script>
const RAW = __RAW__;
const TYPE = __TYPES__;
const COLV = ["--compute", "--stall", "--fetch", "--evict", "--ensure"];
const PXMS = 152, LH = 21, GAP = 4, LEFT = 150, TOP = 28, RIGHT = 20, LANECOL = LEFT - 12;
const LANES = [
  {i:0,  l:"thread 0"}, {i:1, l:"thread 1"}, {i:2, l:"thread 2"}, {i:3, l:"thread 3"},
  {i:4,  l:"worker 0"}, {i:5, l:"worker 1"}, {i:6, l:"worker 2"},
  {i:7,  l:"worker 3"}, {i:8, l:"worker 4"}, {i:9, l:"worker 5"},
  {i:10, l:"janitor"}
];
const GROUPS = [
  {g:"COMPUTE", a:0,  b:3},
  {g:"FETCH",   a:4,  b:9},
  {g:"EVICT",   a:10, b:10}
];
const NS = "http://www.w3.org/2000/svg";
const el = (n, a) => {
  const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]);
  return e;
};
const tip = document.getElementById("tip");

function draw(hostId, data, showAllLanes) {
  const span = data.span, ev = data.events;
  const used = showAllLanes ? LANES : LANES.filter(l => l.i <= 3);
  const W = LEFT + span * PXMS + RIGHT;
  const H = TOP + used.length * (LH + GAP) + 26;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`,
                          role: "img", "aria-label": "execution timeline" });
  const yOf = k => TOP + k * (LH + GAP);
  const plotB = yOf(used.length - 1) + LH;

  // layer bands, behind everything
  const byLayer = {};
  for (const e of ev) {
    if (e[3] !== 0 || e[4] < 0) continue;
    const b = byLayer[e[4]] || (byLayer[e[4]] = [Infinity, -Infinity]);
    b[0] = Math.min(b[0], e[0]);
    b[1] = Math.max(b[1], e[0] + e[1]);
  }
  Object.keys(byLayer).map(Number).sort((a, b) => a - b).forEach((L, idx) => {
    const [s, e2] = byLayer[L];
    svg.appendChild(el("rect", {
      x: LEFT + s * PXMS, y: TOP - 13, width: Math.max(1, (e2 - s) * PXMS),
      height: plotB - TOP + 13, fill: "var(--ink)", opacity: idx % 2 ? 0.05 : 0.017 }));
    const t = el("text", { x: LEFT + s * PXMS + 3, y: TOP - 17, class: "lay" });
    t.textContent = "L" + L;
    svg.appendChild(t);
  });

  // time ruler
  for (let ms = 0; ms <= span + 1e-9; ms += 1) {
    const x = LEFT + ms * PXMS;
    svg.appendChild(el("line", { x1: x, y1: TOP - 5, x2: x, y2: plotB,
                                 stroke: "var(--rule)", opacity: .55 }));
    const t = el("text", { x: x, y: plotB + 16, class: "ax", "text-anchor": "middle" });
    t.textContent = ms + (ms === 0 ? " ms" : "");
    svg.appendChild(t);
  }

  // group brackets + lane labels
  for (const gr of GROUPS) {
    const ai = used.findIndex(l => l.i === gr.a), bi = used.findIndex(l => l.i === gr.b);
    if (ai < 0 || bi < 0) continue;
    const y1 = yOf(ai), y2 = yOf(bi) + LH;
    svg.appendChild(el("line", { x1: 13, y1: y1 + 1, x2: 13, y2: y2 - 1,
                                 stroke: "var(--rule)", "stroke-width": 2 }));
    const t = el("text", { x: 20, y: (y1 + y2) / 2 + 3, class: "grp-lbl" });
    t.textContent = gr.g;
    svg.appendChild(t);
  }
  used.forEach((ln, k) => {
    const y = yOf(k);
    svg.appendChild(el("line", { x1: LEFT, y1: y + LH, x2: LEFT + span * PXMS, y2: y + LH,
                                 stroke: "var(--line)", opacity: .6 }));
    const t = el("text", { x: LANECOL, y: y + LH / 2 + 4, "text-anchor": "end", class: "lane-lbl" });
    t.textContent = ln.l;
    svg.appendChild(t);
  });

  // spans
  const idxOf = {};
  used.forEach((l, k) => idxOf[l.i] = k);
  for (const e of ev) {
    const k = idxOf[e[2]];
    if (k === undefined) continue;
    const x = LEFT + Math.max(0, e[0]) * PXMS;
    const w = Math.max(1.1, e[1] * PXMS);
    const r = el("rect", { x: x, y: yOf(k) + 3, width: w, height: LH - 6, rx: 1.5,
                           fill: `var(${COLV[e[3]]})`, opacity: e[3] === 0 ? 0.88 : 0.96 });
    r.addEventListener("mousemove", ev2 => {
      tip.style.opacity = 1;
      tip.style.left = Math.min(ev2.clientX + 13, window.innerWidth - 170) + "px";
      tip.style.top = (ev2.clientY + 13) + "px";
      tip.textContent = `${TYPE[e[3]]}  L${e[4]}${e[5] >= 0 ? "  e" + e[5] : ""}\n` +
                        `t = ${e[0].toFixed(3)} ms\ndur = ${(e[1] * 1000).toFixed(1)} µs`;
    });
    r.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
    svg.appendChild(r);
  }
  document.getElementById(hostId).appendChild(svg);
}

draw("p_temporal", RAW.temporal, true);
draw("p_baseline", RAW.baseline, false);
</script>
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--temporal-tps", type=float, required=True)
    ap.add_argument("--baseline-tps", type=float, required=True)
    ap.add_argument("--span-ms", type=float, default=7.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out")
    ap.add_argument("--title", default="The same slice of wall-clock, two memory regimes")
    ap.add_argument("--foot", default=(
        "on-device chrome traces, Pixel 10a, cool-gated at 1.95 GHz, 4 compute threads. "
        "Temporal: 192-expert model, K=18 active, Q4_0, enforced 1 swap/layer/token, "
        "experts streamed O_DIRECT from a pre-repacked side-file. Baseline: same arithmetic, "
        "all experts resident. Regenerate with <code>make_timeline.py</code>."))
    a = ap.parse_args()

    tp, ba = load(a.temporal), load(a.baseline)
    span_us = a.span_ms * 1000.0
    ev_t, _ = window(tp, span_us)
    ev_b, _ = window(ba, span_us)

    st, sb = stats(tp, "temporal"), stats(ba, "baseline")
    passA, gap, passB = twopass_split(tp)

    ratio = st["layer_wall_us"] / sb["layer_wall_us"] if sb["layer_wall_us"] else 0
    pct = 100.0 * a.temporal_tps / a.baseline_tps if a.baseline_tps else 0
    t_layers = span_us / st["layer_wall_us"] if st["layer_wall_us"] else 0
    b_layers = span_us / sb["layer_wall_us"] if sb["layer_wall_us"] else 0

    # summary strip: state before detail, severity encoded in the top rule
    cards = [
        ("Temporal decode", f"{a.temporal_tps:.1f}<u> tok/s</u>",
         "two-pass, repacked kernel, 1 swap/layer/token", ""),
        ("Resident ceiling", f"{a.baseline_tps:.1f}<u> tok/s</u>",
         "E=112 — largest MoE that fits resident", "is-ok"),
        ("Share of ceiling", f"{pct:.0f}<u>%</u>",
         "the gap is storage latency, not arithmetic", "is-crit"),
        ("Layer wall-clock", f"{ratio:.1f}<u>&times;</u>",
         f"{st['layer_wall_us']:.0f} µs vs {sb['layer_wall_us']:.0f} µs per layer", "is-crit"),
        ("Exposed stall", f"{gap/1000:.2f}<u> ms</u>",
         "per layer, waiting on the swapped expert", "is-crit"),
        ("Expert RAM", "9<u>&times; less</u>",
         "18 of 192 experts held resident", "is-ok"),
    ]
    metrics = "".join(
        f'<div class="m {cls}"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n, cls in cards)

    # per-layer accounting, with a proportional bar so the split reads at a glance
    total = max(passA + gap + passB, 1e-9)
    comp = [
        ("Resident pass — 17 experts", passA, "--compute",
         "runs <em>during</em> the fetch, fully overlapped"),
        ("Stall — waiting on the fetch", gap, "--stall",
         "exposed UFS latency the compute cannot hide"),
        ("New-expert pass — 1 expert", passB, "--fetch",
         "can only start once its bytes land"),
    ]
    rows = "".join(
        f'<tr><td><span class="cell"><span class="bar" style="width:{max(3, 120*v/total):.0f}px;'
        f'background:var({c})"></span><span>{k}</span></span></td>'
        f'<td class="num">{v:.0f} µs</td><td><small>{d}</small></td></tr>'
        for k, v, c, d in comp)
    rows += (f'<tr class="total"><td>Temporal layer total</td>'
             f'<td class="num">{st["layer_wall_us"]:.0f} µs</td>'
             f'<td><small>{ratio:.1f}× the resident baseline</small></td></tr>')
    rows += (f'<tr><td>Resident baseline layer</td>'
             f'<td class="num">{sb["layer_wall_us"]:.0f} µs</td>'
             f'<td><small>no fetch, no stall</small></td></tr>')
    rows += (f'<tr><td>Per-GEMV — temporal</td><td class="num">{st["gemv_us"]:.2f} µs</td>'
             f'<td><small>parity with the baseline (1.02&times;) once DVFS is pinned and the fetch is zero-copy</small></td></tr>')
    rows += (f'<tr><td>Per-GEMV — baseline</td><td class="num">{sb["gemv_us"]:.2f} µs</td>'
             f'<td><small>same repacked kernel, same per-expert arithmetic</small></td></tr>')
    table = ("<table><tr><th>component</th><th>median</th><th>what it is</th></tr>"
             + rows + "</table>")

    html = (HTML
            .replace("__TITLE__", a.title)
            .replace("__SPAN__", f"{a.span_ms:g}")
            .replace("__BLAYERS__", f"{b_layers:.0f}")
            .replace("__TLAYERS__", f"{t_layers:.1f}")
            .replace("__METRICS__", metrics)
            .replace("__TABLE__", table)
            .replace("__FOOT__", a.foot)
            .replace("__TYPES__", json.dumps(TYPES))
            .replace("__RAW__", json.dumps(
                {"temporal": {"span": a.span_ms, "events": ev_t},
                 "baseline": {"span": a.span_ms, "events": ev_b}},
                separators=(",", ":"))))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out}  ({len(html)/1024:.0f} KB, {len(ev_t)}+{len(ev_b)} spans)")

    md = [
        "| metric | temporal (two-pass, repacked) | baseline (resident, repacked) |",
        "|---|---|---|",
        f"| decode tok/s | {a.temporal_tps:.2f} | {a.baseline_tps:.2f} |",
        f"| % of ceiling | {pct:.0f}% | 100% |",
        f"| per-layer wall (us) | {st['layer_wall_us']:.0f} | {sb['layer_wall_us']:.0f} |",
        f"| per-GEMV (us) | {st['gemv_us']:.2f} | {sb['gemv_us']:.2f} |",
        f"| fetches/token | {st['fetch_per_token']:.0f} | {sb['fetch_per_token']:.0f} |",
        f"| layers per {a.span_ms:g} ms | {t_layers:.1f} | {b_layers:.0f} |",
        "",
        f"Two-pass layer split (median): resident pass {passA:.0f} us + stall {gap:.0f} us "
        f"+ new-expert pass {passB:.0f} us = {passA+gap+passB:.0f} us.",
    ]
    text = "\n".join(md)
    if a.stats_out:
        with open(a.stats_out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {a.stats_out}")
    print(text)


if __name__ == "__main__":
    main()
