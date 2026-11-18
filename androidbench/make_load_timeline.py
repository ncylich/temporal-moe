#!/usr/bin/env python3
"""Per-layer load and expert-compute timeline from on-device traces (ledger S1-8).

    python3 make_load_timeline.py --trained results/pixel_skliar/trace/trainedtext_stream_s1.json \
        --skliar results/pixel_skliar/trace/skliar_stream_s1.json --out results/pixel_skliar/load_timeline.html

Traces come from LLAMA_TEMPORAL_TRACE=1 (chrome-trace JSON; lanes: tid 0..3 compute threads, 100..105 fetch
workers, 200 janitor). Same palette as make_timeline.py. For each arm: one decoded token drawn layer by layer
(when each read was queued, started and finished; when thread 0 computed residents, stalled, and computed the
new experts), one layer zoomed to every expert's matrix products on all four threads, and the mean stall per
layer against the number of experts the layer loaded. Tracing adds overhead, so absolute times are a little
above the timed rows.
"""
import argparse, collections, html, json, re, statistics as st


def load(p):
    t = json.load(open(p))
    for e in t:
        m = re.match(r"\w+ L(-?\d+) e(-?\d+)", e["name"])
        e["L"], e["e"], e["end"] = int(m.group(1)), int(m.group(2)), e["ts"] + e["dur"]
    return t


def layers(t):
    """Every decoded layer-token after the first five tokens: dict with its events, times relative to the submit."""
    g = sorted((e for e in t if e["cat"] == "GEMV" and e["tid"] == 0), key=lambda e: e["ts"])
    toks, cur, last = [], [], None
    for e in g:
        if last is not None and e["L"] < last:
            toks.append(cur); cur = []
        cur.append(e); last = e["L"]
    toks.append(cur)
    toks = [k for k in toks if len({e["L"] for e in k}) == 13 and len(k) < 3000][5:]
    by = collections.defaultdict(list)
    for e in t:
        if e["cat"] != "GEMV" or e["tid"] != 0:
            by[e["L"]].append(e)
    for v in by.values():
        v.sort(key=lambda e: e["ts"])
    out = []
    for ti, k in enumerate(toks):
        bl = collections.defaultdict(list)
        for e in k:
            bl[e["L"]].append(e)
        Ls = sorted(bl); prev = None
        for L in Ls:
            g0 = bl[L]; cs, ce = g0[0]["ts"], g0[-1]["end"]
            lo = prev if prev is not None else cs - 2500
            ev = [e for e in by[L] if lo <= e["ts"] <= ce]
            fs = [e for e in ev if e["cat"] == "FETCH"]; qs = [e for e in ev if e["cat"] == "ROUTER"]
            ws = [e for e in ev if e["cat"] == "WAIT"]
            sub = min(e["ts"] for e in qs) if qs else (min(e["ts"] for e in fs) if fs else cs)
            new = {e["e"] for e in fs}
            out.append(dict(tok=ti, L=L, t0=sub, cs=cs, ce=ce, gap=(cs - prev) if prev is not None else None,
                            n=len(new), new=new, fetch=fs, q=qs, wait=ws, gemv0=g0,
                            gemv=[e for e in ev if e["cat"] == "GEMV"], stall=sum(e["dur"] for e in ws),
                            read=(max(e["end"] for e in fs) - sub) if fs else 0.0,
                            comp=sum(e["dur"] for e in g0 if e["e"] not in new)))
            prev = ce
    return out


def summary(ls):
    by = collections.defaultdict(list)
    for r in ls:
        by[min(r["n"], 4)].append(r)
    s = dict(n=len(ls), stall=st.mean(r["stall"] for r in ls), comp=st.mean(r["comp"] for r in ls),
             gap=st.mean(r["gap"] for r in ls if r["gap"] is not None),
             read1=st.mean(r["read"] for r in ls if r["n"] == 1),
             rd=st.mean(e["dur"] for r in ls for e in r["fetch"]),
             qw=st.mean(e["dur"] for r in ls for e in r["q"]),
             classes={k: dict(share=len(v) / len(ls), stall=st.mean(r["stall"] for r in v),
                              read=st.mean(r["read"] for r in v), n=len(v)) for k, v in sorted(by.items())})
    return s


def pick_token(ls, want):
    """A token near the median token time; for Skliar one that has both an empty layer and a burst."""
    by = collections.defaultdict(list)
    for r in ls:
        by[r["tok"]].append(r)
    span = {k: v[-1]["ce"] - v[0]["cs"] for k, v in by.items()}
    med = st.median(span.values())
    ok = [k for k, v in by.items() if want(v)] or list(by)
    return by[min(ok, key=lambda k: abs(span[k] - med))]


def esc(s):
    return html.escape(str(s))


def svg_token(rows, xmax, label):
    """One row per layer, time 0 = the reads are queued. Top half: the six fetch workers. Bottom: thread 0."""
    W, LM, RH = 1040, 64, 38
    H = 30 + RH * len(rows) + 6
    sx = lambda us: LM + (W - LM - 12) * max(-60.0, min(us, xmax)) / xmax
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(label)}">']
    for ms in range(0, int(xmax // 500) + 1):
        x = sx(ms * 500)
        o.append(f'<line x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{H - 6}" class="grid"/>'
                 f'<text x="{x:.1f}" y="14" class="tick" text-anchor="middle">{ms * 0.5:g} ms</text>')
    for i, r in enumerate(rows):
        y = 30 + RH * i
        o.append(f'<text x="{LM - 8}" y="{y + 15}" class="lab" text-anchor="end">L{r["L"]}</text>'
                 f'<text x="{LM - 8}" y="{y + 28}" class="sub" text-anchor="end">{r["n"]} load{"" if r["n"] == 1 else "s"}</text>')
        o.append(f'<rect x="{LM}" y="{y}" width="{W - LM - 12}" height="{RH - 6}" class="rowbg"/>')
        t0 = r["t0"]
        for e in r["q"]:                                   # queue wait, then the read, on the worker's own mini-lane
            ly = y + 1 + 2.5 * (e["tid"] - 100)
            o.append(f'<rect x="{sx(e["ts"] - t0):.1f}" y="{ly:.1f}" width="{max(0.6, sx(e["end"] - t0) - sx(e["ts"] - t0)):.1f}" height="2" class="queue"/>')
        for e in r["fetch"]:
            ly = y + 1 + 2.5 * (e["tid"] - 100)
            o.append(f'<rect x="{sx(e["ts"] - t0):.1f}" y="{ly:.1f}" width="{max(0.8, sx(e["end"] - t0) - sx(e["ts"] - t0)):.1f}" height="2" class="fetch">'
                     f'<title>read for expert {e["e"]}, worker {e["tid"] - 100}: {e["ts"] - t0:.0f} to {e["end"] - t0:.0f} us</title></rect>')
        res = [e for e in r["gemv0"] if e["e"] not in r["new"]]; nw = [e for e in r["gemv0"] if e["e"] in r["new"]]
        cy = y + 18
        if res:
            a, b = res[0]["ts"] - t0, res[-1]["end"] - t0
            o.append(f'<rect x="{sx(a):.1f}" y="{cy}" width="{max(1, sx(b) - sx(a)):.1f}" height="12" class="comp">'
                     f'<title>resident experts: {a:.0f} to {b:.0f} us</title></rect>')
        for e in r["wait"]:
            a, b = e["ts"] - t0, e["end"] - t0
            o.append(f'<rect x="{sx(a):.1f}" y="{cy}" width="{max(0.8, sx(b) - sx(a)):.1f}" height="12" class="stall">'
                     f'<title>stalled on expert {e["e"]}: {a:.0f} to {b:.0f} us</title></rect>')
        for e in nw:
            a, b = e["ts"] - t0, e["end"] - t0
            o.append(f'<rect x="{sx(a):.1f}" y="{cy}" width="{max(1.5, sx(b) - sx(a)):.1f}" height="12" class="newx">'
                     f'<title>new expert {e["e"]}: {a:.0f} to {b:.0f} us</title></rect>')
        if r["stall"] > 0:
            o.append(f'<text x="{min(W - 110, sx(r["ce"] - t0) + 6):.1f}" y="{cy + 10}" class="note">stalled {r["stall"]:.0f} us</text>')
    o.append("</svg>")
    return "".join(o)


def svg_zoom(r, xmax, label):
    """One layer: every matrix product of every expert on the four compute threads, and every read."""
    W, LM, LH = 1040, 112, 15
    lanes = [("thread 0", 0), ("thread 1", 1), ("thread 2", 2), ("thread 3", 3)] + [(f"read worker {i}", 100 + i) for i in range(6)]
    H = 30 + LH * len(lanes) + 14
    sx = lambda us: LM + (W - LM - 12) * max(-xmax * 0.06, min(us, xmax)) / (xmax * 1.0)
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(label)}">']
    step = 100 if xmax <= 1200 else 250
    for k in range(0, int(xmax // step) + 1):
        x = sx(k * step)
        o.append(f'<line x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{H - 10}" class="grid"/>'
                 f'<text x="{x:.1f}" y="14" class="tick" text-anchor="middle">{k * step} us</text>')
    t0 = r["t0"]
    for i, (name, tid) in enumerate(lanes):
        y = 28 + LH * i
        o.append(f'<text x="{LM - 8}" y="{y + 10}" class="lab" text-anchor="end">{name}</text>'
                 f'<rect x="{LM}" y="{y}" width="{W - LM - 12}" height="{LH - 3}" class="rowbg"/>')
        if tid < 100:
            for e in r["gemv"] + (r["gemv0"] if tid == 0 else []):
                if e["tid"] != tid:
                    continue
                a, b = e["ts"] - t0, e["end"] - t0
                cls = "newx" if e["e"] in r["new"] else "comp"
                o.append(f'<rect x="{sx(a):.2f}" y="{y}" width="{max(1.2, sx(b) - sx(a) - 0.6):.2f}" height="{LH - 3}" class="{cls}">'
                         f'<title>expert {e["e"]}, one matrix product: {e["dur"]:.1f} us at {a:.0f} us</title></rect>')
            if tid == 0:
                for e in r["wait"]:
                    a, b = e["ts"] - t0, e["end"] - t0
                    o.append(f'<rect x="{sx(a):.2f}" y="{y}" width="{max(1, sx(b) - sx(a)):.2f}" height="{LH - 3}" class="stall">'
                             f'<title>stalled on expert {e["e"]}: {e["dur"]:.0f} us</title></rect>')
        else:
            for e in r["q"]:
                if e["tid"] == tid:
                    o.append(f'<rect x="{sx(e["ts"] - t0):.2f}" y="{y + 4}" width="{max(0.8, sx(e["end"] - t0) - sx(e["ts"] - t0)):.2f}" height="{LH - 11}" class="queue">'
                             f'<title>queued {e["dur"]:.0f} us before worker {tid - 100} picked it up</title></rect>')
            for e in r["fetch"]:
                if e["tid"] == tid:
                    a, b = e["ts"] - t0, e["end"] - t0
                    o.append(f'<rect x="{sx(a):.2f}" y="{y}" width="{max(1, sx(b) - sx(a)):.2f}" height="{LH - 3}" class="fetch">'
                             f'<title>read for expert {e["e"]}: {a:.0f} to {b:.0f} us ({e["dur"]:.0f} us)</title></rect>')
    o.append("</svg>")
    return "".join(o)


def svg_curve(st_tr, st_sk):
    """Mean stall per layer against the number of experts the layer loaded; bar width = share of layers."""
    W, H, LM, BM = 1040, 250, 64, 44
    ymax = 2000.0
    sy = lambda v: H - BM - (H - BM - 16) * min(v, ymax) / ymax
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="stall per layer against loads in the layer">']
    for k in range(0, 5):
        y = sy(k * 500)
        o.append(f'<line x1="{LM}" y1="{y:.1f}" x2="{W - 12}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{LM - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{k * 0.5:g} ms</text>')
    cw = (W - LM - 12) / 5
    one = st_sk["classes"].get(1, {"stall": 0})["stall"]
    for k in range(5):
        c = st_sk["classes"].get(k)
        x = LM + cw * k
        o.append(f'<text x="{x + cw / 2:.1f}" y="{H - BM + 16}" class="lab" text-anchor="middle">{k}{"+" if k == 4 else ""} load{"" if k == 1 else "s"} in the layer</text>')
        if not c:
            continue
        bw = cw * 0.34
        o.append(f'<rect x="{x + cw / 2 - bw - 3:.1f}" y="{sy(c["stall"]):.1f}" width="{bw:.1f}" height="{max(0.5, sy(0) - sy(c["stall"])):.1f}" class="stall"/>'
                 f'<text x="{x + cw / 2 - bw / 2 - 3:.1f}" y="{sy(c["stall"]) - 5:.1f}" class="val" text-anchor="middle">{c["stall"] / 1000:.2f}</text>')
        if k >= 1:                                         # what k loads would cost if each cost as much as the first
            v = one * k
            o.append(f'<rect x="{x + cw / 2 + 3:.1f}" y="{sy(v):.1f}" width="{bw:.1f}" height="{max(0.5, sy(0) - sy(v)):.1f}" class="ghost"/>'
                     f'<text x="{x + cw / 2 + bw / 2 + 3:.1f}" y="{sy(v) - 5:.1f}" class="sub" text-anchor="middle">{v / 1000:.2f}</text>')
        o.append(f'<text x="{x + cw / 2:.1f}" y="{H - BM + 31}" class="sub" text-anchor="middle">{100 * c["share"]:.0f}% of Skliar\'s layers</text>')
    o.append("</svg>")
    return "".join(o)


CSS = r"""
/* layout: one column, a lede with four figures, then three full-width instrument panels read top to bottom */
:root{--ground:#f2f5f5;--surface:#ffffff;--ink:#0e1719;--muted:#5b6a6f;--line:#dbe3e4;--rule:#c8d4d6;
  --compute:#0f8b80;--fetch:#bd7a12;--stall:#c4402f;--newx:#1b4f9c;--queue:#d9b877;--ghost:#e6c9c4;
  --sans:"Inter",ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;--display:"Space Grotesk",var(--sans);
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0a1113;--surface:#101a1d;--ink:#e6edee;--muted:#8b9ba0;--line:#1d2b2f;--rule:#2a3c41;
  --compute:#2ec5b6;--fetch:#e0a63c;--stall:#f0705f;--newx:#7fb0ff;--queue:#6b5526;--ghost:#4a2b27;color-scheme:dark}}
:root[data-theme="dark"]{--ground:#0a1113;--surface:#101a1d;--ink:#e6edee;--muted:#8b9ba0;--line:#1d2b2f;--rule:#2a3c41;
  --compute:#2ec5b6;--fetch:#e0a63c;--stall:#f0705f;--newx:#7fb0ff;--queue:#6b5526;--ghost:#4a2b27;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;font:14px/1.55 var(--sans);padding-block:28px 64px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:28px}
.eyebrow{font:600 10.5px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--compute)}
h1{font:600 clamp(22px,3vw,29px)/1.18 var(--display);letter-spacing:-.02em;margin:8px 0 0;text-wrap:balance;max-width:30ch}
h2{font:600 17px/1.25 var(--display);margin:0;text-wrap:balance}
p{margin:0;max-width:72ch} .lede{color:var(--muted);margin-top:10px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}
.fig{background:var(--surface);padding:14px 16px;min-width:0}
.fig b{display:block;font:600 24px/1.1 var(--display);font-variant-numeric:tabular-nums}
.fig span{color:var(--muted);font-size:12.5px}
section{display:flex;flex-direction:column;gap:12px}
.panel{background:var(--surface);border:1px solid var(--line);padding:14px 14px 8px}
.panel h3{font:600 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.scroll{overflow-x:auto} svg{display:block;max-width:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--muted)}
.legend i{display:inline-block;width:14px;height:9px;margin-right:6px;vertical-align:baseline}
.k-comp{background:var(--compute)}.k-new{background:var(--newx)}.k-stall{background:var(--stall)}.k-fetch{background:var(--fetch)}.k-queue{background:var(--queue)}.k-ghost{background:var(--ghost)}
.grid{stroke:var(--line);stroke-width:1} .rowbg{fill:var(--ground)}
.tick{fill:var(--muted);font:10.5px var(--mono)} .lab{fill:var(--ink);font:11.5px var(--mono)} .sub{fill:var(--muted);font:10.5px var(--mono)}
.note{fill:var(--stall);font:10.5px var(--mono)} .val{fill:var(--ink);font:600 11.5px var(--mono)}
.comp{fill:var(--compute)}.newx{fill:var(--newx)}.stall{fill:var(--stall)}.fetch{fill:var(--fetch)}.queue{fill:var(--queue)}.ghost{fill:var(--ghost)}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;width:100%;min-width:560px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)} th:first-child,td:first-child{text-align:left}
th{font:600 11px/1.3 var(--mono);color:var(--muted);letter-spacing:.04em}
ol{margin:0;padding-left:20px;max-width:72ch} li{margin:4px 0} .foot{color:var(--muted);font-size:12.5px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True); ap.add_argument("--skliar", required=True)
    ap.add_argument("--trained-off"); ap.add_argument("--skliar-off"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr, sk = layers(load(a.trained)), layers(load(a.skliar))
    s_tr, s_sk = summary(tr), summary(sk)
    tok_tr = pick_token(tr, lambda v: True)
    tok_sk = pick_token(sk, lambda v: any(r["n"] == 0 for r in v) and any(r["n"] >= 3 for r in v))
    xmax = max(r["ce"] - r["t0"] for r in tok_tr + tok_sk) * 1.3
    z_tr = min((r for r in tr if r["n"] == 1), key=lambda r: abs(r["stall"] - s_tr["stall"]))
    sk3 = [r for r in sk if r["n"] == 3]
    z_sk = min(sk3, key=lambda r: abs(r["stall"] - st.mean(x["stall"] for x in sk3)))
    zx = max(z_tr["ce"] - z_tr["t0"], z_sk["ce"] - z_sk["t0"]) * 1.06
    off = ""
    if a.trained_off and a.skliar_off:
        o_tr, o_sk = summary(layers(load(a.trained_off))), summary(layers(load(a.skliar_off)))
        off = (f"<tr><td>Shared expert after the wait (the earlier build)</td><td>{o_tr['stall']:.0f}</td><td>{o_sk['stall']:.0f}</td>"
               f"<td>{13 * (o_sk['stall'] - o_tr['stall']) / 1000:+.2f}</td></tr>")
    c = s_sk["classes"]
    rows = "".join(f"<tr><td>{k}{'+' if k == 4 else ''}</td><td>{100 * v['share']:.1f}%</td><td>{v['read']:.0f}</td><td>{v['stall']:.0f}</td>"
                   f"<td>{v['share'] * v['stall']:.0f}</td></tr>" for k, v in c.items())
    d = 13 * (s_sk["stall"] - s_tr["stall"]) / 1000
    page = f"""<title>Load Overlap Timeline</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">Pixel 10a &middot; 1.26 B model &middot; 18 of 192 experts per layer in RAM</div>
  <h1>Why one load per layer is not much faster than Skliar's bursts</h1>
  <p class="lede">Traced on the phone, one 48-token run per arm, context 1,024, shared expert first. Every layer that loads
  an expert stalls, because a read takes about {s_tr['read1'] / s_tr['comp']:.0f} times longer than the computation meant to hide it.
  Once every load stalls, the stall is roughly proportional to the number of loads, so only the average matters, and the averages are matched.</p>
</header>
<div class="figs">
  <div class="fig"><b>{s_tr['comp'] / 1000:.2f} ms</b><span>thread 0 computing the 17 resident experts of a layer. This is the window a read can hide under.</span></div>
  <div class="fig"><b>{s_tr['read1'] / 1000:.2f} ms</b><span>one expert's read, from queued to last byte (trained model, six reads of about {s_tr['rd']:.0f} us each).</span></div>
  <div class="fig"><b>{s_tr['stall'] / 1000:.2f} ms</b><span>stall per layer, trained model. Times 13 layers: {13 * s_tr['stall'] / 1000:.1f} ms per token.</span></div>
  <div class="fig"><b>{s_sk['stall'] / 1000:.2f} ms</b><span>stall per layer, Skliar, averaged over empty layers and bursts. Times 13: {13 * s_sk['stall'] / 1000:.1f} ms per token.</span></div>
</div>

<section>
  <h2>One token, layer by layer: when each load starts and stops</h2>
  <p>Each row is one layer. Time zero is the moment the layer's reads are queued. The thin bars on top are the six read workers;
  the thick bar below is compute thread 0. Both panels use the same time scale. Hover a bar for its times.</p>
  <div class="legend"><span><i class="k-queue"></i>read waiting for a worker</span><span><i class="k-fetch"></i>read in progress</span>
  <span><i class="k-comp"></i>resident experts computing</span><span><i class="k-stall"></i>thread stalled on a missing expert</span><span><i class="k-new"></i>new expert computing</span></div>
  <div class="panel"><h3>Trained model: exactly one load in every layer</h3><div class="scroll">{svg_token(tok_tr, xmax, "trained model, one token")}</div></div>
  <div class="panel"><h3>Skliar on the plain model: {sum(r['n'] == 0 for r in tok_sk)} layers load nothing, the largest burst is {max(r['n'] for r in tok_sk)}</h3><div class="scroll">{svg_token(tok_sk, xmax, "Skliar, one token")}</div></div>
</section>

<section>
  <h2>One layer, every expert computation</h2>
  <p>Each small block is one matrix product for one expert (three per expert: up, gate, down) on one of the four compute threads.
  The 17 residents finish in about {s_tr['comp']:.0f} us. The reads below them are still running, so the threads stop until the last byte lands.</p>
  <div class="panel"><h3>Trained model, layer {z_tr['L']}: one load, stalled {z_tr['stall']:.0f} us</h3><div class="scroll">{svg_zoom(z_tr, zx, "trained model, one layer")}</div></div>
  <div class="panel"><h3>Skliar, layer {z_sk['L']}: three loads, stalled {z_sk['stall']:.0f} us</h3><div class="scroll">{svg_zoom(z_sk, zx, "Skliar, one layer with three loads")}</div></div>
</section>

<section>
  <h2>The stall is roughly proportional to the loads, so only the average matters</h2>
  <p>Mean stall per layer in Skliar's run, by how many experts the layer loaded. The pale bar is what that many loads would cost
  if each cost as much as the first. Measured stalls track that line up to three loads and sit a little below it. A cost that is
  proportional to loads gives the same total for one load everywhere as for bursts and empty layers with the same average.</p>
  <div class="legend"><span><i class="k-stall"></i>measured stall, ms</span><span><i class="k-ghost"></i>if every load cost as much as the first</span></div>
  <div class="panel"><div class="scroll">{svg_curve(s_tr, s_sk)}</div></div>
  <div class="panel"><div class="scroll"><table>
  <tr><th>Loads in the layer (Skliar)</th><th>Share of layers</th><th>Queued to last byte, us</th><th>Stall, us</th><th>Contribution to the mean stall, us</th></tr>
  {rows}
  <tr><td><b>Skliar, all layers</b></td><td>100%</td><td></td><td></td><td><b>{s_sk['stall']:.0f}</b></td></tr>
  <tr><td><b>Trained, all layers (one load each)</b></td><td>100%</td><td>{s_tr['read1']:.0f}</td><td>{s_tr['stall']:.0f}</td><td><b>{s_tr['stall']:.0f}</b></td></tr>
  </table></div></div>
  <div class="panel"><div class="scroll"><table>
  <tr><th>Stall per layer, us</th><th>Trained</th><th>Skliar</th><th>Skliar minus trained, ms per token (13 layers)</th></tr>
  <tr><td>Shared expert first (this page)</td><td>{s_tr['stall']:.0f}</td><td>{s_sk['stall']:.0f}</td><td>{d:+.2f}</td></tr>{off}
  </table></div></div>
</section>

<section>
  <h2>Reading it from first principles</h2>
  <ol>
    <li>A hard cap of one load per layer pays off only if one load fits under the layer's computation and two do not. Here one does not fit:
    {s_tr['comp'] / 1000:.2f} ms of resident compute against {s_tr['read1'] / 1000:.2f} ms for the read.</li>
    <li>The read is slow because of latency, not volume. Each of the six reads takes about {s_tr['rd']:.0f} us whatever its size,
    and waits about {s_tr['qw']:.0f} us for a worker first.</li>
    <li>The other {s_tr['gap'] / 1000:.1f} ms of each layer (attention, norms, the shared expert) cannot hide the read: the router needs the
    attention output before it knows which expert to load. Moving the shared expert first bought about 0.08 ms of the window.</li>
    <li>So both methods stall on every layer that loads, and the stall scales with the loads: two experts cost {c.get(2, c[1])['stall'] / c[1]['stall']:.1f} times one,
    three cost {c.get(3, c[1])['stall'] / c[1]['stall']:.1f} times. Skliar's bursts are paid back by the {100 * c.get(0, {'share': 0})['share']:.0f}% of its layers that load nothing.
    Net difference: {d:+.2f} ms per token out of about 30.</li>
    <li>What is left of the difference is not the bursts. A single load stalls Skliar's run for {c[1]['stall']:.0f} us against {s_tr['stall']:.0f} us for the trained
    model, because its reads take longer ({s_sk['rd']:.0f} against {s_tr['rd']:.0f} us each). The two runs use different model files; the cause was not looked into.</li>
    <li>The gap would open where a layer computes for longer than one read takes: a larger model, or a faster read path.</li>
  </ol>
  <p class="foot">Source: LLAMA_TEMPORAL_TRACE on the timing binary (engine tree cacc843b), text 1, 48 decoded tokens per arm, the first five
  dropped; {s_tr['n']} layer-tokens per arm. Skliar at lambda 0.7310. One run per arm, with tracing on, made without the cool gate:
  treat the per-layer times as structure, not as benchmark rows. Thread 0's stall is the time it waited; the other three threads wait alongside it.</p>
</section>
</div>
"""
    open(a.out, "w").write(page)
    print(json.dumps(dict(trained=dict(stall=s_tr["stall"], comp=s_tr["comp"], read1=s_tr["read1"], gap=s_tr["gap"], rd=s_tr["rd"], qw=s_tr["qw"]),
                          skliar=dict(stall=s_sk["stall"], comp=s_sk["comp"], read1=s_sk["read1"], rd=s_sk["rd"],
                                      classes={k: {x: round(y, 3) for x, y in v.items()} for k, v in c.items()}),
                          delta_ms_per_token=d), indent=1))


if __name__ == "__main__":
    main()
