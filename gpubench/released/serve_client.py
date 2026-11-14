#!/usr/bin/env python3
"""Talks to one llama-server (one routing arm per server process) for the released-model runs.

  contexts : build chat contexts from GSM8K test items (earlier questions with their reference solutions
             as finished turns, then one open question), chat template applied by the server with thinking
             off, tokenized by the server. Writes prompts.json: {ctx: {"prompt": [ids], "n": len}}.
  generate : greedy continuation of each prompt under this server's routing; writes, per ctx,
             <out>/ctx<N>_<tag>.ctx (first N tokens of prompt + answer, int32) and .gen (the next
             --n-gen tokens). llama-bench replays them (TEMPORAL_BENCH_CTX / TEMPORAL_BENCH_GEN).
  gsm8k    : the zero-shot CoT prompt of lm-eval's gsm8k_cot_zeroshot as one user turn, thinking off,
             sampled, one request at a time, prompt cache off (every request starts at position 0, so the
             engine's routing state starts fresh). Writes one JSON line per item.

The server must be started with --jinja -np 1 and -ub/-b at least as large as the longest prompt, so a
prompt is never split into micro-batches (a one-token tail would be routed as a decode step).
"""
import argparse, json, os, struct, sys, urllib.request

Q = "Q: {q}\nA: Let's think step by step."


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def template(url, messages):
    return post(url, "/apply-template", {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]


def tokenize(url, text):
    return post(url, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def load_gsm8k(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pydict()
    return [{"question": q, "answer": a} for q, a in zip(t["question"], t["answer"])]


def cmd_contexts(a):
    items = load_gsm8k(a.gsm8k)
    out = {}
    for ctx in a.ctx:
        # finished turns from the front of the test set, the open question from the back
        target_lo, target_hi = ctx - a.slack_hi, ctx - a.slack_lo
        msgs, j, best = [], 0, None
        open_q = {"role": "user", "content": Q.format(q=items[-1 - a.ctx.index(ctx)]["question"])}
        while j < len(items):
            trial = msgs + [{"role": "user", "content": Q.format(q=items[j]["question"])},
                            {"role": "assistant", "content": items[j]["answer"].replace("####", "The answer is")}]
            toks = tokenize(a.url, template(a.url, trial + [open_q]))
            if len(toks) > target_hi:
                break
            msgs, best = trial, toks; j += 1
        assert best is not None and len(best) >= target_lo, f"ctx {ctx}: could not fit turns ({best and len(best)})"
        out[ctx] = {"prompt": best, "n": len(best), "turns": j}
        print(f"ctx {ctx}: prompt {len(best)} tokens, {j} finished turns", flush=True)
    json.dump(out, open(a.out, "w"))


def cmd_generate(a):
    P = json.load(open(a.prompts))
    os.makedirs(a.out, exist_ok=True)
    for ctx, rec in P.items():
        ctx = int(ctx)
        need = ctx + a.n_gen - rec["n"]
        r = post(a.url, "/completion", {"prompt": rec["prompt"], "n_predict": need + 8, "temperature": 0.0,
                                        "cache_prompt": False, "return_tokens": True, "ignore_eos": False})
        gen = r["tokens"]
        seq = rec["prompt"] + gen
        if len(seq) < ctx + a.n_gen:
            sys.exit(f"ctx {ctx}: continuation stopped after {len(gen)} tokens; need {need}")
        for ext, part in (("ctx", seq[:ctx]), ("gen", seq[ctx:ctx + a.n_gen])):
            with open(os.path.join(a.out, f"ctx{ctx}_{a.tag}.{ext}"), "wb") as f:
                f.write(struct.pack(f"<{len(part)}i", *part))
        with open(os.path.join(a.out, f"ctx{ctx}_{a.tag}.txt"), "w") as f:
            f.write(r["content"])
        print(f"ctx {ctx} [{a.tag}]: prompt {rec['n']}, generated {len(gen)}, wrote {ctx} + {a.n_gen}", flush=True)


def cmd_gsm8k(a):
    items = load_gsm8k(a.gsm8k)[a.start:a.start + a.n]
    with open(a.out, "a") as f:
        for i, it in enumerate(items):
            prompt = template(a.url, [{"role": "user", "content": Q.format(q=it["question"])}])
            r = post(a.url, "/completion", {"prompt": prompt, "n_predict": a.max_tokens, "temperature": a.temp,
                                            "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed + i,
                                            "cache_prompt": False})
            row = {"i": a.start + i, "tag": a.tag, "content": r["content"], "tokens_predicted": r["tokens_predicted"],
                   "tokens_evaluated": r["tokens_evaluated"], "stop_type": r.get("stop_type"),
                   "answer": it["answer"].split("####")[-1].strip()}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{a.tag} item {a.start + i}: prompt {r['tokens_evaluated']} gen {r['tokens_predicted']}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contexts"); c.add_argument("--url", required=True); c.add_argument("--gsm8k", required=True)
    c.add_argument("--ctx", type=int, nargs="+", default=[1024, 2048, 4096]); c.add_argument("--out", required=True)
    c.add_argument("--slack-lo", type=int, default=30); c.add_argument("--slack-hi", type=int, default=250)
    g = sub.add_parser("generate"); g.add_argument("--url", required=True); g.add_argument("--prompts", required=True)
    g.add_argument("--tag", required=True); g.add_argument("--out", required=True); g.add_argument("--n-gen", type=int, default=100)
    s = sub.add_parser("gsm8k"); s.add_argument("--url", required=True); s.add_argument("--gsm8k", required=True)
    s.add_argument("--tag", required=True); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=40)
    s.add_argument("--start", type=int, default=0); s.add_argument("--max-tokens", type=int, default=2048)
    s.add_argument("--temp", type=float, default=0.7); s.add_argument("--top-p", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=20); s.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    {"contexts": cmd_contexts, "generate": cmd_generate, "gsm8k": cmd_gsm8k}[a.cmd](a)


if __name__ == "__main__":
    main()
