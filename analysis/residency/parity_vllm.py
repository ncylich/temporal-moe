#!/usr/bin/env python3
"""End-to-end parity and throughput of the serving-side residency stack.

Greedy generation on N GSM8K questions through the patched vLLM engine, free arm and one
constrained arm, dumping generated token ids, wall time, and measured swap traffic. Run
it once per configuration (the walker and eager/graph mode are chosen by env at import:
TEMPORAL_WALKER, TEMPORAL_EAGER) and diff the dumps:

    TEMPORAL_WALKER=slots TEMPORAL_EAGER=1 parity_vllm.py --path M --out old.json
    parity_vllm.py --path M --out new.json          # fast walker + CUDA graphs
    parity_vllm.py --compare old.json new.json

Greedy + the same kernels => the constrained arm must be token-identical between the two
walkers (the masks are bit-exact by test_residency_kernels.py); CUDA-graph replay of the
same ops is expected identical too, and this is where that expectation is checked.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path")
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--think", choices=("on", "off"), default=None)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--sleep-mode", action="store_true", help="enable_sleep_mode + one sleep/wake cycle before generating (what the in-process sampler does)")
    ap.add_argument("--warm", type=int, default=4, help="prompts in the warm-up call (0 = none)")
    ap.add_argument("--adapter", default=None, help="adapter .pt to apply to the engine (raw base + adapter, same class as the online sampler)")
    ap.add_argument("--no-hooks", action="store_true", help="plain vLLM: do not install the residency hooks (free arm only)")
    A = ap.parse_args()
    if A.compare:
        a, b = (json.load(open(p)) for p in A.compare)
        for arm in a["gens"]:
            same = sum(x == y for x, y in zip(a["gens"][arm], b["gens"][arm]))
            div = [next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), "len") for x, y in zip(a["gens"][arm], b["gens"][arm]) if x != y]
            if div:
                print(f"{arm:<5} first differing token per non-identical row: {div}")
            print(f"{arm:<5} identical {same}/{len(a['gens'][arm])}  "
                  f"tok/s {a['tps'][arm]:.0f} -> {b['tps'][arm]:.0f} ({b['tps'][arm]/a['tps'][arm]:.2f}x)  "
                  f"swaps/token {a['swaps'][arm]:.4f} -> {b['swaps'][arm]:.4f}")
        return
    import vllm_glue
    if not A.no_hooks:
        vllm_glue.install()
        import vllm_residency  # noqa: F401
    from decode_state import DEC
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from temporal import temporal_router as TR
    qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][: A.n]
    msgs = [[{"role": "user", "content": q}] for q in qs]
    llm = LLM(model=A.path, **vllm_glue.llm_kwargs(), gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_model_len, enable_sleep_mode=A.sleep_mode)
    if A.sleep_mode:
        llm.sleep(level=1); llm.wake_up(); print("[parity] sleep/wake cycle done", flush=True)
    if A.adapter:
        from apply_adapter import apply_adapter
        apply_adapter(llm, A.adapter, A.path)
    sp = SamplingParams(temperature=0.0, max_tokens=A.max_new, logprobs=0)   # logprobs=0: the chosen token's logprob
    ctk = {} if A.think is None else {"chat_template_kwargs": {"enable_thinking": A.think == "on"}}
    res = {"gens": {}, "lps": {}, "tps": {}, "secs": {}, "swaps": {},
           "cfg": {k: os.environ.get(k) for k in ("TEMPORAL_WALKER", "TEMPORAL_EAGER", "TEMPORAL_RHO")}}
    for arm in (("free",) if A.no_hooks else ("free", f"R{A.R}")):
        DEC.update(on=arm != "free", R=A.R, swaps=1)
        DEC["state"].clear()
        if A.warm:
            llm.chat(msgs[:A.warm], sp, use_tqdm=False, **ctk)   # warm (graph capture is at init)
        DEC["state"].clear()
        t0 = time.time()
        outs = llm.chat(msgs, sp, use_tqdm=False, **ctk)
        secs = time.time() - t0
        gens = [list(o.outputs[0].token_ids) for o in outs]
        res["lps"][arm] = [[d[t].logprob for t, d in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)] for o in outs]
        ntok = sum(len(g) for g in gens)
        sw, rows, rate = TR.swap_stats() if not A.no_hooks else (0, 0, 0.0)
        res["gens"][arm] = gens; res["secs"][arm] = secs; res["tps"][arm] = ntok / secs
        res["swaps"][arm] = rate
        print(f"[parity] {arm}: {ntok} tokens in {secs:.1f}s = {ntok/secs:.0f} tok/s; "
              f"swaps/token {rate:.4f} ({sw}/{rows})", flush=True)
    json.dump(res, open(A.out, "w"))
    print(f"[parity] wrote {A.out}")


if __name__ == "__main__":
    main()
