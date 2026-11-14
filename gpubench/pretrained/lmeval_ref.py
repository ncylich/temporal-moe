#!/usr/bin/env python3
"""lm-eval on the plain-PyTorch reference forward (scripts/serving_convert/ref_model.py), so every
routing rule (stock top-k, our window, Skliar) is scored by the same code on the same weights.
Batched for throughput; routing is per sequence (each row's rule runs over its own tokens only, and
right padding cannot reach earlier tokens because attention and both rules are causal).

Reuses analysis/probes/run_lmeval.py's import shims and dataset-name mapping by importing it, then
registers model `refmoe` and hands off to lm_eval's CLI.

    python gpubench/pretrained/lmeval_ref.py --model refmoe \
        --model_args weights=/root/m/moe_fine.safetensors,routing=skliar,cb_lambda=0.5,batch=64 \
        --tasks arc_easy --num_fewshot 0 --output_path out/
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "analysis", "probes"), os.path.join(ROOT, "scripts", "serving_convert")]
import run_lmeval  # noqa: E402,F401  (shims only; its CLI call is under __main__)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lm_eval.api.model import TemplateLM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from ref_model import RefModel  # noqa: E402

SEQ = 2048
EOD = 0          # pythia <|endoftext|>, Megatron's eod for these runs


@register_model("refmoe")
class RefMoE(TemplateLM):
    def __init__(self, weights, routing="topk", topk=18, cb_C=18, cb_lambda=0.5, batch=64,
                 tokenizer="EleutherAI/pythia-12b", loads_out=None, **kw):
        super().__init__()
        self.routing, self.bs, self.loads_out = routing, int(batch), loads_out
        self.m = RefModel(weights, int(topk), device="cuda", cb_C=int(cb_C), cb_lam=float(cb_lambda))
        self.tok = Tokenizer.from_pretrained(tokenizer)

    @property
    def eot_token_id(self):
        return EOD

    def tok_encode(self, string, **kw):
        return self.tok.encode(string, add_special_tokens=False).ids

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # requests: [((ctx, cont), ctx_ids, cont_ids)]; score longest first so batches pack tightly
        items = []
        for i, (_, c, k) in enumerate(requests):
            ids = (c + k)[-(SEQ + 1):]
            items.append((i, ids, len(k)))
        items.sort(key=lambda x: -len(x[1]))
        out = [None] * len(items)
        for b in range(0, len(items), self.bs):
            chunk = items[b:b + self.bs]
            L = max(len(x[1]) - 1 for x in chunk)
            inp = torch.full((len(chunk), L), EOD, dtype=torch.long)
            valid = torch.zeros((len(chunk), L), dtype=torch.bool)
            for j, (_, ids, _) in enumerate(chunk):
                inp[j, :len(ids) - 1] = torch.tensor(ids[:-1]); valid[j, :len(ids) - 1] = True
            self.m.valid = valid.cuda()
            logp = F.log_softmax(self.m.forward(inp.cuda(), self.routing).float(), -1)
            for j, (i, ids, n) in enumerate(chunk):
                T = len(ids) - 1
                lp = logp[j, T - n:T]                                    # predicts the continuation
                tgt = torch.tensor(ids[-n:], device=lp.device)
                out[i] = (float(lp.gather(1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
            print(f"[refmoe] {min(b + self.bs, len(items))}/{len(items)}", file=sys.stderr, flush=True)
        if self.loads_out:
            with open(self.loads_out, "a") as f:
                f.write(json.dumps({"routing": self.routing, "loads": self.m.loads[0], "token_layers": self.m.loads[1]}) + "\n")
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("no rolling-loglikelihood task is in the suite")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("no generation task is in the suite")


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
