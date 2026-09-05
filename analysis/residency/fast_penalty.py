"""Presence penalty as a vLLM V1 logits processor with a persistent per-row vocab mask.

vLLM's native path rebuilds, every decode step, a padded copy of every request's output tokens
(host -> device) and a (num_seqs, vocab) scatter_add count; at batch 256 on a 248k vocab that
halves throughput (qwen: 4171 -> 2220 tok/s, 2026-08-29). The math is only
    logits[row, tok] -= pp   for every tok that has appeared in row's OUTPUT so far,
so keep a (max_num_seqs, vocab) 0/1 mask on the device, set the few new entries per step, and
apply one fused addcmul_ over the logits. Prompt tokens are excluded, exactly as in vLLM.

Use: LLM(..., logits_processors=[FastPresencePenalty]) and per request
     SamplingParams(presence_penalty=0.0, extra_args={"fast_presence_penalty": 1.5}).
"""
import torch
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor, MoveDirectionality

KEY = "fast_presence_penalty"


class FastPresencePenalty(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.max_reqs = vllm_config.scheduler_config.max_num_seqs
        self.vocab = vllm_config.model_config.get_vocab_size()
        self.mask = None                                    # (max_reqs, vocab) in the logits dtype, lazily
        self.pp = torch.zeros(self.max_reqs, device=device, dtype=torch.float32)
        self.rows: dict[int, list] = {}                     # index -> [pp, output_tok_ids (live ref), n_seen]
        self.active = False

    def is_argmax_invariant(self) -> bool:
        return False

    def _ensure(self, dtype):
        if self.mask is None or self.mask.dtype != dtype:
            self.mask = torch.zeros(self.max_reqs, self.vocab, device=self.device, dtype=dtype)

    def update_state(self, batch_update: BatchUpdate | None):
        if batch_update:
            self._ensure(torch.float32)
            for index, params, _prompt, out in batch_update.added:
                pp = float((params.extra_args or {}).get(KEY, 0.0) or 0.0)
                self.mask[index].zero_()
                if pp:
                    self.rows[index] = [pp, out, 0]; self.pp[index] = pp
                else:
                    self.rows.pop(index, None); self.pp[index] = 0.0
            for index in batch_update.removed:
                if self.rows.pop(index, None) is not None:
                    self.pp[index] = 0.0; self.mask[index].zero_()
            for a, b, direct in batch_update.moved:
                ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
                if direct == MoveDirectionality.SWAP:
                    self.mask[[a, b]] = self.mask[[b, a]]
                    self.pp[[a, b]] = self.pp[[b, a]]
                    if ra is not None: self.rows[b] = ra
                    if rb is not None: self.rows[a] = rb
                else:                                       # unidirectional a -> b
                    self.mask[b] = self.mask[a]; self.pp[b] = self.pp[a]
                    self.mask[a].zero_(); self.pp[a] = 0.0
                    if ra is not None: self.rows[b] = ra
            self.active = bool(self.rows)
        if not self.rows:
            return
        rr, tt = [], []
        for index, st in self.rows.items():
            out = st[1]; n = len(out); k = st[2]
            # vLLM appends a -1 placeholder for the token being sampled and fills it in later:
            # consume only real tokens and never advance past a placeholder.
            while k < n and out[k] >= 0:
                rr.append(index); tt.append(out[k]); k += 1
            st[2] = k
        if rr:
            self.mask[torch.tensor(rr, device=self.device, dtype=torch.long),
                      torch.tensor(tt, device=self.device, dtype=torch.long)] = 1.0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return logits
        n = logits.shape[0]
        if self.mask.dtype != logits.dtype:
            self.mask = self.mask.to(logits.dtype)
        logits.addcmul_(self.mask[:n], self.pp[:n, None].to(logits.dtype), value=-1.0)
        return logits


def sampling_kwargs(pp: float, fast: bool):
    """Sampling kwargs for a presence penalty: vLLM native, or the fast processor."""
    if not pp:
        return {}
    return {"presence_penalty": 0.0, "extra_args": {KEY: pp}} if fast else {"presence_penalty": pp}
