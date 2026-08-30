#!/usr/bin/env python3
"""CE adaptation of gemma4-26B-A4B-IT to decode-time residency (R=k=8, response tokens only).

Plain cross-entropy on the model's OWN vLLM-generated responses (gemma's greedy outputs are
low-entropy, so hard labels ~ soft labels and distillation buys nothing). The constraint is
enforced exactly as served: prefill free (scan observes the prompt), R=8 from the first
response token, warm; with --micro-batch > 1 rows are length-sorted, padded, and the rule
is applied per row (scan batch columns are independent; trailing pads sit after each row's
response and touch nothing scored). Loss on response tokens only.

Trainable surface: attention LoRA r32 + router projections + RMSNorm gains, plus optional
per-expert LoRA on the 3D expert tensors (--expert-lora-r) via a grouped-GEMM forward
(torch._grouped_mm; 98 -> 2900 tok/s vs the stock eager expert loop at micro-batch 8).

Loads via unsloth when its patches keep our Gemma4TextRouter hook alive; every load asserts
constraint engagement. --smoke runs the full regression gauntlet (grouped-path parity vs the
eager loop, LoRA engagement/restore, batched-plumbing exactness, free/constrained batch
parity, gradient flow, timed steps, save/reload) and exits.

    train_gemma_ce.py --traj gemma4_train5k --tokens 3400000 \
        --expert-lora-r 16 --out .../gemma_ce_expert_adapter.pt
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_expert_lora(model, r):
    """Per-expert LoRA on the 3D expert tensors via torch._grouped_mm.

    The stock HF forward is a Python loop over hit experts (~1500 dispatches per
    layer pass -- overhead-bound, 98 tok/s measured). Here (token, expert) pairs
    are sorted by expert once per forward and each projection becomes ONE grouped
    GEMM over all experts; LoRA deltas are two more grouped GEMMs. Base expert
    weights are transposed IN PLACE to the grouped layout (E, in, out) -- they are
    frozen, our forward is the only consumer, and merge transposes back.
    B zero-init: delta starts at exactly 0."""
    import torch.nn as nn
    patched = 0
    for mod in model.modules():
        gu = getattr(mod, "gate_up_proj", None)
        dp = getattr(mod, "down_proj", None)
        if not (isinstance(gu, nn.Parameter) and gu.dim() == 3
                and isinstance(dp, nn.Parameter) and dp.dim() == 3):
            continue
        E, twoI, H = gu.shape                  # stored (E, 2I, H)
        _, H2, I = dp.shape                    # stored (E, H, I)
        with torch.no_grad():                  # -> grouped layout (E, in, out)
            gu.data = gu.data.transpose(1, 2).contiguous()      # (E, H, 2I)
            dp.data = dp.data.transpose(1, 2).contiguous()      # (E, I, H)
        dev, dt = gu.device, gu.dtype
        mod.elora_gu_A = nn.Parameter(torch.randn(E, H, r, device=dev, dtype=dt) / r)
        mod.elora_gu_B = nn.Parameter(torch.zeros(E, r, twoI, device=dev, dtype=dt))
        mod.elora_dp_A = nn.Parameter(torch.randn(E, I, r, device=dev, dtype=dt) / r)
        mod.elora_dp_B = nn.Parameter(torch.zeros(E, r, H2, device=dev, dtype=dt))
        mod.elora_scale = 2.0                  # alpha/r with alpha = 2r

        def fwd(self, hidden_states, top_k_index, top_k_weights):
            T, k = top_k_index.shape
            Emax = self.num_experts
            flat_e = top_k_index.reshape(-1)
            keep = flat_e < Emax               # sentinel guard (reference loop skips E)
            if not bool(keep.all()):
                flat_e = flat_e[keep]
            order = torch.argsort(flat_e, stable=True)
            src_idx = (torch.arange(T * k, device=flat_e.device)[keep]
                       if not bool(keep.all())
                       else torch.arange(T * k, device=flat_e.device))[order]
            tok = src_idx // k
            x = hidden_states.index_select(0, tok)
            offs = torch.bincount(
                flat_e[order], minlength=Emax).cumsum(0).to(torch.int32)
            s = self.elora_scale if getattr(self, "elora_on", True) else 0.0
            g = torch._grouped_mm
            gate_up = g(x, self.gate_up_proj, offs=offs)
            if s:
                gate_up = gate_up + s * g(
                    g(x, self.elora_gu_A, offs=offs), self.elora_gu_B, offs=offs)
            gate, up = gate_up.chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            down = g(h, self.down_proj, offs=offs)
            if s:
                down = down + s * g(
                    g(h, self.elora_dp_A, offs=offs), self.elora_dp_B, offs=offs)
            w = top_k_weights.reshape(-1)[src_idx].unsqueeze(1)
            # deterministic combine: one unique slot per (token, expert-slot) pair
            # (a single index_add_ over duplicate token ids would use atomics --
            # nondeterministic run-to-run; the smoke restore-gate caught this)
            contrib = torch.zeros(T * k, hidden_states.shape[1],
                                  device=down.device, dtype=down.dtype)
            contrib = contrib.index_copy(0, src_idx, down * w)
            return contrib.view(T, k, -1).sum(1).to(hidden_states.dtype)

        import types
        mod.forward = types.MethodType(fwd, mod)
        patched += 1
    assert patched, "no 3D expert tensors found to patch"
    print(f"[gce] expert LoRA r={r} (grouped_mm path) on {patched} layers", flush=True)
    return patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/dev/shm/gemma4-26b-it")
    ap.add_argument("--traj", default="gemma4_train5k")
    ap.add_argument("--tokens", type=int, default=3_400_000, help="response-token budget")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--router-only", action="store_true",
                    help="BASELINE #2 (ReMoE): train ONLY the router projections, freezing "
                         "attention LoRA, expert LoRA and norms. Their method is a post-hoc "
                         "router finetune, so giving it our full adaptation surface would "
                         "compare capacity rather than objective.")
    ap.add_argument("--remoe-lambda", type=float, default=0.0,
                    help="weight on the ReMoE recency-reuse objective (0 = off). Their "
                         "recipe trains with the residency constraint OFF, so pair this "
                         "with --no-constraint for the faithful remake.")
    ap.add_argument("--remoe-gamma", type=float, default=0.9,
                    help="recency decay for the reuse objective; 0.9 ~ a 10-token window")
    ap.add_argument("--data-seed", type=int, default=0,
                    help="permutation seed for batch order. Default 0 reproduces every "
                         "run made before 2026-08-26. Vary it to get a training-variance "
                         "replicate: the paired McNemar error bar on a benchmark covers "
                         "question sampling only and says nothing about run-to-run spread.")
    ap.add_argument("--extra-lr-div", type=float, default=5.0,
                    help="router/norm full-weight lr = lr / this")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--out", default="/workspace/olmoe-adapt/data/gemma_ce_adapter.pt")
    ap.add_argument("--resume", action="store_true",
                    help="load tensors+seen from --out and continue (fresh Adam)")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="step counter for ckpts saved before 'step' was stored")
    ap.add_argument("--R", type=int, default=8,
                    help="residency budget during training (half-expert units on "
                         "split checkpoints)")
    ap.add_argument("--no-constraint", action="store_true",
                    help="CONTROL: identical run with residency OFF during training "
                         "(isolates constraint-aware adaptation from plain self-SFT)")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --out adapter, score frozen-500 self-CE (free and R8), exit")
    ap.add_argument("--expert-lora-r", type=int, default=0,
                    help="per-expert LoRA rank on the 3D expert tensors (0 = off); "
                         "delta applied per HIT expert inside the loop -- never "
                         "materialises the full-tensor delta")
    ap.add_argument("--max-seq", type=int, default=1024,
                    help="loader max sequence length; think-on trajectories need "
                         "2048 (prompt 512 + think+answer 1024+)")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="rows per forward; rows are length-sorted into chunks and "
                         "padded to the chunk max. Constraint applied per row via "
                         "CFG batch/enforce_from-vector. 16 rows per optimizer "
                         "step regardless (accum adjusts)")
    ap.add_argument("--precompute-kl", default=None,
                    help="forward-only pass storing the BASE model's top-50 free-"
                         "routing logprobs per response token over the trajectory "
                         "set; feeds --kl-anchor. Run on the UNADAPTED model")
    ap.add_argument("--kl-anchor", default=None,
                    help="path from --precompute-kl: adds kl-weight * KL(student "
                         "free-mode || base top-50) on response tokens (anti-"
                         "forgetting anchor)")
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--kl-only", action="store_true",
                    help="distillation phase: skip the CE term entirely; train on "
                         "the KL-to-ref alone (use with self-generated trajectories)")
    ap.add_argument("--kl-arm", choices=("free", "constrained"), default="free",
                    help="which arm the KL forward runs on. 'free' anchors the "
                         "unconstrained arm (adaptation default); 'constrained' "
                         "distills the constrained-arm distribution toward the ref "
                         "on the trajectory's own states")
    ap.add_argument("--aux-traj", default=None,
                    help="second trajectory set (self-generated UNDER the constraint) used only for an "
                         "on-policy KL term: constrained-arm student vs --aux-kl-anchor (free base) on the "
                         "student's own prefixes. CE and the free-arm anchor stay on --traj")
    ap.add_argument("--aux-kl-anchor", default=None, help="--precompute-kl output over --aux-traj")
    ap.add_argument("--aux-kl-weight", type=float, default=1.0)
    ap.add_argument("--online-every", type=int, default=0,
                    help="ON-POLICY: every N optimizer steps, sync the current adapter into an "
                         "in-process vLLM engine, sample --online-n prompts under the constraint, "
                         "label them with the frozen base in-process, and make them the aux set "
                         "(replaces --aux-traj/--aux-kl-anchor). 0 = off")
    ap.add_argument("--online-n", type=int, default=256, help="rows sampled per refresh (16 steps x 16 rows)")
    ap.add_argument("--online-prompts", default="/workspace/olmoe-adapt/data/d7_prompts.jsonl")
    ap.add_argument("--online-quota", default="mathlane_v2=2341,d5_fewshot=1183,domain8k=1000")
    ap.add_argument("--online-max-new", type=int, default=1024)
    ap.add_argument("--online-temp", type=float, default=0.7, help="student sampling temperature")
    ap.add_argument("--budget-on", choices=("data", "sampled"), default="data",
                    help="what --tokens counts: the D7 rows walked (data) or the on-policy tokens trained on (sampled). "
                         "With --kl-only and no --kl-anchor, 'sampled' also skips the D7 walk entirely")
    ap.add_argument("--online-gpu-mem", type=float, default=0.5, help="vLLM share of GPU memory (fraction of total)")
    ap.add_argument("--log-every", type=int, default=50, help="steps between [gce] step lines (each also reports the window's wall time per step)")
    ap.add_argument("--online-presence-penalty", type=float, default=0.0, help="sampler presence penalty (qwen card: 1.5; halves vLLM throughput)")
    ap.add_argument("--save-opt", action="store_true", default=True, help="store the AdamW state in the adapter file so --resume continues as one long run")
    ap.add_argument("--no-save-opt", dest="save_opt", action="store_false")
    ap.add_argument("--online-think", choices=("on", "off"), default="off", help="sampler generates with the model's thinking mode")
    ap.add_argument("--online-offload", type=int, default=0,
                    help="expert layers whose frozen base weights sit on the host while the engine is awake (qwen35: 12)")
    ap.add_argument("--online-smoke", default=None,
                    help="path of a parity_vllm.py dump made from the MERGED checkpoint of the adapter this "
                         "run resumes from: sync once, generate the same prompts greedily, compare tokens, exit")
    ap.add_argument("--aux-kl-temp", type=float, default=1.0,
                    help="KL temperature for revkl_full: teacher top-50 and student logits softened by T and "
                         "renormalised over the top-50 support (tail term dropped for T != 1), loss x T^2")
    ap.add_argument("--aux-loss", choices=("fwdkl", "revkl", "revkl_full"), default="fwdkl",
                    help="fwdkl: teacher-weighted KL over the teacher top-50 at the student's states. "
                         "revkl: reverse KL, sampled-token estimator: loss = -sum_t A_t log p_s(y_t), "
                         "A_t = log p_teacher(y_t) - log p_student(y_t) held fixed (the on-policy "
                         "distillation objective; needs a --precompute-kl file that stores the "
                         "teacher log-prob at the sampled token). revkl_full: ANALYTIC reverse KL at "
                         "the student's states over the teacher's top-50 support plus a tail term: "
                         "sum_{y in top50} p_s(y)(log p_s(y) - log p_t(y)) + m_s (log m_s - log m_t), "
                         "m = mass outside the top-50; exact given the stored teacher, low variance")
    ap.add_argument("--precompute-tokw", default=None,
                    help="forward-only pass over all trajectories computing per-"
                         "response-token CE under free and R8 routing; saves "
                         "{row_index: (ce_free fp16, ce_r8 fp16)} to PATH, then "
                         "exits. Feeds --tok-weights")
    ap.add_argument("--digit-weight", type=float, default=1.0,
                    help="multiply the CE loss on DIGIT tokens by this factor. Failure "
                         "analysis (REBUILD_RESULTS.md, 2026-08-27): residency breaks the "
                         "arithmetic primitive (5+4+2=8) while digit tokens are 6.3% of the "
                         "response and digits after '=' 0.66%, so plain CE spends >99% of its "
                         "gradient on tokens that do not fail. 1.0 = off (bit-identical).")
    ap.add_argument("--tok-weights", default=None,
                    help="path from --precompute-tokw: weight response-token CE by "
                         "w = 1 + 2*clip(ce_R8 - ce_free, 0, 3) (constraint-"
                         "disagreement weighting)")
    ap.add_argument("--smoke-tol", type=float, default=0.02,
                    help="constrained batch-parity gate tolerance (relax only for "
                         "pair-degenerate half-grain inits; see gate comment)")
    ap.add_argument("--smoke", action="store_true",
                    help="engagement checks + 2 timed steps + save/reload, then exit")
    ap.add_argument("--merge-scale", type=float, default=1.0,
                    help="scale the adapter delta at merge: LoRA B tensors are "
                         "multiplied by s; full-weight tensors (router/norm) are "
                         "interpolated base*(1-s)+ckpt*s. 1.0 = full adapter")
    ap.add_argument("--merge-out", default=None,
                    help="after loading the adapter, save the merged model to this dir and exit")
    ap.add_argument("--family", default="gemma4", choices=("gemma4", "qwen35"),
                    help="router-patch family; everything else is layout-generic")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "adamw8bit", "paged8bit"),
                    help="paged8bit: bnb paged 8-bit moments (crossmodel qwen precedent)")
    ap.add_argument("--no-unsloth", action="store_true",
                    help="force the HF+peft stack (qwen35: unsloth's batched constrained "
                         "path drifts 4.9%% where plain HF shows 0.0-0.3%%)")
    A = ap.parse_args()

    assert not (A.expert_lora_r and A.out.endswith("gemma_ce_adapter.pt")), \
        "expert-LoRA run would overwrite the attention-only adapter; pass --out"
    rows = torch.load(f"/workspace/instruct-traj/{A.traj}.pt", weights_only=False)["rows"]
    print(f"[gce] {len(rows)} trajectories", flush=True)

    import granularity_ladder as GL

    # Select a non-cuDNN SDPA backend BEFORE either stack loads. This call already
    # existed in the HF+peft branch below, for a memory reason; it is hoisted here
    # because the unsloth branch needs it too and never had it. On a mismatched cuDNN
    # (this pod: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH out of
    # scaled_dot_product_attention) the cuDNN fused-attention backend fails outright and
    # takes gemma's unsloth path down before the smoke can run. "cuDNN SDP off" is
    # already a documented accommodation of the published qwen recipe, so this is the
    # known-good configuration rather than a new one. Numerics-neutral: flash and
    # mem-efficient SDPA compute the same attention.
    torch.backends.cuda.enable_cudnn_sdp(False)

    use_unsloth = not A.no_unsloth
    try:
        if A.no_unsloth:
            raise ImportError("--no-unsloth: forcing HF+peft stack")
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(A.model, max_seq_length=A.max_seq,
                                               dtype=torch.bfloat16, load_in_4bit=False,
                                               full_finetuning=False)
        tok = getattr(tok, "tokenizer", tok)
        model = FastModel.get_peft_model(model, r=32, lora_alpha=64, lora_dropout=0.0,
                                         use_gradient_checkpointing=True,
                                         **({"target_modules": ["q_proj", "k_proj",
                                                                "v_proj", "o_proj"]}
                                            if A.family == "qwen35" else {}))
        # qwen3.5: unsloth's default targets hit the DeltaNet projections too
        # (~1.9B extra trainable at r32 x 40 layers); gemma's surface is q/k/v/o
    except Exception as e:
        print(f"[gce] unsloth path failed ({type(e).__name__}: {e}); falling back to HF+peft",
              flush=True)
        use_unsloth = False
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        tok = AutoTokenizer.from_pretrained(A.model)
        model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16).to("cuda")
        model.gradient_checkpointing_enable()
        # cuDNN fused-attention backward needs a workspace it cannot get at
        # <1GB headroom (mha_graph.execute failure at 81.1/81.6GB); flash/
        # mem-efficient SDPA backends are leaner
        torch.backends.cuda.enable_cudnn_sdp(False)
        # Target the LANGUAGE model's attention projections by regex, and never the
        # vision tower. gemma4 wraps ONLY the vision tower's projections in
        # Gemma4ClippableLinear (vision hidden 1152); the language model's q/k/v/o are
        # plain nn.Linear. A bare ["q_proj", ...] therefore hits the wrapper and raises,
        # while retargeting to "q_proj.linear" silently does the opposite: it matches the
        # vision tower ONLY, so a text-only run puts LoRA where no gradient ever flows and
        # every lora_B stays at its zero init. That is invisible -- training completes,
        # the adapter has 108 attention tensors, and all of them are exactly zero, so the
        # merged model carries expert-LoRA and no attention LoRA at all. Anchor on
        # language_model instead, and assert the attachment rather than trusting it.
        # Enumerate the ACTUAL text-side attention projections rather than assuming a
        # naming convention. Two architectures, two shapes: gemma4 nests its text stack
        # under language_model AND wraps only the VISION tower's projections in
        # Gemma4ClippableLinear, while qwen3.5 has no language_model prefix at all -- a
        # regex tuned for one raises "target modules not found" on the other. Selecting
        # concrete nn.Linear module names that are not under a vision tower works for both
        # and cannot silently land on modules a text-only forward never reaches, which is
        # exactly how the attention LoRA once trained to all-zero on the vision tower.
        import torch.nn as _nn
        _names = [n for n, m in model.named_modules()
                  if isinstance(m, _nn.Linear)
                  and n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
                  and "vision_tower" not in n and "visual" not in n]
        assert _names, "found no text-side attention projections to attach LoRA to"
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0, target_modules=_names))
        _lora = [n for n, q in model.named_parameters() if q.requires_grad and "lora_" in n]
        assert _lora, "attention LoRA attached to nothing trainable"
        assert not any("vision_tower" in n or "visual" in n for n in _lora), \
            "attention LoRA leaked onto the vision tower"
        print(f"[gce] attention LoRA on {len(_names)} text-side projections "
              f"({len(_lora)} trainable tensors)", flush=True)

    if A.family == "qwen35":
        GL.patch_qwen35()
        n_routers = GL.tag_qwen35(model)
    else:
        GL.patch_gemma4()
        n_routers = GL.tag_gemma4(model)
    assert n_routers, f"no routers tagged for family {A.family}"
    PAD = tok.pad_token_id or 0

    def make_batch(rs, ridx=None):
        S = max(r["ids"].shape[0] for r in rs)
        B = len(rs)
        ids = torch.full((B, S), PAD, dtype=torch.long)
        am = torch.zeros((B, S), dtype=torch.long)
        tgt = torch.full((B, S), -100, dtype=torch.long)
        plens, ntok = [], 0
        for b, r_ in enumerate(rs):
            L = r_["ids"].shape[0]
            pl = int(r_["prompt_len"])
            ids[b, :L] = r_["ids"]
            am[b, :L] = 1
            tgt[b, pl:L] = r_["ids"][pl:L]
            plens.append(pl)
            ntok += L - pl
        return (ids.to("cuda"), am.to("cuda"), tgt.to("cuda"), plens, ntok)

    KLREF = None
    if A.kl_anchor:
        KLREF = torch.load(A.kl_anchor, weights_only=False)
        print(f"[gce] KL anchor loaded for {len(KLREF)} rows "
              f"(weight {A.kl_weight})", flush=True)

    AUX = None
    if A.aux_traj:
        AUX = torch.load(f"/workspace/instruct-traj/{A.aux_traj}.pt", weights_only=False)["rows"]
        AUXREF = torch.load(A.aux_kl_anchor, weights_only=False)
        print(f"[gce] aux on-policy KL: {len(AUX)} rows, ref {len(AUXREF)} rows, weight {A.aux_kl_weight}, "
              f"constrained arm", flush=True)
    TOKW = None
    if A.tok_weights:
        TOKW = torch.load(A.tok_weights, weights_only=False)
        print(f"[gce] disagreement weights loaded for {len(TOKW)} rows "
              f"(w = 1 + 2*clip(ce_R8 - ce_free, 0, 3))", flush=True)

    DIGW = None
    if A.digit_weight != 1.0:
        import re as _re
        _V = len(tok)
        DIGW = torch.ones(_V, device="cuda")
        _n = 0
        for _i in range(_V):
            if _re.fullmatch(r"\s?\d+", tok.decode([_i])):
                DIGW[_i] = A.digit_weight; _n += 1
        print(f"[gce] digit-weight {A.digit_weight} on {_n} digit token ids", flush=True)

    DEC = HEAD = None
    if A.no_unsloth:
        # chunked-head path: never materialise [B,S,V] logits (2.5GB bf16 at
        # qwen's 248k vocab + graph). Decoder forward keeps [B,S,H] (~21MB);
        # the head is applied inside checkpointed 512-token CE/KL slices.
        _causal = model.base_model.model if hasattr(model, "base_model") else model
        DEC, HEAD = _causal.model, _causal.lm_head

    def batch_ce_hid(hid, tgt, scale):
        targets = tgt[:, 1:]
        num = den = 0
        for b in range(targets.shape[0]):
            m_ = targets[b] != -100
            if not bool(m_.any()):
                continue
            hb, tb = hid[b][m_], targets[b][m_]
            if DIGW is None:
                for j in range(0, tb.shape[0], 512):
                    num = num + torch.utils.checkpoint.checkpoint(
                        lambda h_, t_: torch.nn.functional.cross_entropy(
                            HEAD(h_).float(), t_, reduction="sum"),
                        hb[j:j + 512], tb[j:j + 512], use_reentrant=False)
                den += int(m_.sum())
            else:
                # weighted mean: digit tokens carry --digit-weight, everything else 1
                for j in range(0, tb.shape[0], 512):
                    num = num + torch.utils.checkpoint.checkpoint(
                        lambda h_, t_: (torch.nn.functional.cross_entropy(
                            HEAD(h_).float(), t_, reduction="none") * DIGW[t_]).sum(),
                        hb[j:j + 512], tb[j:j + 512], use_reentrant=False)
                den += float(DIGW[tb].sum())
        return num / den / scale

    def batch_ce(logits, tgt, scale, ridx=None):
        """Mean response-token CE over a padded batch; per-row float() keeps the
        [B,S,V] float materialisation bounded. With --tok-weights, tokens where
        the BASE model's constrained CE exceeds its free CE are upweighted."""
        targets = tgt[:, 1:]
        num = den = 0
        for b in range(targets.shape[0]):
            m_ = targets[b] != -100
            if not bool(m_.any()):
                continue
            if TOKW is None or ridx is None:
                # chunked + checkpointed: the fp32 [resp,V] cast (2GB/row at
                # 248k vocab) is recomputed per 512-token slice in backward
                lgb_, tg_ = logits[b][m_], targets[b][m_]
                for j in range(0, tg_.shape[0], 512):
                    if DIGW is None:
                        num = num + torch.utils.checkpoint.checkpoint(
                            lambda lg_, t_: torch.nn.functional.cross_entropy(
                                lg_.float(), t_, reduction="sum"),
                            lgb_[j:j + 512], tg_[j:j + 512], use_reentrant=False)
                    else:
                        num = num + torch.utils.checkpoint.checkpoint(
                            lambda lg_, t_: (torch.nn.functional.cross_entropy(
                                lg_.float(), t_, reduction="none") * DIGW[t_]).sum(),
                            lgb_[j:j + 512], tg_[j:j + 512], use_reentrant=False)
                den += int(m_.sum()) if DIGW is None else float(DIGW[tg_].sum())
                continue
            ce = torch.nn.functional.cross_entropy(
                logits[b][m_].float(), targets[b][m_], reduction="none")
            if TOKW is not None and ridx is not None:
                cf, cr = TOKW[ridx[b]]
                assert cf.shape[0] == ce.shape[0], \
                    f"weight len {cf.shape[0]} vs resp len {ce.shape[0]} row {ridx[b]}"
                w = 1.0 + 2.0 * (cr.float() - cf.float()).clamp(0, 3)
                w = w.to(ce.device)
                num = num + (ce * w).sum()
                den += float(w.sum())
            else:
                num = num + ce.sum()
                den += int(m_.sum())
        return num / den / scale

    ORIG_EXTRA = None

    def teacher_ref(rows_, adapted=False):
        """Frozen free-base top-50 log-probs (+ log-prob at the row's own token) per row. With
        adapted=True the live model carries an adapter: LoRAs are switched off and the trained
        router/norm tensors are swapped for their originals for the duration of the pass."""
        import contextlib
        was_training = model.training
        model.eval()
        mb = max(1, A.micro_batch)
        lidx = sorted(range(len(rows_)), key=lambda i: rows_[i]["ids"].shape[0])
        outk = {}
        ctx = model.disable_adapter() if (adapted and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
        if adapted:
            for m_ in model.modules():
                if hasattr(m_, "elora_gu_A"):
                    m_.elora_on = False
            saved = [(p_, p_.data.clone()) for p_, _ in ORIG_EXTRA]
            for p_, o_ in ORIG_EXTRA:
                p_.data.copy_(o_)
        with torch.no_grad(), ctx:
            for c0 in range(0, len(lidx), mb):
                ridx = lidx[c0:c0 + mb]
                rs = [rows_[i] for i in ridx]
                ids, am, tgt, plens, _ = make_batch(rs)
                GL.CFG.update(on=False, enforce_from=0, batch=len(rs))
                if DEC is not None:   # chunked head: full [B,S,V] logits OOM at 4608 seq
                    x = DEC(ids, attention_mask=am).last_hidden_state[:, :-1]
                else:
                    x = model(ids, attention_mask=am).logits[:, :-1]
                targets = tgt[:, 1:]
                for b, ri in enumerate(ridx):
                    m_ = targets[b] != -100
                    xb = x[b][m_]
                    tb = targets[b][m_]
                    tis, tvs, tat = [], [], []
                    for j in range(0, xb.shape[0], 512):
                        lg_ = HEAD(xb[j:j + 512]) if DEC is not None else xb[j:j + 512]
                        lp = torch.log_softmax(lg_.float(), -1)
                        top = lp.topk(50, dim=-1)
                        tis.append(top.indices.to(torch.int32).cpu())
                        tvs.append(top.values.half().cpu())
                        # teacher log-prob at the trajectory's OWN token: the sampled-token
                        # estimator of reverse KL needs it (--aux-loss revkl)
                        tat.append(lp.gather(1, tb[j:j + 512, None]).squeeze(1).float().cpu())
                    outk[ri] = (torch.cat(tis), torch.cat(tvs), torch.cat(tat))
                if (c0 // mb) % 100 == 0 and not adapted:
                    print(f"[gce-kl] {c0}/{len(lidx)}", flush=True)
        if adapted:
            for m_ in model.modules():
                if hasattr(m_, "elora_gu_A"):
                    m_.elora_on = True
            for p_, v_ in saved:
                p_.data.copy_(v_)
        if was_training:
            model.train()
        return outk

    if A.precompute_kl:
        import shutil
        outk = teacher_ref(rows)
        GL.CFG.update(batch=1)
        torch.save(outk, "/tmp/gce_kl_tmp.pt")
        shutil.move("/tmp/gce_kl_tmp.pt", A.precompute_kl)
        print(f"[gce-kl] saved {len(outk)} rows -> {A.precompute_kl} -- DONE",
              flush=True)
        return

    if A.precompute_tokw:
        import shutil
        model.eval()
        mb = max(1, A.micro_batch)
        lidx = sorted(range(len(rows)), key=lambda i: rows[i]["ids"].shape[0])
        outw = {}
        t0 = time.time()
        with torch.no_grad():
            for c0 in range(0, len(lidx), mb):
                ridx = lidx[c0:c0 + mb]
                rs = [rows[i] for i in ridx]
                ids, am, tgt, plens, _ = make_batch(rs)
                per = {}
                for on in (False, True):
                    GL.CFG.update(on=on, R=A.R, enforce_from=plens,
                                  batch=len(rs), cold_start=False)
                    lg = model(ids, attention_mask=am).logits[:, :-1]
                    targets = tgt[:, 1:]
                    for b, ri in enumerate(ridx):
                        m_ = targets[b] != -100
                        ce = torch.nn.functional.cross_entropy(
                            lg[b][m_].float(), targets[b][m_],
                            reduction="none").half().cpu()
                        per.setdefault(ri, []).append(ce)
                for ri, (cf, cr) in per.items():
                    outw[ri] = (cf, cr)
                if (c0 // mb) % 50 == 0:
                    print(f"[gce-tokw] {c0}/{len(lidx)} rows "
                          f"({(time.time()-t0):.0f}s)", flush=True)
        GL.CFG.update(batch=1)
        torch.save(outw, "/tmp/gce_tokw_tmp.pt")
        shutil.move("/tmp/gce_tokw_tmp.pt", A.precompute_tokw)
        up = sum(float(((cr.float()-cf.float()).clamp(0,3) > 0.1).float().mean())
                 for cf, cr in outw.values()) / len(outw)
        print(f"[gce-tokw] saved {len(outw)} rows -> {A.precompute_tokw}; "
              f"mean frac tokens upweighted {up:.3f} -- DONE", flush=True)
        return

    smoke_ref = None
    if A.expert_lora_r:
        if A.smoke:  # reference logits from the UNPATCHED forward, free routing
            _probe = rows[0]["ids"][:256].to("cuda").long().unsqueeze(0)
            with torch.no_grad():
                GL.CFG.update(on=False, enforce_from=0)
                smoke_ref = model(_probe).logits[:, -1].float()
        add_expert_lora(model, A.expert_lora_r)

    # router + norm gains trainable alongside the LoRA
    extra = []
    for n, p in model.named_parameters():
        if ("router" in n.lower() and "proj" in n) or n.endswith("norm.weight") \
                or n.endswith(".mlp.gate.weight"):  # qwen35 router (not shared_expert_gate)
            p.requires_grad_(True)
            extra.append(p)
    if A.router_only:
        # ReMoE is a router-only method; freeze every other trainable surface so the
        # comparison is objective-vs-objective, not capacity-vs-capacity.
        keep = {id(p) for p in extra}
        n_off = 0
        for n, p in model.named_parameters():
            if p.requires_grad and id(p) not in keep:
                p.requires_grad_(False); n_off += 1
        # norms are not the router either -- drop them too
        for n, p in model.named_parameters():
            if p.requires_grad and n.endswith("norm.weight"):
                p.requires_grad_(False); n_off += 1
        print(f"[gce] --router-only: froze {n_off} non-router tensors", flush=True)
    ORIG_EXTRA = [(p, p.data.clone()) for p in extra]      # pristine router/norms for the in-process teacher
    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[gce] stack={'unsloth' if use_unsloth else 'hf+peft'} trainable="
          f"{sum(p.numel() for p in train_params)/1e6:.1f}M (extra router/norm "
          f"{sum(p.numel() for p in extra)/1e6:.1f}M)", flush=True)

    # CONSTRAINT-ENGAGEMENT GATE: constrained forward must differ from free, or the
    # loader's patches ate our hook and training would silently adapt to nothing.
    probe = rows[0]["ids"][:256].to("cuda").long().unsqueeze(0)
    plen = min(int(rows[0]["prompt_len"]), 128)
    with torch.no_grad():
        GL.CFG.update(on=False, enforce_from=0)
        lf = model(probe).logits[:, -1].float()
        GL.CFG.update(on=True, R=A.R, free_set=None, R_map=None, enforce_from=plen,
                      cold_start=False)
        lc = model(probe).logits[:, -1].float()
    d = float((lf - lc).abs().max())
    assert d > 1e-3, f"constraint NOT engaged under this loader (max logit delta {d:.2e})"
    print(f"[gce] constraint engaged (max logit delta {d:.3f})", flush=True)

    if A.smoke:
        # 1) expert-LoRA engagement: B is zero-init, so a forward must be identical
        #    to base; bumping one B must change logits; restoring must restore.
        assert A.expert_lora_r, "--smoke requires --expert-lora-r"
        emods = [m_ for m_ in model.modules() if hasattr(m_, "elora_gu_B")]
        # 0) PARITY: grouped path with B=0 must reproduce the unpatched forward
        with torch.no_grad():
            GL.CFG.update(on=False, enforce_from=0)
            lp = model(probe).logits[:, -1].float()
        pd = float((lp - smoke_ref).abs().max())
        # single-module parity vs the eager loop is 1e-6 in fp32 / 0.008 in bf16;
        # at model level bf16 accumulation reorder compounds over 30 layers, so the
        # gate is semantic (top-1 identical) plus a drift bound.
        top1_ok = bool((lp.argmax(-1) == smoke_ref.argmax(-1)).all())
        # half-grain inits (relaxed --smoke-tol) double the expert count and sit
        # near routing ties, so the accumulation-reorder tail is wider; top-1
        # identity remains the hard requirement.
        pd_lim = 2.0 if A.smoke_tol <= 0.02 else 4.0
        assert top1_ok and pd < pd_lim, \
            f"grouped-path parity FAIL (top1_ok={top1_ok}, max logit diff {pd:.3f})"
        print(f"[gce-smoke] grouped-path parity OK (top-1 identical, max logit "
              f"drift {pd:.3f} bf16)", flush=True)
        with torch.no_grad():
            l0 = model(probe).logits[:, -1].float()
            emods[0].elora_gu_B.add_(0.05)
            l1 = model(probe).logits[:, -1].float()
            emods[0].elora_gu_B.zero_()
            l2 = model(probe).logits[:, -1].float()
        dd, dr = float((l0 - l1).abs().max()), float((l0 - l2).abs().max())
        assert dd > 1e-3, f"expert LoRA NOT engaged (delta {dd:.2e})"
        assert dr < 1e-4, f"expert LoRA restore failed (delta {dr:.2e})"
        print(f"[gce-smoke] expert LoRA engaged (bump delta {dd:.3f}, "
              f"restore {dr:.2e})", flush=True)
        # 1b) BATCH PARITY: per-row response-CE at mb1 vs one padded batch
        pr = [rows[i] for i in (0, 40, 200, 400)]

        def parity(on):
            def row_ce(r_):
                ids = r_["ids"].to("cuda").long().unsqueeze(0)
                pl = int(r_["prompt_len"])
                GL.CFG.update(on=on, R=A.R, enforce_from=pl, batch=1,
                              cold_start=False)
                with torch.no_grad():   # attention_mask=ones: same kernel path
                    lg_ = model(ids, attention_mask=torch.ones_like(ids)) \
                        .logits[0, pl - 1:-1].float()
                return float(torch.nn.functional.cross_entropy(
                    lg_, ids[0, pl:], reduction="mean"))
            ce1 = [row_ce(r_) for r_ in pr]
            ids, am, tgt, plens, _ = make_batch(pr)
            GL.CFG.update(on=on, R=A.R, enforce_from=plens, batch=len(pr),
                          cold_start=False)
            with torch.no_grad():
                lgb = model(ids, attention_mask=am).logits[:, :-1]
            ceb = []
            for b in range(len(pr)):
                m_ = tgt[b, 1:] != -100
                lgm_, tg_ = lgb[b][m_], tgt[b, 1:][m_]
                s_ = sum(float(torch.nn.functional.cross_entropy(
                    lgm_[j:j + 512].float(), tg_[j:j + 512], reduction="sum"))
                    for j in range(0, tg_.shape[0], 512))
                ceb.append(s_ / int(tg_.shape[0]))
            GL.CFG.update(batch=1)
            return ce1, ceb

        # (a0) EXACT plumbing invariant: one row through the batched path
        # (make_batch + CFG batch + enforce_from list) has identical shapes to
        # mb1, so logits must match to numerical identity.
        r0 = pr[1]
        ids0 = r0["ids"].to("cuda").long().unsqueeze(0)
        pl0 = int(r0["prompt_len"])
        GL.CFG.update(on=True, R=A.R, enforce_from=pl0, batch=1, cold_start=False)
        with torch.no_grad():
            la = model(ids0, attention_mask=torch.ones_like(ids0)).logits.float()
        idsb, amb, _, plb, _ = make_batch([r0])
        GL.CFG.update(on=True, R=A.R, enforce_from=plb, batch=1, cold_start=False)
        with torch.no_grad():
            lb = model(idsb, attention_mask=amb).logits.float()
        GL.CFG.update(batch=1)
        d0 = float((la - lb).abs().max())
        assert d0 < 1e-3, f"single-row batched-plumbing mismatch (diff {d0:.2e})"
        print(f"[gce-smoke] batched-plumbing exactness OK (single row diff "
              f"{d0:.2e})", flush=True)
        # (a) constraint OFF, B=4: only batch-SHAPE bf16 drift may remain.
        # Judge in absolute nats (baselines here are near zero).
        c1, cb = parity(on=False)
        ad = max(abs(a - b_) for a, b_ in zip(c1, cb))
        # pair-degenerate half-grain inits (relaxed --smoke-tol): free routing
        # also has near-tied pairs, so batch-shape jitter flips a few free
        # selections too; widen the absolute bound with the same rationale as
        # the constrained gate below.
        assert ad < (0.01 if A.smoke_tol <= 0.02 else 0.03), \
            f"FREE-mode batch parity FAIL (abs drift {ad:.4f} nats): {c1} vs {cb}"
        print(f"[gce-smoke] batch parity, constraint OFF: max abs drift "
              f"{ad:.4f} nats -- mechanics OK", flush=True)
        # (b) constraint ON: resident-set ties flip under bf16 batch-shape drift
        # and cascade discretely, so per-row CE cannot match exactly. Judge the
        # objective in aggregate, report per row.
        c1, cb = parity(on=True)
        m1, mbt = sum(c1) / len(c1), sum(cb) / len(cb)
        rel_on = abs(m1 - mbt) / m1
        print(f"[gce-smoke] batch parity, constraint ON: per-row mb1 "
              f"{['%.3f' % c for c in c1]} vs batched {['%.3f' % c for c in cb]}; "
              f"mean {m1:.4f} vs {mbt:.4f} ({rel_on*100:+.2f}%)", flush=True)
        # Half-grain split inits are pair-degenerate: bf16 batch-shape jitter on
        # router logits flips pair adjudications inside the scan, so mb1 and
        # batched are different fair draws of the SAME trajectory distribution
        # (scan itself verified bit-exact batched-vs-per-row; constraint-off
        # drift ~0.005 nats). --smoke-tol relaxes the gate for those inits;
        # rerun the smoke on the merged model to verify drift returns <2%.
        assert rel_on < A.smoke_tol, f"batched constrained objective drifted: {m1} vs {mbt}"
        # 2) timed optimiser steps, batched exactly like training
        tp = [p for p in model.parameters() if p.requires_grad]
        print(f"[gce-smoke] trainable {sum(p.numel() for p in tp)/1e6:.1f}M "
              f"(expert {sum(p.numel() for m_ in emods for p in (m_.elora_gu_A, m_.elora_gu_B, m_.elora_dp_A, m_.elora_dp_B))/1e6:.1f}M)",
              flush=True)
        sopt = torch.optim.AdamW(tp, lr=1e-5)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        tok_seen, t0 = 0, time.time()
        steady_tok, steady_t = 0, 0.0
        mb_ = max(1, A.micro_batch)
        sl = sorted(range(400), key=lambda i: rows[i]["ids"].shape[0])
        for si in range(6):
            sopt.zero_grad(set_to_none=True)
            t_step, stoks = time.time(), 0
            for bi in range(max(1, 16 // mb_)):
                rs = [rows[i] for i in
                      sl[(si * 2 + bi) * mb_:(si * 2 + bi + 1) * mb_]]
                ids, am, tgt, plens, ntok = make_batch(rs)
                GL.CFG.update(on=True, R=A.R, enforce_from=plens, batch=len(rs),
                              cold_start=False)
                logits = model(ids, attention_mask=am).logits[:, :-1]
                loss = batch_ce(logits, tgt, max(1, 16 // mb_))
                loss.backward()
                tok_seen += ntok
                stoks += ntok
            GL.CFG.update(batch=1)
            # LoRA grad flow: at step 0 B (zero-init) gets grad through A while
            # A's grad is exactly 0 (= B^T g); after B's first update A must flow.
            gB = emods[0].elora_gu_B.grad
            assert gB is not None and float(gB.abs().max()) > 0, "no grad on expert B"
            if si == 1:
                gA = emods[0].elora_gu_A.grad
                assert gA is not None and float(gA.abs().max()) > 0,                     "no grad on expert A after B moved"
            sopt.step()
            torch.cuda.synchronize()
            st = time.time() - t_step
            print(f"[gce-smoke] step {si} loss {loss.item()*2:.4f} "
                  f"{st:.1f}s ({stoks/st:.0f} tok/s)", flush=True)
            if si >= 2:
                steady_tok += stoks
                steady_t += st
        dt_ = time.time() - t0
        print(f"[gce-smoke] STEADY (steps 2-5): {steady_tok/steady_t:.0f} tok/s | "
              f"total {tok_seen} toks {dt_:.1f}s | peak mem "
              f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        # 3) save/reload roundtrip
        named = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad}
        torch.save({"tensors": named}, "/tmp/smoke_expert_adapter.pt")
        ck = torch.load("/tmp/smoke_expert_adapter.pt", map_location="cpu",
                        weights_only=False)
        allp = dict(model.named_parameters())
        miss = [n for n in ck["tensors"] if n not in allp]
        assert not miss, f"roundtrip mismatch: {miss[:3]}"
        n_e = sum(1 for n in ck["tensors"] if "elora" in n)
        print(f"[gce-smoke] roundtrip OK ({len(ck['tensors'])} tensors, "
              f"{n_e} expert-LoRA) -- SMOKE PASS", flush=True)
        return

    if A.eval_only or A.merge_out:
        ck = torch.load(A.out, map_location="cpu", weights_only=False)
        cur_stack = "unsloth" if use_unsloth else "hf+peft"
        assert ck.get("stack") in (None, cur_stack), \
            (f"merge stack mismatch: adapter trained on {ck.get('stack')} but this "
             f"process is on {cur_stack} (transient unsloth failure? retry)")
        named = dict(model.named_parameters())
        # transformers exposes/hides the language_model prefix inconsistently
        # across load paths; match adapter tensors to params by their stable
        # tail (layers.N... / final norm), like the qwen streaming patcher.
        import re as _re

        def _tail(n):
            m = _re.search(r"((?:layers\.\d+\.).*$)", n)
            if m:
                return m.group(1)
            return "model.norm.weight" if n.endswith("model.norm.weight") else n
        tail_map = {}
        for n in named:
            if "vision_tower" in n or "embed_vision" in n:
                continue     # vision params share layers.N tails; never trained
            t = _tail(n)
            if t != n:
                assert t not in tail_map, f"tail collision {t}"
                tail_map[t] = n
        ck["tensors"] = {(tail_map[_tail(n)] if _tail(n) in tail_map else n): v
                         for n, v in ck["tensors"].items()}
        miss = [n for n in ck["tensors"] if n not in named]
        # vision-tower LoRA pairs get attached by unsloth's default targets but
        # receive no gradients in text-only training: B stays zero-init, so the
        # delta is exactly zero and they are droppable. Anything else unmatched
        # is a real error.
        droppable = set()
        for n in miss:
            if ".lora_B." in n and not ck["tensors"][n].count_nonzero():
                droppable.add(n)
                droppable.add(n.replace(".lora_B.", ".lora_A."))
        real_miss = [n for n in miss if n not in droppable]
        assert not real_miss, f"{len(real_miss)} adapter tensors unmatched, e.g. {real_miss[:3]}"
        if droppable:
            print(f"[gce] dropping {len(droppable)} zero-delta LoRA tensors "
                  f"(untrained vision-tower pairs)", flush=True)
            for n in droppable:
                ck["tensors"].pop(n, None)
        s = A.merge_scale
        with torch.no_grad():
            for n, t in ck["tensors"].items():
                t = t.to(named[n].data.device, named[n].dtype)
                if s != 1.0:
                    if "lora_B" in n or "elora_gu_B" in n or "elora_dp_B" in n:
                        t = t * s              # scales the low-rank delta linearly
                    elif "lora_A" not in n and "elora" not in n:
                        # full-weight tensors store absolutes: interpolate to base
                        t = named[n].data * (1 - s) + t * s
                named[n].data.copy_(t)
        if s != 1.0:
            print(f"[gce] adapter merged at scale {s}", flush=True)
        print(f"[gce] adapter loaded (seen={ck['seen']/1e6:.2f}M)", flush=True)
        if A.merge_out:
            with torch.no_grad():
                for mod in model.modules():
                    if hasattr(mod, "elora_gu_A"):
                        # weights live in grouped layout (E, in, out); fold then
                        # transpose back to the checkpoint layout (E, out, in)
                        mod.gate_up_proj.data += mod.elora_scale * torch.bmm(
                            mod.elora_gu_A.data, mod.elora_gu_B.data)
                        mod.down_proj.data += mod.elora_scale * torch.bmm(
                            mod.elora_dp_A.data, mod.elora_dp_B.data)
                        mod.gate_up_proj.data =                             mod.gate_up_proj.data.transpose(1, 2).contiguous()
                        mod.down_proj.data =                             mod.down_proj.data.transpose(1, 2).contiguous()
                        for nm in ("elora_gu_A", "elora_gu_B", "elora_dp_A",
                                   "elora_dp_B"):
                            delattr(mod, nm)
            m = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
            m.save_pretrained(A.merge_out, safe_serialization=True)
            tok.save_pretrained(A.merge_out)
            # save_pretrained does not carry the multimodal processor config
            # every family in this program ships under its OWN filename
            # (gemma4: processor_config.json; qwen3.5: preprocessor_config.json
            # + video_preprocessor_config.json) -- vLLM's engine boot fails
            # without it even for text-only serving. Copy whichever exist from
            # the source checkpoint, so the merged dir is directly vLLM-servable
            # regardless of family (found the gemma4 case the hard way; checked
            # qwen3.5's filenames before it could repeat under a different name).
            import shutil
            for fname in ("processor_config.json", "preprocessor_config.json",
                         "video_preprocessor_config.json"):
                src = os.path.join(A.model, fname)
                if os.path.exists(src):
                    shutil.copy(src, A.merge_out)
                    print(f"[gce] copied {fname} -> {A.merge_out}", flush=True)
            print(f"[gce] merged model -> {A.merge_out}", flush=True)
            return
        ev = torch.load("/workspace/instruct-traj/gemma4_instruct.pt",
                        weights_only=False)["rows"]
        model.eval()
        for label, on in (("free", False), ("R8", True)):
            tot = ntok = 0
            with torch.no_grad():
                for r in ev:
                    ids = r["ids"].to("cuda").long().unsqueeze(0)
                    plen = int(r["prompt_len"])
                    GL.CFG.update(on=on, R=A.R, enforce_from=plen if on else 0,
                                  cold_start=False, free_set=None, R_map=None)
                    lg = model(ids).logits[0].float()
                    tot += float(torch.nn.functional.cross_entropy(
                        lg[plen - 1:-1], ids[0, plen:], reduction="sum"))
                    ntok += ids.shape[1] - plen
            print(f"[gce-eval] {label} self-CE {tot/ntok:.4f} nats/tok "
                  f"(frozen 500, held out)", flush=True)
        return

    extra_ids = {id(p) for p in extra}
    lora_ps = [p for p in train_params if id(p) not in extra_ids]
    opt_cls = torch.optim.AdamW
    if A.opt != "adamw":            # 70GB-weight models: moments cannot live on-GPU
        import bitsandbytes as bnb
        opt_cls = {"adamw8bit": bnb.optim.AdamW8bit,
                   "paged8bit": bnb.optim.PagedAdamW8bit}[A.opt]
    opt = opt_cls([{"params": lora_ps, "lr": A.lr},
                             {"params": extra, "lr": A.lr / A.extra_lr_div}],
                            weight_decay=0.0)
    print(f"[gce] lr groups: lora {A.lr} | router/norm {A.lr / A.extra_lr_div}", flush=True)
    model.train()
    seen = step = 0
    t0 = time.time()
    mb = max(1, A.micro_batch)
    accum_batches = max(1, A.accum // mb)      # 16 rows per optimizer step
    lidx = sorted(range(len(rows)), key=lambda i: rows[i]["ids"].shape[0])
    chunks = [lidx[i:i + mb] for i in range(0, len(lidx), mb)]
    order = torch.randperm(
        len(chunks),
        generator=torch.Generator().manual_seed(A.data_seed)).tolist()
    oi = 0
    if A.resume:
        ck = torch.load(A.out, map_location="cpu", weights_only=False)
        rnamed = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in ck["tensors"].items():
                rnamed[n].data.copy_(t.to(rnamed[n].data.device, rnamed[n].dtype))
        seen = ck["seen"]
        step = ck.get("step", A.resume_step)
        oi = step * accum_batches      # data order is deterministic (seed-0 perm)
        adam = "fresh Adam state"
        if ck.get("opt") is not None:  # continuation = one long run: Adam moments restored
            try:
                opt.load_state_dict(ck["opt"]); adam = "Adam state restored"
            except Exception as e_:
                adam = f"Adam state NOT restored ({type(e_).__name__})"
        print(f"[gce] resumed {A.out}: seen={seen/1e6:.2f}M step={step} ({adam})", flush=True)

    def save():
        # torch.save straight onto the quota-limited network mount produced a
        # short write (inline_container pos mismatch) that killed the first
        # expert run at its step-400 checkpoint. Local write, atomic move.
        import shutil
        named = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                 if p.requires_grad}
        torch.save({"tensors": named, "seen": seen, "step": step, "family": A.family, "stack":
                    "unsloth" if use_unsloth else "hf+peft",
                    "opt": {k: v for k, v in opt.state_dict().items()} if A.save_opt else None,
                    "R": 0 if A.no_constraint else A.R, "expert_lora_r": A.expert_lora_r,
                    "elora_layout": "grouped(E,in,out)",
                    "traj": A.traj, "lr": A.lr}, "/tmp/gce_ckpt_tmp.pt")
        shutil.move("/tmp/gce_ckpt_tmp.pt", A.out)

    AUXS = {"chunks": None, "order": None, "i": 0}
    aux_tot = 0.0; aux_est = 0.0; aux_steps = 0

    def _rebuild_aux(rows_, seed_):
        alidx = sorted(range(len(rows_)), key=lambda i: rows_[i]["ids"].shape[0])
        AUXS["chunks"] = [alidx[i:i + mb] for i in range(0, len(alidx), mb)]
        AUXS["order"] = torch.randperm(len(AUXS["chunks"]), generator=torch.Generator().manual_seed(seed_)).tolist()
        AUXS["i"] = 0

    if AUX is not None:
        _rebuild_aux(AUX, A.data_seed + 1)
    SAMPLER = None
    if A.online_every:
        from online_sampler import OnlineSampler
        SAMPLER = OnlineSampler(model, A.model, A.R, 1, A.online_prompts, A.online_quota,
                                max_new=A.online_max_new, gpu_mem=A.online_gpu_mem, seed=A.data_seed,
                                arch=A.family, temperature=A.online_temp, offload_layers=A.online_offload, presence_penalty=A.online_presence_penalty, think=A.online_think == "on")
        if step:                                     # resumed: continue the prompt pool where the first run stopped
            SAMPLER.cursor = (step // A.online_every) * A.online_n
            print(f"[online] prompt cursor advanced to {SAMPLER.cursor} for the resumed run", flush=True)
        if A.online_smoke:
            import json as _json
            from datasets import load_dataset as _ld
            ref = _json.load(open(A.online_smoke)); n_ref = len(ref["gens"]["free"])
            qs = [r["question"] for r in _ld("openai/gsm8k", "main", split="test")][:n_ref]
            SAMPLER.sync()
            # tensor-level check of the sync against the merged checkpoint on disk (generation-free; gemma names)
            if A.family != "gemma4":
                ref.setdefault("merged_dir", None)
            import glob as _glob
            from safetensors import safe_open as _so
            mdir = ref.get("merged_dir") or ("/root/models/gemma4-digit3-merged" if A.family == "gemma4" else None)
            ck = {}
            for f in (_glob.glob(f"{mdir}/*.safetensors") if mdir else []):
                with _so(f, "pt", device="cpu") as fh:
                    for k in fh.keys():
                        if "layers.0." in k or "layers.29." in k or k.endswith("language_model.norm.weight"):
                            ck[k] = fh.get_tensor(k)
            vp = dict(SAMPLER.vmodel.named_parameters())
            for L_ in ("0", "29"):
                pre = f"model.language_model.layers.{L_}."; hits = 0
                for n_, t_ in vp.items():
                    if f"layers.{L_}." not in n_:
                        continue
                    tail = n_.split(f"layers.{L_}.", 1)[1]
                    want = None
                    if tail.endswith("qkv_proj.weight"):
                        parts = [ck.get(pre + f"self_attn.{x}_proj.weight") for x in "qkv"]
                        want = torch.cat([q for q in parts if q is not None], 0) if parts[0] is not None else None
                    elif tail.endswith("o_proj.weight"):
                        want = ck.get(pre + "self_attn.o_proj.weight")
                    elif tail.endswith("router.proj.weight"):
                        want = ck.get(pre + "router.proj.weight")
                    elif "w13" in tail and tail.endswith("weight") or tail.endswith("gate_up_proj"):
                        want = ck.get(pre + "experts.gate_up_proj")
                    elif ("w2" in tail and tail.endswith("weight")) or tail.endswith("down_proj") and "experts" in tail:
                        want = ck.get(pre + "experts.down_proj")
                    if want is None or want.shape != t_.shape:
                        continue
                    d_ = (t_.detach().float().cpu() - want.float()).abs().max().item(); hits += 1
                    print(f"[online-smoke] tensor {n_}: max|engine - checkpoint| = {d_:.3e}", flush=True)
                if hits == 0:
                    print(f"[online-smoke] layer {L_}: no comparable tensors matched; engine names: "
                          f"{[n for n in vp if f'layers.{L_}.' in n][:12]}", flush=True)
            if A.family == "qwen35":                    # exact streaming compare against the merged W=3 qwen
                from apply_adapter import check_engine
                w_, n_ = check_engine(SAMPLER.vmodel, ref.get("merged_dir") or "/root/models/qwen35-digit3-merged")
                print(f"[online-smoke] qwen tensor check: {n_} tensors, worst max|diff| = {w_:.3e} -> {'EXACT' if w_ == 0 else 'NOT EXACT'}", flush=True)
            for arm in ref["gens"]:
                rows_ = SAMPLER.sample(n_ref, greedy=True, prompts=qs, constrained=arm != "free", max_tokens=256)
                gens = [r["ids"][r["prompt_len"]:].tolist() for r in rows_]
                same = sum(g == rg for g, rg in zip(gens, ref["gens"][arm]))
                print(f"[online-smoke] {arm}: {same}/{n_ref} generations identical to the merged checkpoint", flush=True)
                for i_, (g, rg) in enumerate(zip(gens, ref["gens"][arm])):      # where does each one diverge?
                    k_ = next((j for j in range(min(len(g), len(rg))) if g[j] != rg[j]), None)
                    if g != rg:
                        print(f"[online-smoke]   {arm} row {i_}: first differing token at {k_} (lens {len(g)} vs {len(rg)}; prompt_len {rows_[i_]['prompt_len']})", flush=True)
                if arm == "free" and ref.get("lps"):        # distribution-level parity: logprob of the reference tokens here vs there
                    lp_ = SAMPLER.score(qs, ref["gens"]["free"], constrained=False)
                    d_ = [abs(a_ - b_) for la, lb in zip(lp_, ref["lps"]["free"]) for a_, b_ in zip(la, lb)]
                    print(f"[online-smoke] free logprob parity on the reference tokens: max |dlogprob| {max(d_):.4f}, "
                          f"mean {sum(d_)/len(d_):.5f} over {len(d_)} tokens", flush=True)
                if arm == "free":                            # HF-argmax agreement of THIS engine's greedy tokens
                    SAMPLER.sleep()
                    rows_g = [{"ids": r["ids"], "prompt_len": r["prompt_len"]} for r in rows_]
                    out_g = teacher_ref(rows_g, adapted=False)
                    am_g = sum(int(t_ == a_) for i_ in range(len(rows_g)) for t_, a_ in zip(gens[i_], out_g[i_][0][:, 0].tolist()))
                    n_g = sum(len(g) for g in gens); lp_g = [v for i_ in range(len(rows_g)) for v in out_g[i_][2].tolist()]
                    print(f"[online-smoke] HF (adapter on, free) on THIS engine's greedy tokens: {am_g}/{n_g} are HF's argmax; "
                          f"mean HF logprob {sum(lp_g)/len(lp_g):.4f}", flush=True)
                    SAMPLER.sync()
                dump_ = A.online_smoke.replace(".json", f"_inprocess_{arm}.json")
                _json.dump({"gens": gens, "prompt_lens": [r["prompt_len"] for r in rows_]}, open(dump_, "w"))
            SAMPLER.sleep()                                 # engine asleep, expert base weights back on the GPU
            # Which vLLM class is faithful to HF? Score each reference's free-arm tokens with the HF model
            # (adapter on, no residency) and compare with the reference engine's own decode logprobs.
            import os as _os
            for rp in [A.online_smoke] + [x for x in _os.environ.get("TMOE_SMOKE_HF_REFS", "").split(":") if x]:
                rj = _json.load(open(rp))
                if not rj.get("lps"):
                    continue
                rows_h = []
                for q_, g_ in zip(qs, rj["gens"]["free"]):
                    enc_ = SAMPLER.tok.apply_chat_template([{"role": "user", "content": q_}], add_generation_prompt=True,
                                                          tokenize=True, return_dict=True)
                    p_ = list(enc_["input_ids"]); rows_h.append({"ids": torch.tensor(p_ + list(g_), dtype=torch.int32), "prompt_len": len(p_)})
                out_h = teacher_ref(rows_h, adapted=False)
                d_ = [abs(a_ - b_) for i_ in range(len(rows_h)) for a_, b_ in zip(out_h[i_][2].tolist(), rj["lps"]["free"][i_])]
                lp_own = [v for i_ in range(len(rows_h)) for v in out_h[i_][2].tolist()]
                am_ = sum(int(t_ == a_) for i_ in range(len(rows_h)) for t_, a_ in zip(rj["gens"]["free"][i_], out_h[i_][0][:, 0].tolist()))
                print(f"[online-smoke] HF (adapter on, free) on {_os.path.basename(rp)}: {am_}/{len(lp_own)} reference tokens are HF's argmax; "
                      f"mean HF logprob of the reference tokens {sum(lp_own)/len(lp_own):.4f}; "
                      f"vs recorded decode logprobs max |d| {max(d_):.4f}, mean {sum(d_)/len(d_):.5f}", flush=True)
            return
    step0 = step
    while seen < A.tokens:
        if SAMPLER is not None and (step - step0) % A.online_every == 0:   # first step always refreshes
            t_on = time.time()
            AUX = SAMPLER.refresh(A.online_n)
            AUXREF = teacher_ref(AUX, adapted=True)
            _rebuild_aux(AUX, A.data_seed + 1 + step)
            print(f"[online] refresh at step {step}: {len(AUX)} fresh on-policy rows in {time.time()-t_on:.0f}s total", flush=True)
        opt.zero_grad(set_to_none=True)
        ntok_a = 0
        for _ in range(accum_batches):
            ridx = chunks[order[oi % len(order)]]
            rs = [rows[i] for i in ridx]
            oi += 1
            ids, am, tgt, plens, ntok = make_batch(rs)
            GL.CFG.update(on=not A.no_constraint, R=A.R, enforce_from=plens,
                          batch=len(rs), cold_start=False)
            if A.kl_only:
                loss = torch.zeros((), device=ids.device)
            elif DEC is not None:
                assert TOKW is None, "tok-weights unsupported on the chunked-head path"
                hid = DEC(ids, attention_mask=am).last_hidden_state[:, :-1]
                loss = batch_ce_hid(hid, tgt, accum_batches)
                loss.backward()      # free the constrained graph BEFORE the KL
                del hid
            else:
                # ReMoE (baseline #2): its objective is defined over ROUTER logits, so the
                # router forward stashes them for the duration of this step only.
                if A.remoe_lambda > 0:
                    GL.CFG["collect"] = []
                logits = model(ids, attention_mask=am).logits[:, :-1]
                loss = batch_ce(logits, tgt, accum_batches, ridx=ridx)
                if A.remoe_lambda > 0:
                    from temporal.ablation_mechanisms import remoe_reuse_loss
                    col = GL.CFG.pop("collect", []) or []
                    if col:
                        k_ = int(getattr(model.config, "num_experts_per_tok", 8) or 8)
                        reuse = sum(remoe_reuse_loss(c, k_, A.remoe_gamma)
                                    for c in col) / len(col)
                        loss = loss + (A.remoe_lambda * reuse / accum_batches)
                loss.backward()
                del logits
                GL.CFG.pop("collect", None)
            if KLREF is not None:    # forward: two live graphs OOM'd at mb2/4096
                if A.kl_arm == "free":
                    GL.CFG.update(on=False, enforce_from=0, batch=1)
                targets = tgt[:, 1:]
                # per-row free forward + immediate backward: KL is a per-row sum,
                # so accumulation is exact while only ONE row's graph + [1,S,V]
                # logits are live (the batched free forward OOM'd at 248k vocab)
                kl_den = sum(int((targets[b] != -100).sum())
                             for b, ri in enumerate(ridx) if ri in KLREF)
                kl_tot = 0.0
                for b, ri in enumerate(ridx):
                    if ri not in KLREF:
                        continue
                    if A.kl_arm == "constrained":
                        GL.CFG.update(on=True, R=A.R, enforce_from=int(plens[b]),
                                      batch=1, cold_start=False)
                    m_ = targets[b] != -100
                    tid, tlp = KLREF[ri][:2]
                    tid = tid.to(ids.device).long()
                    tlp = tlp.to(ids.device).float()
                    def _slice_kl(x_, tid_, tlp_):
                        # x_ is hidden [n,H] on the chunked-head path, else
                        # logits [n,V]; fp32 full-vocab intermediates are
                        # recomputed in backward either way
                        lgb = (HEAD(x_) if DEC is not None else x_).float()
                        s_at = lgb.gather(1, tid_) - torch.logsumexp(lgb, -1, keepdim=True)
                        p = tlp_.exp()
                        p = p / p.sum(1, keepdim=True)  # renormalise top-50
                        return (p * (tlp_ - s_at)).sum()
                    if DEC is not None:
                        row_x = DEC(ids[b:b + 1],
                                    attention_mask=am[b:b + 1]).last_hidden_state[0, :-1][m_]
                    else:
                        row_x = model(ids[b:b + 1],
                                      attention_mask=am[b:b + 1]).logits[0, :-1][m_]
                    kl_sum = 0
                    for j in range(0, row_x.shape[0], 512):
                        kl_sum = kl_sum + torch.utils.checkpoint.checkpoint(
                            _slice_kl, row_x[j:j + 512], tid[j:j + 512],
                            tlp[j:j + 512], use_reentrant=False)
                    kl_b = (A.kl_weight / kl_den / accum_batches) * kl_sum
                    kl_b.backward()
                    kl_tot += float(kl_b)
                    del row_x
                loss = loss.detach() + kl_tot
                GL.CFG.update(on=not A.no_constraint, R=A.R, enforce_from=plens,
                              batch=len(rs), cold_start=False)
            if AUX is not None:   # on-policy term: one aux micro-batch per main micro-batch
                ridx_a = AUXS["chunks"][AUXS["order"][AUXS["i"] % len(AUXS["order"])]]; AUXS["i"] += 1
                rs_a = [AUX[i] for i in ridx_a]
                ids_a, am_a, tgt_a, plens_a, ntok_a = make_batch(rs_a)
                targets_a = tgt_a[:, 1:]
                kl_den_a = sum(int((targets_a[b] != -100).sum())
                               for b, ri in enumerate(ridx_a) if ri in AUXREF)
                for b, ri in enumerate(ridx_a):
                    if ri not in AUXREF or kl_den_a == 0:
                        continue
                    GL.CFG.update(on=True, R=A.R, enforce_from=int(plens_a[b]), batch=1, cold_start=False)
                    m_ = targets_a[b] != -100
                    ref_ = AUXREF[ri]
                    tid = ref_[0].to(ids_a.device).long(); tlp = ref_[1].to(ids_a.device).float()
                    if A.aux_loss == "revkl":
                        if len(ref_) < 3:
                            raise RuntimeError("--aux-loss revkl needs a --precompute-kl file with the "
                                               "teacher log-prob at the sampled token; re-run the precompute")
                        tat = ref_[2].to(ids_a.device).float()
                        tok = targets_a[b][m_].long()
                    def _aux_kl(x_, tid_, tlp_):
                        lgb = (HEAD(x_) if DEC is not None else x_).float()
                        s_at = lgb.gather(1, tid_) - torch.logsumexp(lgb, -1, keepdim=True)
                        p = tlp_.exp(); p = p / p.sum(1, keepdim=True)
                        return (p * (tlp_ - s_at)).sum()
                    def _aux_rev_full(x_, tid_, tlp_):
                        lgb = (HEAD(x_) if DEC is not None else x_).float()
                        T_ = A.aux_kl_temp
                        if T_ == 1.0:
                            ls = lgb - torch.logsumexp(lgb, -1, keepdim=True)        # student log-probs [n,V]
                            s_in = ls.gather(1, tid_)                                  # on the teacher top-50
                            p_in = s_in.exp()
                            m_s = (1.0 - p_in.sum(1)).clamp_min(1e-6)                  # student mass outside
                            m_t = (1.0 - tlp_.exp().sum(1)).clamp_min(1e-6)            # teacher mass outside
                            kl = (p_in * (s_in - tlp_)).sum(1) + m_s * (m_s.log() - m_t.log())
                        else:                          # softened, both renormalised over the top-50 support
                            s_in = torch.log_softmax(lgb.gather(1, tid_) / T_, -1)
                            t_in = torch.log_softmax(tlp_ / T_, -1)
                            kl = (s_in.exp() * (s_in - t_in)).sum(1) * (T_ * T_)
                        return kl.sum(), kl.detach().sum()
                    def _aux_rev(x_, tok_, tat_):
                        lgb = (HEAD(x_) if DEC is not None else x_).float()
                        lps = lgb.gather(1, tok_[:, None]).squeeze(1) - torch.logsumexp(lgb, -1)
                        adv = (tat_ - lps).detach()          # teacher agrees more than the student -> push up
                        return -(adv * lps).sum(), (lps - tat_).detach().sum()   # loss, reverse-KL estimate
                    if DEC is not None:
                        row_x = DEC(ids_a[b:b + 1], attention_mask=am_a[b:b + 1]).last_hidden_state[0, :-1][m_]
                    else:
                        row_x = model(ids_a[b:b + 1], attention_mask=am_a[b:b + 1]).logits[0, :-1][m_]
                    kl_sum = 0
                    if A.aux_loss == "revkl":
                        for j in range(0, row_x.shape[0], 512):
                            l_, e_ = torch.utils.checkpoint.checkpoint(
                                _aux_rev, row_x[j:j + 512], tok[j:j + 512], tat[j:j + 512], use_reentrant=False)
                            kl_sum = kl_sum + l_; aux_est += float(e_) / kl_den_a / accum_batches
                    elif A.aux_loss == "revkl_full":
                        for j in range(0, row_x.shape[0], 512):
                            l_, e_ = torch.utils.checkpoint.checkpoint(
                                _aux_rev_full, row_x[j:j + 512], tid[j:j + 512], tlp[j:j + 512], use_reentrant=False)
                            kl_sum = kl_sum + l_; aux_est += float(e_) / kl_den_a / accum_batches
                    else:
                        for j in range(0, row_x.shape[0], 512):
                            kl_sum = kl_sum + torch.utils.checkpoint.checkpoint(
                                _aux_kl, row_x[j:j + 512], tid[j:j + 512], tlp[j:j + 512], use_reentrant=False)
                    kl_b = (A.aux_kl_weight / kl_den_a / accum_batches) * kl_sum
                    kl_b.backward()
                    aux_tot += float(kl_b)
                    del row_x
                GL.CFG.update(on=not A.no_constraint, R=A.R, enforce_from=plens, batch=len(rs), cold_start=False)
            seen += ntok_a if A.budget_on == "sampled" else ntok      # sampled: the on-policy tokens trained on
        GL.CFG.update(batch=1)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        step += 1; aux_steps += 1
        if step % A.log_every == 0:
            now_ = time.time(); w_ = now_ - globals().get("_T_LAST_LOG", t0); globals()["_T_LAST_LOG"] = now_
            print(f"[gce] step {step} seen {seen/1e6:.2f}M loss {loss.item()*accum_batches:.4f} "
                  f"({seen/(now_-t0):.0f} tok/s) window {w_/A.log_every:.1f} s/step", flush=True)
            if AUX is not None:
                if A.aux_loss in ("revkl", "revkl_full"):
                    print(f"[gce] aux-{A.aux_loss} loss {aux_tot/max(1,aux_steps):.4f}; reverse-KL estimate {aux_est/max(1,aux_steps):.4f} nats/tok "
                          f"(student constrained || teacher free, student-sampled tokens)", flush=True)
                else:
                    print(f"[gce] aux-kl {aux_tot/max(1,aux_steps):.4f} nats/tok (constrained arm, own prefixes)", flush=True)
                aux_tot = 0.0; aux_est = 0.0; aux_steps = 0
        if step % A.save_every == 0:
            save()
    save()
    print(f"[gce] DONE seen={seen/1e6:.2f}M -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
