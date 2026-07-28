# Overlap-friendly MoE architecture variants (orch 0150)

Two param-neutral, iso-FLOP variants that move the MoE routing decision earlier so expert
streaming can overlap the attention-compute window (the architecture-side answer to the O-series
drift finding). Implemented as edits to the vendored Megatron-LM submodule, captured in
`overlap_variants_megatron.patch` (apply from inside `Megatron-LM/` with `git apply`).

Flags (default OFF == baseline exactly; verified to within bf16 run-to-run non-determinism):
- `--overlap-early-router` (config `overlap_early_router`): MoE router logits from the PRE-attention
  normalized hidden state (input_layernorm output = LN1(x)); experts still process the post-attention
  pre_mlp_layernorm output.
- `--overlap-parallel-ffn` (config `overlap_parallel_ffn`): PaLM/GPT-J parallel block
  y = x + Attn(LN1(x)) + MoE(LN2(x)); the MoE branch (and its router) reads the pre-attention input x.

Applies to MoE layers only; dense layer-0 is untouched. Files: transformer_config.py (2 fields),
arguments.py (2 store_true args), moe_layer.py (optional router_hidden_states), transformer_layer.py
(V1/V2 hooks). Inject via the flame38m_run.sh `EXTRA_MODEL_ARGS` hook.

Parity (g3-temporal, 20it, seed1234): |edited_OFF - original_OFF| = 1.2e-4 < run-to-run floor
|edited_OFF - edited_OFF_2| = 9.7e-4; V1/V2 differ from OFF by ~1e-2 (flags active). See relay 0154.
