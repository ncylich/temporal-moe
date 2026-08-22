# Paper TODO

## Prose pass

- [ ] Remove residual jargon throughout: arm names like "R8"/"R16"/"R32" should read as
      $R=8$ / 8 resident experts / the matched fraction, "R=k arm" phrasing checked per
      use, and any other run-record shorthand (screen/authoritative instrument language)
      rewritten for a reader who never saw the repo. Figures included (axis and legend
      labels still say R8/R16 in places).

## Length section

- [ ] Section 7 (Generation length under the constraint) is a trim candidate if the page
      budget binds: the decomposition figure + one paragraph could fold into Section 6,
      with the flip strip and the null moving to the appendix.
- [ ] FAILURE TO FIX: trajectories were NOT saved for a large slice of the instruct grid,
      despite an explicit instruction that all trajectories be saved when the generations
      were originally run (by a Fable 5 agent). Missing entirely: every family-specific
      HumanEval cell (gemma channel-aware, both gpt-oss variants, LFM), qwen's think-on
      HumanEval, and every think-on MMLU cell. Damaged: the earliest MMLU dual dumps kept
      only 4 of 228 items (per-subject overwrite bug, since fixed). Consequence: the
      Section 7 flip and blow-up analysis cannot cover code or knowledge, the two damage
      loci where thinking-mode damage runs -9 to -15 points. Fix by regenerating those
      cells WITH per-item dumps (raw pre-strip text, doc-keyed). Cheap partial recovery
      available first: think-off MMLU lengths for the gemma and qwen families are
      re-tokenizable from the existing 228-item dual dumps at zero GPU cost (no doc ids,
      so verify pairing by the fixed item ordering).

## Instruct grid

- [ ] OLMoE MMLU is still the stock strict-filter score (no dual relaxed re-score exists for
      it, unlike qwen/gemma/gpt-oss). The paper currently presents all MMLU cells as
      extraction-scored without flagging the exception. Re-score OLMoE's saved generations
      with the relaxed extractor (or regenerate with dumps) so the statement is true.

## Serving benchmarks

- [ ] Update Pixel measurements on a fairer baseline. The current curve runs a random-weight
      router that churns 2.09 experts/layer, about twice the ~1-swap/layer design point of a
      trained temporal router, and its all-resident ceiling is inferred from byte accounting
      because the full model cannot run on the device. Re-measure with a trained
      temporally-coherent router (or prescribed design-point turnover) and a defensible
      ceiling.
- [ ] Get Mohsen's Intel iGPU PC numbers for running temporal MoE.
- [ ] Abstract AND conclusion pair the Pixel 0.70x with the 5.1x memory cut (inherited from
      the repo README). Measured: 0.70x is at a 3x cut; a ~5x cut measures 0.60-0.63x.
      Resolve by re-measurement above, then update the serving numbers in the abstract, the
      conclusion's serving sentence, and Section 5's Pixel paragraph together. The repo
      README's serving sentence should be corrected in the same pass.
