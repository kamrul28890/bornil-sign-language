# Results Overview

Generated: 2026-07-16

This file is the report-facing map of the Bornil experiment results. Keep the
official full-data benchmark separate from filtered diagnostic experiments.

## Reporting Structure

1. Official benchmark: vanilla SignTransformer on all target lengths and all
   three frozen splits.
2. Full-data ablations: model changes tested on official `random-v1` first.
3. Filtered LTE12 diagnostic track: TConv on targets with `<= 12` tokens,
   evaluated separately to study data quality and target-length effects.
4. Decode diagnostics: eval-only checks that should not replace official
   numbers unless the protocol is explicitly changed.

CI and paired-significance tables for the saved random-v1 baseline/ablation
runs are in `docs/logs/SIGNIFICANCE.md` (vanilla vs TConv),
`docs/logs/SIGNIFICANCE_CTC.md` (CTC vs TConv),
`docs/logs/SIGNIFICANCE_CTC_VS_VANILLA.md` (CTC vs vanilla),
`docs/logs/SIGNIFICANCE_CTC_HARDSPLITS.md` (CTC vs hard-split vanilla), and
`docs/logs/SIGNIFICANCE_BILSTM.md` (BiLSTM vs Transformer). All p-values are
two-sided paired bootstrap.

## Table 1: Official Full-Data Baselines

Use this as the main benchmark table.

| Split | Run | Train/Test | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| random-v1 | `c19d0ea08824` | 14390 / 1799 | 0.520 | 9.646 | 0.055 | 0.862 | easiest split |
| unseen-sentence-v1 | `60320f4c6a60` | 14390 / 1799 | 0.419 | 9.187 | 0.048 | 0.860 | unseen text groups |
| signerindep-v1 | `46a706570712` | 14500 / 1500 | 0.351 | 8.382 | 0.048 | 0.855 | hardest full-data split |

Source: `docs/logs/BASELINE_TABLE.md`.

## Table 2: Full-Data Random-v1 Ablations

Use this to justify which model directions helped and which failed.

All comparisons are vs the vanilla baseline unless noted; significance is
two-sided paired bootstrap on random-v1. BLEU-4 is at the noise floor here
(all values < 1.0 on the 0–100 scale), so **chrF++ is the discriminating
metric** — read the interpretation column off chrF++, not BLEU-4.

| Model | Run | Test rows | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Vanilla SignTransformer | `c19d0ea08824` | 1799 | 0.520 | 9.646 | 0.055 | 0.862 | official baseline |
| TConv2k5 | `c26953197929` | 1799 | 0.655 | 9.482 | 0.054 | 0.864 | **no supported win** — BLEU-4 gain n.s. (p=0.486, n=1000); chrF++ n.s. (p=0.344) |
| TConv2k5 + joint CTC | `b0a590ea3317` | 1799 | 0.684 | 10.913 | 0.067 | — | **best so far** — chrF++ significantly > vanilla and > TConv (both p=0.000); BLEU-4 n.s. |
| BiLSTM d256 (T0.1) | `07709687f916` | 1799 | 0.621 | 8.990 | 0.064 | — | chrF++ significantly **below** vanilla (p=0.000); BLEU-4 n.s. → Transformer preferred |
| TConv2k5 + EOS penalty 0.2 | `a459d1148d0c` | 1799 | 0.113 | 7.262 | 0.031 | 0.857 | failed; significant regression, output diversity collapsed |

Decision: **TConv alone is not a supported model-side win** (BLEU-4 gain not
significant, chrF++ unchanged) — do not claim it. **Joint CTC/attention is the
current best model-side change**: a significant chrF++ improvement over both
vanilla and TConv, and it additionally enables the WS-E alignment workstream
(CTC forced alignment). BiLSTM (T0.1) underperforms the Transformer on chrF++.
Abandon `early_eos_penalty=0.2`. The joint-CTC chrF++ gain generalizes to both
hard official splits: signerindep-v1 improves from 8.382 to 10.361 (p=0.000)
and unseen-sentence-v1 improves from 9.187 to 10.943 (p=0.000), using n=1000
paired bootstrap. The firmed-up random-v1 CTC comparisons at n=1000 keep
chrF++ significant vs vanilla and vs TConv (both p=0.000), while BLEU-4 remains
not significant on random-v1 (p=0.462 vs vanilla; p=0.894 vs TConv).

## Table 2b: Joint CTC Hard-Split Generalization

Use this to decide whether the random-v1 joint-CTC chrF++ gain transfers to the
harder official settings. All comparisons are vs the vanilla baseline for the
same split, with two-sided paired bootstrap (n=1000).

| Split | Model | Run | Test rows | BLEU-4 | chrF++ | ROUGE-L | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| signerindep-v1 | Vanilla SignTransformer | `46a706570712` | 1500 | 0.351 | 8.382 | 0.048 | official baseline |
| signerindep-v1 | TConv2k5 + joint CTC | `40cffb9e62bb` | 1500 | 0.792 | 10.361 | 0.072 | chrF++ gain +1.979, p=0.000; BLEU-4 p=0.000 |
| unseen-sentence-v1 | Vanilla SignTransformer | `60320f4c6a60` | 1799 | 0.419 | 9.187 | 0.048 | official baseline |
| unseen-sentence-v1 | TConv2k5 + joint CTC | `a5a9060499a4` | 1799 | 0.889 | 10.943 | 0.070 | chrF++ gain +1.756, p=0.000; BLEU-4 p=0.014 |

Source: `docs/logs/SIGNIFICANCE_CTC_HARDSPLITS.md`.

## Table 3: Filtered LTE12 Diagnostic Track

Use this as a separate diagnostic table, not as the official benchmark table.
The filter is applied after loading the frozen split: keep rows where
`clean_text` has `<= 12` whitespace tokens.

| Split | Run | Train/Test | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| random-v1 | `bc38907bece6` | 12162 / 1546 | 0.716 | 9.953 | 0.055 | 0.869 | cleanest LTE12 win |
| unseen-sentence-v1 | `58269a7b1003` | 12170 / 1521 | 0.602 | 9.325 | 0.050 | 0.871 | useful but chrF++ close to most-freq |
| signerindep-v1 | `2ff734ae1aae` | 12507 / 1108 | 0.813 | 9.219 | 0.055 | 0.870 | promising, but not final signer-generalization proof |

Source: `docs/logs/FILTERED_LTE12_RESULTS.md`.

## Table 4: Trivial Floors

Use this to show the models learn more than a language prior or a
nearest-neighbour lookup. **Retrieval is the strongest trivial baseline on every
split**, so it is the only floor that matters; chrF++ is the discriminating
metric (BLEU-4 is at the noise floor for all systems).

| Split | Metric | Retrieval floor | Vanilla | TConv+CTC | CTC − floor |
|---|---|---:|---:|---:|---:|
| random-v1 | BLEU-4 | 0.10 | 0.520 | 0.684 | +0.58 |
| random-v1 | chrF++ | 9.20 | 9.646 | 10.913 | **+1.71** |
| random-v1 | ROUGE-L | 0.016 | 0.055 | 0.067 | +0.051 |
| unseen-sentence-v1 | BLEU-4 | 0.08 | 0.419 | 0.889 | +0.81 |
| unseen-sentence-v1 | chrF++ | 9.24 | 9.187 | 10.943 | **+1.70** |
| unseen-sentence-v1 | ROUGE-L | 0.015 | 0.048 | 0.070 | +0.055 |
| signerindep-v1 | BLEU-4 | 0.07 | 0.351 | 0.792 | +0.72 |
| signerindep-v1 | chrF++ | 9.14 | 8.382 | 10.361 | **+1.22** |
| signerindep-v1 | ROUGE-L | 0.014 | 0.048 | 0.072 | +0.058 |

**Key result.** The vanilla baseline **loses to retrieval on chrF++ on both hard
splits** (−0.05 on unseen-sentence-v1, −0.76 on signerindep-v1) — the weakest
point in the earlier results narrative, since a nearest-neighbour lookup matched
our model on the character-level metric. **TConv+CTC clears the retrieval floor
on all three splits on every metric.** This removes that objection.

> **Caveat (2026-07-19, from the input ablation).** Do not lead with this table.
> Feeding the trained model pure Gaussian noise scores **9.08 chrF++** — within
> 0.12 of the random-v1 retrieval floor (9.20). The retrieval floor therefore sits
> at roughly the *no-information* level, so clearing it by +1.71 is a much lower
> bar than it reads. Use Table 4b instead, where the input distribution is held
> exactly constant. See `docs/logs/INPUT_ABLATION.md`.

Source: `docs/logs/TRIVIAL_FLOORS.md`; CTC rows from runs `b0a590ea3317`,
`a5a9060499a4`, `40cffb9e62bb`.

## Table 4b: Input Ablation (the grounding control)

Same checkpoint (`b0a590ea3317`, random-v1, n=1799), decoder untouched, keypoint
stream corrupted. `mismatch` deranges only which clip sits opposite which
sentence, holding real clips, real feature statistics and real lengths constant —
so it differs from the control in exactly one respect.

| Condition | BLEU-4 | chrF++ | ROUGE-L | unique hyps |
|---|---:|---:|---:|---:|
| control (`none`) | **0.684** | **10.91** | **0.067** | 1777 |
| frame order shuffled | 0.074 | 10.04 | 0.018 | 1763 |
| Gaussian noise input | 0.037 | 9.08 | 0.007 | 1403 |
| **video↔text mismatched** | 0.050 | **6.20** | 0.007 | 1776 |
| zero input | 0.032 | 4.70 | 0.005 | 112 |

**Two results.** (1) *Grounding is real*: `mismatch` costs −4.71 chrF++ and 13.7×
BLEU-4. This is the strongest evidence in the project that the model translates
rather than generates from a prior, and it is a stricter control than any trivial
floor. (2) *Grounding is static pose, not motion*: shuffling frame order costs
only −0.88 chrF++ (92% retained), while BLEU-4 retains 11%. The encoder largely
behaves as a bag-of-poses classifier despite the TConv frontend and CTC
objective. This is the measured version of "weak visual grounding" and is the
target WS-E should be evaluated against.

Source: `docs/logs/INPUT_ABLATION.md`, `docs/logs/SIGNIFICANCE_INPUT_ABLATION.md`.

## Table 5: Output Diversity & Length Calibration (why joint CTC helps)

Diagnostic that rules out the obvious alternative explanation for the CTC chrF++
gain — namely that chrF++ merely rewards generic, high-frequency Bengali.

| Model | Split | Test rows | Unique hyps | Unique % | Mean hyp len | Mean ref len | Top-1 share |
|---|---|---:|---:|---:|---:|---:|---:|
| Vanilla | random-v1 | 1799 | 1616 | 89.8% | 6.50 | 7.50 | 0.9% |
| TConv+CTC | random-v1 | 1799 | 1777 | **98.8%** | **7.56** | 7.50 | 0.2% |
| Vanilla | unseen-sentence-v1 | 1799 | 1692 | 94.1% | 6.38 | 7.62 | 0.4% |
| TConv+CTC | unseen-sentence-v1 | 1799 | 1770 | **98.4%** | 7.25 | 7.62 | 0.2% |
| Vanilla | signerindep-v1 | 1500 | 1365 | 91.0% | 6.85 | 9.40 | 0.9% |
| TConv+CTC | signerindep-v1 | 1500 | 1485 | **99.0%** | 7.43 | 9.40 | 0.4% |

Interpretation:

- **The "generic output" explanation is ruled out.** CTC *raises* unique-output
  rate from ~90–94% to **98–99%** and halves top-1 share. The chrF++ gain comes
  with more diverse outputs, not fewer.
- **CTC fixes the long-standing under-generation.** The vanilla model runs ~1
  token short of references; CTC lands at 7.56 vs 7.50 on random-v1 (near-exact
  length calibration) and closes most of the gap on unseen-sentence-v1.
- **Mechanism worth stating in the paper:** per-frame CTC supervision grounds the
  encoder temporally, which fixes premature EOS *without* collapsing diversity.
  Contrast with the `early_eos_penalty=0.2` ablation, which targeted the same
  under-generation symptom directly and collapsed output diversity
  (`docs/logs/TCONV_LENGTH_PRESSURE_RESULT.md`). Two attempts, one principled fix.
- **Honest caveat:** signerindep-v1 still under-generates (7.43 vs 9.40 reference
  length). Signer/session generalization remains genuinely unsolved and motivates
  the next model-side work.

## Decode Decision

Official numbers should keep the original greedy decode policy. The opt-in
`repetition_stop=8` guard is useful for qualitative inspection because it clips
rare repeated-token runaways with almost no corpus-metric change, but it is not
the official protocol yet.

Source: `docs/logs/DECODE_REPETITION_GUARD.md`.

## Main Claims To Write

- The official benchmark is difficult, but **the generalization gap is
  model-dependent** — state it per model, not as a property of the benchmark
  alone. The vanilla baseline drops monotonically across splits (chrF++ 9.646 →
  9.187 → 8.382), whereas TConv+CTC shows **no unseen-sentence penalty** (10.943
  vs 10.913 on random-v1 — statistically indistinguishable) and only a modest
  signer-independent drop (10.361). Open-vocabulary generalization is largely
  solved by a better-grounded encoder; **signer/session generalization is not**.
- **The model performs grounded translation — lead with the input ablation, not
  the floors.** Deranging only the video↔text correspondence costs −4.71 chrF++
  (10.91 → 6.20) and 13.7× BLEU-4 with the input distribution held exactly
  constant (Table 4b). This is the strongest and cleanest grounding evidence
  available. The floor comparison is supporting material: TConv+CTC does clear
  the retrieval floor on all three splits on every metric (Table 4), but that
  floor sits near the noise-input level (9.20 vs 9.08), so a +1.71 margin is weak
  on its own.
- **The grounding is static pose, not motion — state this as an open weakness.**
  Shuffling frame order within clips retains 92% of chrF++ (10.91 → 10.04) while
  BLEU-4 retains 11%. The current best model is close to time-invariant on the
  character metric despite a temporal-conv frontend and a CTC objective. This is
  the honest characterization of the remaining gap and the target for WS-E.
- Character-level metrics are fragile here: chrF++ can reward generic overlap and
  make trivial baselines look deceptively strong. We control for this explicitly
  — the CTC chrF++ gain comes with *higher* output diversity (98–99% unique) and
  better length calibration, not generic text (Table 5).
- BLEU-4 is at the noise floor for this open-vocabulary benchmark (every model
  scores < 1.0 on the 0–100 scale); report it, but base model-comparison claims
  on chrF++ / BERTScore, which actually discriminate.
- Among model-side changes on random-v1, **only joint CTC/attention yields a
  statistically significant chrF++ improvement**; TConv-alone and the BiLSTM
  T0.1 baseline do not (paired bootstrap). The apparent TConv BLEU-4 gain is not
  significant and must not be claimed.
- Joint CTC/attention also significantly improves chrF++ on the two harder
  official splits: signerindep-v1 (p=0.000) and unseen-sentence-v1 (p=0.000).
  This is the current strongest model-side result, though BLEU-4 remains a
  secondary/noise-floor metric for headline claims.
- The LTE12 track shows that long article-like targets are a major data-quality
  problem. Report it separately as evidence for a cleaned short/medium target
  benchmark, not as a replacement for official full-data results.
