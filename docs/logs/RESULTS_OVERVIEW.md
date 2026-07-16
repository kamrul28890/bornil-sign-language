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

Use the full-data trivial floors to show the model is not only copying frequent
sentences or doing nearest-neighbor retrieval.

| Split | Best trivial BLEU-4 | Model BLEU-4 | Best trivial ROUGE-L | Model ROUGE-L | Main caution |
|---|---:|---:|---:|---:|---|
| random-v1 | 0.10 | 0.520 | 0.016 | 0.055 | model clearly above floors |
| unseen-sentence-v1 | 0.08 | 0.419 | 0.015 | 0.048 | retrieval chrF++ is competitive |
| signerindep-v1 | 0.07 | 0.351 | 0.014 | 0.048 | retrieval chrF++ exceeds model chrF++ |

Source: `docs/logs/TRIVIAL_FLOORS.md`.

## Decode Decision

Official numbers should keep the original greedy decode policy. The opt-in
`repetition_stop=8` guard is useful for qualitative inspection because it clips
rare repeated-token runaways with almost no corpus-metric change, but it is not
the official protocol yet.

Source: `docs/logs/DECODE_REPETITION_GUARD.md`.

## Main Claims To Write

- The official benchmark is difficult: scores drop from random-v1 to unseen
  sentence and signer-independent settings.
- The trained model beats trivial floors on BLEU-4 and ROUGE-L, so it learns
  more than a pure language prior or nearest-neighbor retrieval.
- Character-level metrics are fragile here: chrF++ sometimes rewards generic
  overlap and can make trivial baselines look deceptively strong.
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
