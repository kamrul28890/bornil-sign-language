# Filtered LTE12 TConv Results

Generated: 2026-07-16

Config: `configs/full_tconv_lte12.yaml`

Model: `SignTransformer + TConv2k5`

Filter: keep rows with `clean_text` length `<= 12` whitespace tokens after loading
the frozen split. This preserves official split membership while removing long
article-like targets from train/val/test.

## Model Results

| Split | Run | Test rows | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Exact | Unique hyps | Repeated rows | Avg ref/hyp tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random-v1 | `bc38907bece6` | 1546 | 0.716 | 9.953 | 0.055 | 0.869 | 2 | 1463 | 10.0% | 5.72 / 5.51 |
| unseen-sentence-v1 | `58269a7b1003` | 1521 | 0.602 | 9.325 | 0.050 | 0.871 | 0 | 1356 | 18.4% | 5.70 / 5.48 |
| signerindep-v1 | `2ff734ae1aae` | 1108 | 0.813 | 9.219 | 0.055 | 0.870 | 0 | 991 | 16.5% | 5.75 / 5.19 |

## Matching Trivial Floors

| Split | Baseline | BLEU-4 | chrF++ | ROUGE-L |
|---|---|---:|---:|---:|
| random-v1 | most-freq | 0.02 | 5.06 | 0.002 |
| random-v1 | random | 0.09 | 6.61 | 0.006 |
| random-v1 | retrieval | 0.15 | 8.04 | 0.015 |
| unseen-sentence-v1 | most-freq | 0.07 | 9.29 | 0.013 |
| unseen-sentence-v1 | random | 0.07 | 6.53 | 0.007 |
| unseen-sentence-v1 | retrieval | 0.13 | 8.20 | 0.015 |
| signerindep-v1 | most-freq | 0.08 | 9.36 | 0.013 |
| signerindep-v1 | random | 0.13 | 6.52 | 0.006 |
| signerindep-v1 | retrieval | 0.09 | 8.10 | 0.013 |

## Interpretation

The LTE12 filtered track is useful and should be kept as a separate diagnostic
benchmark, not mixed into the official full-data baseline table. It answers a
different question: "How well does the model work on short/medium targets when
long article-like targets are excluded?"

The model beats the matching trivial floors on BLEU-4 and ROUGE-L across all
three filtered splits. That means it is doing more than copying the most common
sentence, random train text, or nearest-neighbor feature retrieval.

The random-v1 filtered result is the cleanest win: BLEU-4 rises from the full
random-v1 TConv score of 0.655 to 0.716, chrF++ improves from 9.482 to 9.953,
and repetition drops. This supports the length-audit diagnosis that the longest
targets were hurting the benchmark.

The harder filtered splits are mixed. BLEU-4 remains above trivial floors, but
the model's chrF++ is only barely above most-frequent on unseen-sentence-v1 and
below most-frequent on signerindep-v1. This reinforces the earlier warning that
chrF++ can reward generic character overlap and should not be read alone.

The filtered signer-independent BLEU-4 is high, but exact matches are zero and
repetition is still elevated. Treat it as promising but not a final claim of
signer generalization.

## Decision

Keep two result tracks:

1. Official full-data track: vanilla/TConv over all target lengths.
2. Filtered LTE12 diagnostic track: TConv over short/medium targets only.

Do not replace official full-data results with LTE12 results in the main table.
Use LTE12 to motivate data-quality filtering and a future cleaned benchmark.
