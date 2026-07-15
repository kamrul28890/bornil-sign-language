# Trivial Baseline Floors

Generated: 2026-07-15

These floors use the same frozen split files and full `data_full` feature pack as
the SignTransformer baseline. They are diagnostic, not competitive models.

## Scores

| Split | Method | BLEU-4 | chrF | chrF++ | ROUGE-L |
|---|---|---:|---:|---:|---:|
| random-v1 | most-freq | 0.02 | 5.70 | 4.31 | 0.002 |
| random-v1 | random | 0.05 | 8.79 | 6.75 | 0.007 |
| random-v1 | retrieval | 0.10 | 11.89 | 9.20 | 0.016 |
| unseen-sentence-v1 | most-freq | 0.06 | 11.43 | 8.74 | 0.012 |
| unseen-sentence-v1 | random | 0.05 | 8.86 | 6.81 | 0.008 |
| unseen-sentence-v1 | retrieval | 0.08 | 11.91 | 9.24 | 0.015 |
| signerindep-v1 | most-freq | 0.06 | 10.75 | 8.21 | 0.012 |
| signerindep-v1 | random | 0.04 | 7.78 | 5.95 | 0.005 |
| signerindep-v1 | retrieval | 0.07 | 11.84 | 9.14 | 0.014 |

## Model Versus Retrieval

| Split | Model BLEU-4 | Retrieval BLEU-4 | Delta | Model chrF++ | Retrieval chrF++ | Delta | Model ROUGE-L | Retrieval ROUGE-L | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random-v1 | 0.520 | 0.10 | +0.420 | 9.646 | 9.20 | +0.446 | 0.055 | 0.016 | +0.039 |
| unseen-sentence-v1 | 0.419 | 0.08 | +0.339 | 9.187 | 9.24 | -0.053 | 0.048 | 0.015 | +0.033 |
| signerindep-v1 | 0.351 | 0.07 | +0.281 | 8.382 | 9.14 | -0.758 | 0.048 | 0.014 | +0.034 |

## Interpretation

- The trained model beats all trivial floors on BLEU-4 and ROUGE-L for all three
  official splits, so it is learning more than a pure language prior or nearest
  neighbor lookup.
- Retrieval remains competitive on chrF++ and beats the model on chrF++ for the
  two harder splits. This is a known risk with character-level metrics: a
  retrieved frequent-ish Bengali sentence can share character n-grams without
  being semantically correct.
- Read BLEU-4 and ROUGE-L as the more discriminating signal for this baseline
  comparison, and keep qualitative prediction inspection in the paper notes.
- The model's ordering is sensible: `random-v1` is easiest, `unseen-sentence-v1`
  is harder, and `signerindep-v1` is hardest.
