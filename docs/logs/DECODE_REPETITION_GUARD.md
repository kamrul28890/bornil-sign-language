# Decode Repetition Guard Ablation

Generated: 2026-07-16

Change tested: greedy checkpoint decoding with `repetition_stop=8`, meaning a
hypothesis is stopped if the same token is produced eight times consecutively.

This is an eval-only guard. It does not retrain the model and it is disabled by
default for existing official results.

## LTE12 TConv Checkpoints

| Split | Run | Decode | BLEU-4 | chrF++ | ROUGE-L | Avg hyp tokens | Unique hyps |
|---|---|---|---:|---:|---:|---:|---:|
| random-v1 | `bc38907bece6` | original greedy | 0.716 | 9.953 | 0.055 | 5.51 | 1463 |
| random-v1 | `bc38907bece6` | greedy + repstop8 | 0.717 | 9.961 | 0.055 | 5.47 | 1463 |
| unseen-sentence-v1 | `58269a7b1003` | original greedy | 0.602 | 9.325 | 0.050 | 5.48 | 1356 |
| unseen-sentence-v1 | `58269a7b1003` | greedy + repstop8 | 0.602 | 9.319 | 0.050 | 5.47 | 1356 |
| signerindep-v1 | `2ff734ae1aae` | original greedy | 0.813 | 9.219 | 0.055 | 5.19 | 991 |
| signerindep-v1 | `2ff734ae1aae` | greedy + repstop8 | 0.814 | 9.228 | 0.055 | 5.14 | 990 |

## Interpretation

The repetition guard is safe as a diagnostic decode option. It clips rare runaway
token loops while leaving corpus-level metrics essentially unchanged. That means
the LTE12 score improvements are not caused by long repeated hypotheses, and the
guard can be used when producing qualitative examples or sanity-checking
checkpoint outputs.

Do not replace the official greedy numbers with repstop8 numbers yet. Keep
official reporting on the original decode policy, and mention repstop8 only as a
stability/diagnostic decode option unless a future protocol explicitly adopts it.
