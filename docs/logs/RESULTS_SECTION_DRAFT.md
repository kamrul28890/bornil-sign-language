# Results Section Draft

Generated: 2026-07-16

This draft is written as report-ready prose. It keeps the official full-data
benchmark separate from filtered diagnostic experiments.

## Experimental Setting

We evaluate gloss-free Bengali sign language translation on the Bornil/BdSL
corpus using MediaPipe keypoint features and an encoder-decoder Transformer
baseline. Each input clip is represented as an upper-body keypoint sequence, and
the model autoregressively generates Bengali subword tokens. We report BLEU-4,
chrF++, ROUGE-L, and BanglaBERT BERTScore F1 where available.

Three frozen benchmark splits are used. `random-v1` randomly partitions clips and
is the easiest setting. `unseen-sentence-v1` prevents normalized sentence text
from appearing in both train and evaluation partitions. `signerindep-v1` keeps
chunk/session groups disjoint, making it the hardest generalization setting.

## Official Full-Data Baseline

The main benchmark uses all target lengths and the original greedy decode policy.

| Split | Train/Test | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| random-v1 | 14390 / 1799 | 0.520 | 9.646 | 0.055 | 0.862 |
| unseen-sentence-v1 | 14390 / 1799 | 0.419 | 9.187 | 0.048 | 0.860 |
| signerindep-v1 | 14500 / 1500 | 0.351 | 8.382 | 0.048 | 0.855 |

The ordering is consistent with the intended difficulty of the splits. The
random split gives the strongest result, while unseen-sentence and
signer-independent evaluation reduce BLEU-4 and ROUGE-L. This indicates that the
model is sensitive to both sentence-level novelty and signer/session shift.

## Comparison With Trivial Floors

We compare the trained model against three non-neural floors: most-frequent
sentence, random training sentence, and nearest-neighbor retrieval using pooled
keypoint features.

| Split | Best trivial BLEU-4 | Model BLEU-4 | Best trivial ROUGE-L | Model ROUGE-L |
|---|---:|---:|---:|---:|
| random-v1 | 0.10 | 0.520 | 0.016 | 0.055 |
| unseen-sentence-v1 | 0.08 | 0.419 | 0.015 | 0.048 |
| signerindep-v1 | 0.07 | 0.351 | 0.014 | 0.048 |

The trained model outperforms all trivial floors on BLEU-4 and ROUGE-L across
the three official splits. This suggests that it learns more than a pure Bengali
language prior or a nearest-neighbor lookup over clip-level features.

However, chrF++ is less reliable in this setting. Retrieval and most-frequent
baselines can score competitively on chrF++ despite being poor translation
systems, because Bengali character overlap can be high even when sentence
meaning is wrong. We therefore interpret chrF++ alongside BLEU-4, ROUGE-L,
BERTScore, and qualitative examples rather than using it alone.

## Model Ablations

We tested model-side changes first on official full-data `random-v1`.

| Model | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| Vanilla SignTransformer | 0.520 | 9.646 | 0.055 | 0.862 | official baseline |
| TConv2k5 | 0.655 | 9.482 | 0.054 | 0.864 | improves BLEU-4, not ROUGE-L |
| TConv2k5 + EOS penalty 0.2 | 0.113 | 7.262 | 0.031 | 0.857 | failed; output collapse |

Adding a small temporal convolutional frontend improves BLEU-4 from 0.520 to
0.655 on full-data `random-v1`, suggesting that local temporal motion modeling
helps the keypoint Transformer. The gain is not universal across metrics:
chrF++ and ROUGE-L do not improve, and length-bin diagnostics still show weak
behavior on longer references.

The early EOS penalty ablation failed. It sharply reduced BLEU-4 and output
diversity, producing repeated generic hypotheses. We therefore do not use this
setting for further evaluation.

## Target-Length Filtering Diagnostic

The target-length audit found that 84.6% of rows have 12 or fewer target tokens,
while the longest rows are article-like, multi-clause text that is often poorly
matched to a short signing clip. To isolate this issue, we created a filtered
LTE12 diagnostic track. The frozen split is loaded first, then rows with target
length greater than 12 tokens are removed inside each partition. This preserves
split provenance while evaluating the short/medium target subset separately.

| Split | Train/Test | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| random-v1 | 12162 / 1546 | 0.716 | 9.953 | 0.055 | 0.869 |
| unseen-sentence-v1 | 12170 / 1521 | 0.602 | 9.325 | 0.050 | 0.871 |
| signerindep-v1 | 12507 / 1108 | 0.813 | 9.219 | 0.055 | 0.870 |

The filtered track gives stronger BLEU-4 than the official full-data track,
especially on `random-v1`. This supports the diagnosis that very long
article-like targets are degrading the benchmark. The result should not replace
the official full-data table, because it answers a different question: how well
the model performs on short and medium targets after excluding long noisy rows.

The harder filtered splits remain mixed. BLEU-4 stays above trivial floors, but
chrF++ is close to or below simple frequency-based baselines. The
signer-independent filtered result is promising, but zero exact matches and
elevated repetition mean it should not be presented as definitive signer
generalization.

## Decode Stability

We added an opt-in repetition guard that stops decoding when the same token is
generated eight times consecutively. This guard is disabled by default and is not
part of the official protocol. On the LTE12 checkpoints, it clips rare runaway
token loops while leaving corpus metrics essentially unchanged. We therefore use
it only as a diagnostic and qualitative-inspection tool.

## Summary of Findings

The experiments establish a reproducible baseline for Bornil/BdSL gloss-free
sign language translation. The model beats trivial floors on the official
splits, but performance remains low in absolute terms, especially under
unseen-sentence and signer/session-shift evaluation. A temporal convolutional
frontend is a useful direction, but it does not solve the long-target problem.
The LTE12 diagnostic track shows that target quality and target length are major
sources of difficulty, motivating a cleaned short/medium benchmark track and
more data-centric filtering before more expensive model scaling.
