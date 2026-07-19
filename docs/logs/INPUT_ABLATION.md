# Input Ablation — does the encoder actually use the video?

Diagnostic for the standing question in the baseline narrative ("visual grounding
is weak"). It re-decodes the frozen test split with the keypoint stream corrupted
in ways that destroy visual information while leaving the trained decoder
untouched. If scores survive corruption, the encoder is decorative and the
decoder's Bengali prior is producing the output.

Run: `data_full/runs/b0a590ea3317` (SignTransformer + TConv2k5 d256 L4+4, joint
CTC, `ctc_weight=0.3`), `random-v1`, greedy decode, seed 42, n=1799.

```
python scripts/input_ablation.py --run-dir data_full/runs/b0a590ea3317 \
    --mode none mismatch frame_shuffle zero noise
```

## Conditions

| Mode | What it destroys | What it preserves |
|---|---|---|
| `none` | — (control) | everything |
| `mismatch` | the video↔text correspondence (deranged pairing) | real clips, real feature statistics, real lengths |
| `frame_shuffle` | temporal order within each clip | the clip's pose distribution |
| `noise` | all visual content (Gaussian matched to per-dim mean/std) | feature scale and clip lengths |
| `zero` | all input (all-zero features) | clip lengths only |

`mismatch` is the primary test: it is the only condition that holds the input
distribution *exactly* constant and changes nothing but which clip sits opposite
which sentence.

## Results

| Mode | BLEU-4 | chrF++ | ROUGE-L | unique hyps |
|---|---:|---:|---:|---:|
| `none` (control) | **0.684** | **10.91** | **0.067** | 1777 |
| `frame_shuffle` | 0.074 | 10.04 | 0.018 | 1763 |
| `noise` | 0.037 | 9.08 | 0.007 | 1403 |
| `mismatch` | 0.050 | 6.20 | 0.007 | 1776 |
| `zero` | 0.032 | 4.70 | 0.005 | 112 |

The `none` row reproduces the run's recorded test metrics exactly (BLEU-4 0.684,
chrF++ 10.913), which validates the harness.

## Finding 1 — the encoder is genuinely used

Breaking only the video↔text correspondence (`mismatch`) costs **−4.71 chrF++**
(10.91 → 6.20) and drops BLEU-4 by **13.7×**. The model is reading the video.

This is a stricter and more informative control than the trivial floors in
`TRIVIAL_FLOORS.md`, because it holds the feature distribution exactly constant
— a retrieval or most-frequent baseline differs from the model in many ways at
once, whereas `mismatch` differs in exactly one. **The +4.71 chrF++ gap over
`mismatch` is the strongest evidence in this project that the model performs
grounded translation rather than prior-driven generation**, and it should be
reported alongside (or instead of) the +1.71 margin over the retrieval floor.

## Finding 2 — the grounding is static pose, not motion

Scrambling frame order costs only **−0.88 chrF++** (10.91 → 10.04): the model
retains **92%** of its chrF++ with time destroyed. BLEU-4 retains **11%**.

The encoder is therefore behaving largely as a *bag-of-poses* classifier. It
identifies which poses occur in a clip but makes limited use of the order in
which they occur — despite a temporal-convolution frontend and a CTC objective
that both exist to model sequence structure. Whatever temporal modelling is
happening shows up in BLEU-4 (word order) and almost not at all in chrF++.

The −0.88 chrF++ drop is small but **statistically significant** (paired
bootstrap n=1000, p<0.001; `docs/logs/SIGNIFICANCE_INPUT_ABLATION.md`), and so is
the 89% BLEU-4 drop (0.684 → 0.074, p<0.001). So temporal order is *used* — the
claim is that it is used *weakly on the character metric*, not that it is unused.
`mismatch` is also significant on both metrics (p<0.001), as expected.

This is the concrete, measured statement of "weak visual grounding" that the
baseline diagnosis asserted qualitatively, and it is what WS-E (CTC forced
alignment) should be evaluated against: **the target is closing the
`frame_shuffle` gap, not raising the headline score.**

## Finding 3 — chrF++ has a high uninformative floor

Pure Gaussian noise as input scores **9.08 chrF++** — within 0.12 of the
retrieval floor (9.20) and above the vanilla baseline on signerindep-v1 (8.382).
A model receiving *no information whatsoever* reaches ~83% of the best model's
chrF++.

chrF++ on this benchmark therefore largely measures "emits fluent Bengali of
roughly the right length and character distribution." Margins of 1–2 chrF++ over
trivial floors are much weaker evidence than they appear. Claims should be
anchored to the `mismatch` control, where the informative dynamic range is
6.20 → 10.91 rather than 9.20 → 10.91.

Note the `zero` condition collapses output diversity to 112 unique hypotheses
(vs 1777), i.e. the pure decoder prior is degenerate and repetitive. `noise` and
`mismatch` keep diversity high (1403 / 1776) while being semantically wrong,
which is why diversity alone (Table 5) cannot establish grounding — a
high-diversity model can still be ungrounded.

## Implications for the results narrative

1. Keep the claim that TConv+CTC learns grounded translation — it is now much
   better supported (`mismatch`, not the retrieval floor, is the evidence).
2. Retire "clears the retrieval floor" as the *headline*. The retrieval floor sits
   at roughly the noise-input level, so clearing it by +1.71 is a low bar.
3. Add the frame-order result as an open weakness: the current best model is
   close to time-invariant on chrF++.
4. Re-read the ST-GCN negative result (`2dfdcbdf4e2d`, BLEU-4 0.356 / chrF++
   8.265) in this light. A spatio-temporal graph encoder's value proposition is
   modelling motion structure; on a benchmark where the exploited signal is
   largely static pose, its failure to beat simpler encoders is coherent rather
   than anomalous.

## Reproduction

Artifacts: `data_full/runs/b0a590ea3317/input_ablation/` (`summary.csv`,
`<mode>_metrics.json`, `<mode>_predictions.csv`).

Paired-bootstrap significance vs `none`:
`docs/logs/SIGNIFICANCE_INPUT_ABLATION.md`.

Caveat on pairing: `mismatch` permutes clip lengths, which changes
length-bucketed batch order, so its rows are realigned to the control by
reference string before the paired bootstrap. 1798 of 1799 references are
unique, so at most one row can be mispaired.
