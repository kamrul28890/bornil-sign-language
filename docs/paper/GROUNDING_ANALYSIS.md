# Paper-ready analysis: visual grounding and metric calibration

Drop-in text for the paper's analysis/discussion section. Present tense, paper
voice. Every number traces to a committed artifact (`docs/logs/INPUT_ABLATION.md`,
`ALIGNMENT_VALIDATION.md`, `WSE_A0_POSTERIOR.md`). Edit register to taste.

---

## Suggested subsection: "Does the model read the video?"

We verify that our best model (TConv+CTC) performs grounded translation rather
than generating fluent Bengali from a language prior, using an input-corruption
control that holds the visual input distribution fixed. On the random-v1 test
split we re-decode with the keypoint stream corrupted while the trained decoder is
left untouched. Deranging only the video↔text correspondence — pairing each
reference with a different clip's real keypoints, so feature statistics and clip
lengths are unchanged — reduces chrF++ from 10.91 to 6.20 (−4.71) and BLEU-4 by
13.7× (paired bootstrap, n=1000, p<0.001). The encoder is therefore genuinely
used: the model's output depends on *which* video it is given, not merely on the
target-language distribution.

This control is stronger evidence of grounding than comparison against trivial
floors, because it changes exactly one factor. We note in particular that
character-level metrics have a high uninformative floor on this open-vocabulary
benchmark: feeding the trained model pure Gaussian noise still yields 9.08 chrF++,
within 0.12 of the nearest-neighbour retrieval floor (9.20) and above the vanilla
baseline on the signer-independent split. Margins of 1–2 chrF++ over such floors
are consequently weak evidence, and we anchor our grounding claim to the mismatch
control (informative range 6.20→10.91) rather than to the floor margin.

## Suggested subsection: "The grounding is static pose, not motion"

The grounding the model exploits is largely static rather than dynamic. Shuffling
frame order within each clip — destroying all temporal structure while preserving
the per-frame pose distribution — costs only 0.88 chrF++ (10.91→10.04, 92%
retained) yet removes 89% of BLEU-4 (0.68→0.07); both changes are significant
(p<0.001). The model behaves largely as a bag-of-poses classifier: temporal order
influences *which* content words are produced (a word-order effect visible in
BLEU-4) but contributes little to character-level overlap. This coheres with our
negative result for a spatio-temporal graph encoder (ST-GCN), whose architectural
advantage is precisely motion modelling: on a benchmark where the exploited signal
is largely static pose, it underperforms simpler encoders (BLEU-4 0.36 vs 0.68 for
TConv+CTC) despite substantially greater training cost.

## Suggested limitation: frame-level temporal alignment

We attempted to recover a frame-level alignment for free from the joint-CTC model
via Viterbi forced alignment, and to validate it against an independent signal
(keypoint-velocity changepoints), since no gold alignment exists. Across all three
splits, agreement does not exceed a circular-shift null (F1/null 0.86–0.96),
i.e. CTC subword placements do not localize to sign-boundary motion rests above
chance. We report this as a limitation rather than a contradiction of the grounding
result above: the CTC head is temporally responsive (shuffling frames collapses its
distinct emitted labels by 56%), but its confident spikes are sparse (~1 per ~12
target subwords) and sit away from motion extrema — they avoid velocity peaks
strongly (F1/null 0.28–0.30) without hitting velocity minima. The likely cause is a
granularity mismatch (subwords are not signs). Frame-level temporal grounding on
Bornil thus remains open and, on this evidence, should be pursued at the sign unit
level rather than the subword level.

---

## Numbers table (for a results/analysis figure or table)

| Probe (random-v1, TConv+CTC) | chrF++ | BLEU-4 | reads as |
|---|---:|---:|---|
| Control (real input) | 10.91 | 0.684 | model as trained |
| Frame order shuffled | 10.04 | 0.074 | static pose only |
| Gaussian-noise input | 9.08 | 0.037 | metric floor |
| **Video↔text mismatched** | **6.20** | **0.050** | grounding removed |
| Zero input | 4.70 | 0.032 | pure decoder prior |
| — retrieval floor (reference) | 9.20 | 0.10 | strongest trivial baseline |

Headline for the caption: the informative gap for grounding is 6.20→10.91
(mismatch→control, +4.71), not 9.20→10.91 (retrieval floor→control, +1.71).
