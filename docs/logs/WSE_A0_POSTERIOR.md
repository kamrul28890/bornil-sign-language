# WS-E A0 — CTC posterior sanity check (gate for Phase B)

Does the joint-CTC head emit genuine per-frame label evidence, or is it
degenerate? If degenerate, forced alignment (A1) segments by tie-breaking rather
than signal and Phase B's agreement-vs-chance measures nothing. Cheap check on an
existing checkpoint, no retraining.

Run: `data_full/runs/b0a590ea3317` (TConv+CTC, random-v1), test split n=1799,
blank = pad_id = 0.

```
python scripts/ctc_posterior_stats.py --run data_full/runs/b0a590ea3317
```

## Results

| Input | blank_rate | distinct non-blank / clip | mean argmax run (frames) | entropy (nats) | entropy / ln V |
|---|---:|---:|---:|---:|---:|
| real | 0.974 | 2.72 | 55.6 | 0.306 | 0.037 |
| frame-shuffled | 0.987 | 1.20 | 119.5 | 0.558 | 0.067 |

Target length for context: **12.2 subword tokens per clip on average** (median 9).

## Verdict: PASS (not degenerate), with a caveat

**The head is not degenerate.** It is highly confident (entropy is 3.7% of the
uniform ceiling) and emits 2.72 distinct non-blank labels per clip — comfortably
above the "~1 label" degeneracy threshold. Phase B can proceed.

**But it is very sparse.** 97.4% of frames argmax to blank; only ~2.6% carry a
label. The model produces its translation from a handful of high-confidence label
spikes, not a dense per-frame labelling.

**And it is temporally responsive.** Shuffling frame order collapses distinct
labels 2.72 → 1.20 (−56%) and doubles the mean argmax run length (55.6 → 119.5).
So temporal order genuinely drives *which* labels fire — the head is not
time-invariant, despite `frame_shuffle` costing the end-to-end model only −0.88
chrF++ (`INPUT_ABLATION.md`). The reconciliation: the head's temporal sensitivity
shows up in *which* content words are produced (a BLEU-4 / word-order effect,
where shuffling cost 89%) and barely in character overlap (chrF++), because a
sparser, more generic label set still decodes to fluent ~10-chrF++ Bengali.

Note `blank_rate` is the wrong probe for time-sensitivity — it barely moves
(0.974 → 0.987) because most frames are blank under either input. The distinct
non-blank label count is the responsive statistic.

## Caveat carried into A2 / Phase B

Forced alignment forces **all** target labels into the frame budget, but only
~2.7 of the ~12.2 target subwords per clip correspond to a confident head spike.
The remaining ~9 labels will be placed in near-blank, low-evidence regions where
Viterbi is choosing among nearly-tied blank frames. **Expect a few well-localized
anchor labels per clip and many weakly-determined placements.** Phase B's
boundary-agreement metric should therefore be read as dominated by the anchor
labels; a per-label confidence signal (e.g. peak posterior at the assigned frame)
would let B2 weight or filter placements rather than treat all label boundaries
as equally reliable. Consider adding that to A2's output.

## Effect on the pre-registered prediction

The prediction (`docs/plan/2026-07-16-ws-e-alignment.md`, "weak but above-chance"
agreement) stands, and A0 mildly *raises confidence in the "above-chance" half*:
there is real, temporally-driven label structure for the anchor labels to align
to. The "weak" half is also supported — most labels lack a confident anchor, so
per-boundary agreement should be partial, not strong. A0 does not change the
prediction; it sharpens why both halves are expected.

Artifacts: A0 writes a JSON summary via `--out`; not committed (under gitignored
`data_full/`). Reproduce with the command above.
