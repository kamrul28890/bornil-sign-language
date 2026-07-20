# WS-E A2 — CTC forced alignment extraction

Runs A1 (Viterbi forced alignment) over the test split of each joint-CTC run and
writes one row per aligned label occurrence, with a per-label peak-posterior
confidence. Consumes existing checkpoints; no retraining.

```
python scripts/align_checkpoint.py --run data_full/runs/<id> --device cuda \
    --out data_full/runs/<id>/alignments.csv
```

Output columns: `row` (recording id — the Phase B join key), `label_index`,
`token_id`, `token_str`, `start_frame`, `end_frame` (inclusive), `n_frames`,
`peak_conf`.

## Results

| Run | Split | clips | infeasible | labels/clip | anchors/clip (conf≥0.5) | clips with ≥1 anchor |
|---|---|---:|---:|---:|---:|---:|
| b0a590ea3317 | random-v1 | 1799 | 0.0% | 12.2 | 0.85 | 1222 (68%) |
| 40cffb9e62bb | signerindep-v1 | 1500 | 0.0% | 15.7 | 0.78 | 950 (63%) |
| a5a9060499a4 | unseen-sentence-v1 | 1799 | 0.0% | 12.4 | 0.88 | 1225 (68%) |

## Findings

**No infeasibility.** 0% of clips failed to align on all three splits, so the
512-frame ceiling never truncated a target below its minimum frame budget. The
plan's >5%-infeasible investigation gate is clear.

**Confidence is sharply bimodal.** On random-v1 the 90th percentile of `peak_conf`
is 0.07 but the 95th is 0.94 — almost nothing sits in between. `peak_conf ≥ 0.5`
is therefore a clean anchor/non-anchor split, not an arbitrary cut. This is the
A0 sparsity finding made per-label: of the ~12 subwords forced into each clip,
only ~1 lands on a frame the head actually favours; the rest are parked on
near-blank frames by Viterbi tie-breaking.

**Alignments are point events, not durations.** Median span width is 1 frame
(mean 1.0, max 2). The head spikes on isolated frames, so forced alignment places
each label at essentially a single time point rather than over an interval.

**Signer-independent is weakest, consistent with everything else.** It has the
most labels/clip (15.7, longer targets in that split) but the fewest anchors/clip
(0.78) and fewest clips with any anchor (63%). The model that generalizes worst
across signers also produces the fewest confident temporal anchors — the same
ordering as chrF++ across splits.

## What this dictates for Phase B

1. **Weight or filter by `peak_conf`.** Treating all ~12 labels/clip as equally
   reliable boundaries would swamp the ~1 real anchor in ~11 tie-break placements.
   B2's agreement-vs-chance must run on the anchor subset (conf ≥ 0.5), and should
   also report the all-labels version to show the difference.
2. **Compare points to points.** Because spans are ~1 frame, boundary agreement is
   a point-to-changepoint distance (within ±k frames), not interval overlap.
3. **~65% clip coverage.** Roughly a third of clips have no confident anchor and
   contribute only tie-break placements; the agreement metric's effective N is the
   anchored clips, ~1150 on random-v1, not 1799.

The pre-registered prediction ("weak but above-chance") is unchanged and now has a
concrete mechanism: agreement should hold on the anchored ~0.85 labels/clip and
degrade toward chance if computed over all placements.

Artifacts (regenerable, under gitignored `data_full/`):
`data_full/runs/<id>/alignments.csv`.
