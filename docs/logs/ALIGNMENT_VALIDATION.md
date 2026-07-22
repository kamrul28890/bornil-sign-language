# WS-E Alignment Validation

Run `b0a590ea3317` · split **random-v1** (`frozen:splits/random-v1`) · test n=1799 · TConv+CTC d256

Params: tolerance=±2 frames, changepoint min_distance=8, velocity smoothing window=5, minima kept below clip quantile=0.5, null trials=100, seed=0. Anchors are labels with peak_conf ≥ 0.5.

## Agreement vs motion changepoints (velocity minima)

| Label set | clips | mean F1 | uniform null | shift null | F1 / shift |
|---|---:|---:|---:|---:|---:|
| anchors (conf≥0.5) | 1222 | 0.0809 | 0.0849 | 0.0841 | 0.96× |
| all labels | 1799 | 0.1925 | 0.2642 | 0.2054 | 0.94× |

## Diagnostic: agreement vs velocity maxima (articulation peaks)

| Label set | clips | mean F1 | shift null | F1 / shift |
|---|---:|---:|---:|---:|
| anchors (conf≥0.5) | 1222 | 0.0231 | 0.0826 | 0.28× |
| all labels | 1799 | 0.1987 | 0.1970 | 1.01× |

## Sanity diagnostics

- Mean span width: 1.01 frames (≈1 → point events, expected from the sparse head)
- Mean aligned labels/clip: 12.2
- Mean motion changepoints/clip: 11.3
- Omitted-label rate (labels receiving no frame): 0.00% (should be ~0 by construction)
- Corr(clip frames, aligned labels): 0.774 (positive → alignment tracks content)

## Replication across all three CTC runs

Same params. `ratio_shift` = mean F1 / shift-null F1 (1.0 = chance).

| Split | anchors vs minima | all vs minima | anchors vs maxima | all vs maxima |
|---|---:|---:|---:|---:|
| random-v1 (b0a590ea3317) | 0.96× | 0.94× | **0.28×** | 1.01× |
| signerindep-v1 (40cffb9e62bb) | 0.89× | 0.92× | **0.30×** | 1.01× |
| unseen-sentence-v1 (a5a9060499a4) | 0.96× | 0.86× | **0.30×** | 1.07× |

The pattern is identical on every split: at chance against velocity minima, and a
strong, reproducible *avoidance* of velocity maxima by the confident anchors
(0.28–0.30×). The metric therefore has power — it detects a real negative
association with peaks — so "at chance against minima" is a genuine null, not a
dead measurement.

## Verdict

AT CHANCE — CTC subword placements do not beat the shift null against velocity
minima, on any split. This is a negative result for the pre-registered hypothesis
(that CTC labels localize to sign-boundary rests). See the reconciliation below;
per the plan we log this to `FAILURES.md` and do **not** proceed to Phase C or
pseudo-glosses on this signal.

## What this does and does not mean

**Does not overturn grounding.** The input ablation already showed the encoder is
genuinely used (deranging the video↔text pairing costs −4.71 chrF++), and A0
showed the CTC head is temporally responsive (distinct labels drop 56% under
frame shuffle). Both stand. This result is narrower: the *frame placement* of CTC
subword spikes does not coincide with keypoint-motion rests.

**Three candidate explanations, which this test cannot separate:**
1. **Granularity mismatch.** Subwords are not signs. One sign can span several
   subwords and one subword can straddle a sign boundary, so subword spikes need
   not sit at sign-level motion rests even for a well-grounded model.
2. **Wrong motion phase.** The confident anchors *avoid* velocity maxima
   (0.28–0.30×) but do not hit minima (≈1.0×). That places them on the rising or
   falling edges of motion — mid-articulation — rather than at either extreme.
   The velocity-minimum reference is then simply the wrong target.
3. **No frame-level correspondence.** The head's temporal responsiveness (A0)
   could be about *which* labels fire, not *when*, leaving no localizable
   frame-level alignment to recover.

The peak-avoidance finding (explanation 2) is the most concrete and is itself a
small positive result worth keeping: whatever the CTC head anchors to, it is
systematically not the moment of maximum hand motion.

**Consequence for WS-E→WS-F.** Frame-level CTC forced alignment is not a validated
bridge to pseudo-glosses or an avatar on this data. If temporal grounding is
pursued further, the next probe should target sign-level (not subword-level)
segmentation and should not assume rests are the anchor — e.g. compare against an
edge/onset signal, or induce sign units first (WS §9.2.5) and align those.
