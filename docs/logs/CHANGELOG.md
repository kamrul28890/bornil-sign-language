# Bornil — Implementation Changelog

> Append-only, newest at top. One entry per meaningful change (code, data, config, decision).
> Keep it terse; link to the experiment log or a commit for detail.

## Format

```
## YYYY-MM-DD — <short title>
- **What:** what changed (files/modules/decision)
- **Why:** motivation / which phase-task this serves
- **Impact:** what now works / what downstream can rely on
- **Commit:** <hash or PR link>
```

---

## 2026-07-25 — WS-C T1.1 close-out: ST-GCN significance + diagnostics
- **What:** Ran `scripts/results_significance.py` for `stgcn` (`2dfdcbdf4e2d`) vs `vanilla` (`c19d0ea08824`) and vs `tconv_ctc` (`b0a590ea3317`) at n=1000, writing `docs/logs/SIGNIFICANCE_STGCN_VS_VANILLA.md` and `docs/logs/SIGNIFICANCE_STGCN_VS_TCONV_CTC.md`. Ran `scripts/prediction_diagnostics.py` on its predictions. Added the run to `docs/logs/RESULTS_OVERVIEW.md` Tables 2 and 5, and checked off the remaining boxes in `docs/plan/2026-07-16-ws-c-phase-d-stgcn.md`.
- **Why:** The full ST-GCN run finished on 2026-07-19 (EXPERIMENTS.md already carried its "negative result, closes WS-C T1.1" verdict from raw metrics), but the plan's D6 (significance) and D7 (results-overview/changelog write-up) tasks were never completed — a loose end found while auditing what's next after the WS-E merge.
- **Impact:** Confirms the raw-metric read statistically: ST-GCN chrF++ (8.265) is significantly **below** both vanilla (p=0.000) and TConv+CTC (p=0.000); BLEU-4 is not significant vs either (p=0.484, p=0.092). Diversity diagnostics show this is not mode collapse (93.3% unique hyps, in line with vanilla) but it does have a distinct failure mode the other full runs don't: 11.3% of rows are runaway repeated-token hypotheses. WS-C T1.1 is now fully closed with the documentation the other model rows have.
- **Commit:** (this commit)

## 2026-07-20 — WS-E Phase B: alignment validation is AT CHANCE (negative result)
- **What:** Added `bornil/align/motion.py` (B1: velocity, smoothing, salient changepoints), `bornil/align/agreement.py` (B2: boundary_f1 + uniform and circular-shift nulls), `scripts/alignment_report.py` (B3), and `docs/logs/ALIGNMENT_VALIDATION.md`. 34 new tests. Ran B3 over all three CTC checkpoints. Logged FAIL-004.
- **Why:** Phase B tests the pre-registered prediction — do CTC forced-alignment placements coincide with an independent motion signal above chance? This is the WS-E go/no-go gate for Phase C and the WS-F bridge.
- **Impact:** **At chance on all three splits.** CTC subword placements do not beat the circular-shift null against velocity minima (ratio_shift 0.86–0.96×), replicated on random-v1/signerindep-v1/unseen-sentence-v1. The metric has power — confident anchors show a strong, reproducible *avoidance* of velocity maxima (0.28–0.30×) — so this is a genuine null, not a dead measurement. **Phase C is NOT unlocked** (plan gate). Crucially this does NOT overturn grounding: the input ablation (mismatch −4.71 chrF++) and A0 (temporally-responsive head) stand; the negative result is narrower — subword spikes don't localize to sign-boundary motion rests, likely a subword≠sign granularity/phase issue (anchors sit mid-articulation, avoiding both motion extrema). Two in-flight fixes: B1's detector was saturating the reference (36.9→11.3 changepoints/clip via smoothing + deepest-quantile filtering; verdict unchanged), and a velocity-maxima diagnostic was added beyond the plan.
- **Commit:** (this commit)

## 2026-07-20 — WS-E A2: CTC forced-alignment extraction (with per-label confidence)
- **What:** Added `scripts/align_checkpoint.py` (`span_confidences` / `spans_to_rows` / `align_split`, 8 tests) and `docs/logs/WSE_A2_ALIGNMENT.md`. Ran it over all three CTC checkpoints; wrote `data_full/runs/<id>/alignments.csv` (gitignored, regenerable).
- **Why:** Task A2 of WS-E — produce the frame→subword alignments Phase B validates against motion changepoints.
- **Impact:** **0% infeasible on all three splits** (random/signerindep/unseen-sentence), so the 512-frame ceiling never truncates a target — the plan's >5% investigation gate is clear. Added a `peak_conf` column (amendment prompted by A0): confidence is sharply bimodal (90th pct 0.07, 95th pct 0.94), so conf≥0.5 cleanly separates ~0.85 real anchors/clip from ~11 tie-break placements; ~65% of clips have ≥1 anchor; spans are ~1 frame (point events). `row` is the recording id, a real Phase-B join key. These findings redefine Phase B: weight/filter by peak_conf, compare points to changepoints (not intervals), effective N is anchored clips (~1150 on random-v1). signerindep is weakest (0.78 anchors/clip, 63% coverage), same ordering as chrF++.
- **Commit:** (this commit)

## 2026-07-19 — WS-E A0: CTC posterior sanity check passes (gate for Phase B)
- **What:** Added `bornil/align/posterior.py` (pure per-clip CTC posterior statistics, 10 tests) and `scripts/ctc_posterior_stats.py` (A0 CLI over a checkpoint, with a frame-shuffle contrast), plus `docs/logs/WSE_A0_POSTERIOR.md`. A0 was inserted ahead of A2 as a Phase-B gate after the input ablation.
- **Why:** Forced alignment only yields an interpretable segmentation if the CTC head emits real per-frame label evidence; a degenerate head (all-blank / one constant label) would make Phase B's agreement-vs-chance meaningless. Check this cheaply before building Phase B.
- **Impact:** **PASS, not degenerate** — on `b0a590ea3317` the head is confident (entropy 3.7% of the uniform ceiling) and emits 2.72 distinct non-blank labels/clip, and it is temporally responsive (distinct labels drop 56% under frame shuffle, reconciling with the −0.88 chrF++ / −89% BLEU-4 ablation split). Caveat carried into A2: the head is 97.4% blank and only ~2.7 of ~12.2 target subwords per clip have a confident spike, so forced alignment will produce a few well-anchored labels and many weak placements — A2 should emit a per-label confidence so B2 can weight boundaries. Mildly raises confidence in the "above-chance" half of the pre-registered Phase-B prediction.
- **Commit:** (this commit)

## 2026-07-19 — WS-E A1: CTC forced alignment (Viterbi)
- **What:** Added `bornil/align/ctc_forced.py` (`forced_align` / `path_to_spans` / `align_clip`) and `tests/align/test_ctc_forced.py` (16 tests). Recorded plan amendments and a pre-registered Phase-B prediction in the WS-E plan.
- **Why:** Task A1 of WS-E — the frame→subword alignment primitive Phase A2/B build on.
- **Impact:** Vectorized Viterbi (12.6 ms/clip at T=512 vs 0.63 s for the plan's Python double loop, ~50x; matters because Phase B re-aligns under many randomized permutations). Tests exercise the DP itself, including a brute-force max-likelihood check — the plan's original cases would have passed a greedy stand-in.
- **Commit:** (this commit)

## 2026-07-19 — Input ablation: grounding confirmed, but it is static pose
- **What:** Added `scripts/input_ablation.py` (re-decodes a checkpoint with the keypoint stream corrupted: `mismatch` / `frame_shuffle` / `noise` / `zero` / `none`), `docs/logs/INPUT_ABLATION.md`, and `docs/logs/SIGNIFICANCE_INPUT_ABLATION.md`. Ran it on `b0a590ea3317` (TConv+CTC, random-v1).
- **Why:** The ST-GCN full run (`2dfdcbdf4e2d`) came in below the trivial floors on chrF++ after 42.8 h, raising the question of whether any of these models use the video at all, or whether the decoder's Bengali prior produces the output.
- **Impact:** **The encoder is genuinely used** — deranging only the video↔text pairing costs −4.71 chrF++ (10.91 → 6.20) and 13.7× BLEU-4, with the input distribution held exactly constant. This is a stricter control than the trivial floors and is now the strongest grounding evidence in the project. **But the grounding is static pose, not motion:** shuffling frame order within clips costs only −0.88 chrF++ (92% retained) while BLEU-4 retains 11%. Also: Gaussian-noise input scores 9.08 chrF++ vs a 9.20 retrieval floor, so "clears the retrieval floor by +1.71" is a much lower bar than it reads — claims should be anchored to the `mismatch` control (dynamic range 6.20 → 10.91). Gives WS-E a concrete target (close the `frame_shuffle` gap) and makes the ST-GCN negative result coherent rather than anomalous.
- **Commit:** (this commit)

## 2026-07-19 — Shared model builder; fixed CTC checkpoint reload
- **What:** Added `bornil/models/build.py` as the single source of truth for reconstructing a model from a run `config.json`, with `tests/models/test_build.py` (11 tests). Repointed `scripts/train_sample.py`, `scripts/eval_checkpoint_decode.py`, and `scripts/input_ablation.py` at it.
- **Why:** `eval_checkpoint_decode.py` had a drifted copy of the construction logic that never passed `use_ctc`, so **every CTC checkpoint failed to load** (`Unexpected key(s) in state_dict: "ctc_head.weight"`); it also hardcoded `SignTransformer`, so BiLSTM and ST-GCN runs could not be evaluated at all. Found while building the input ablation.
- **Impact:** All three architectures now load and decode from saved checkpoints (verified on `b0a590ea3317`, `2dfdcbdf4e2d`, `07709687f916`); decode ablations are usable on CTC runs for the first time. Two deliberate behaviour changes: an unknown `model.arch` now raises instead of silently falling back to a transformer, and `arch=bilstm` with `ctc_head=true` now raises instead of silently ignoring the head.
- **Commit:** (this commit)

## 2026-07-16 — Trivial floors + claims reconciled with the CTC model
- **What:** Rewrote `docs/logs/RESULTS_OVERVIEW.md` Table 4 (trivial floors now compare retrieval vs vanilla vs TConv+CTC across BLEU-4/chrF++/ROUGE-L), added Table 5 (output diversity & length calibration), and corrected "Main Claims To Write".
- **Why:** Table 4 still compared only the vanilla model and claimed "retrieval chrF++ exceeds model chrF++" — stale now that CTC is the best model. The CTC chrF++ gain also needed a control against the "chrF++ rewards generic output" explanation.
- **Impact:** **TConv+CTC is the first model to clear the retrieval floor on all three splits on every metric** (vanilla lost to retrieval on chrF++ on both hard splits) — this is now the headline claim. Diversity diagnostic rules out generic output: CTC raises unique-output rate from ~90–94% to 98–99% and fixes under-generation (7.56 vs 7.50 ref length on random-v1) where `early_eos_penalty=0.2` collapsed diversity. Generalization gap documented as model-dependent: CTC shows no unseen-sentence penalty; signerindep still under-generates (7.43 vs 9.40) and remains unsolved.
- **Commit:** (this commit)

## 2026-07-16 - Joint CTC hard-split confirmation
- **What:** Ran `configs/full_ctc.yaml` on `signerindep-v1` (`40cffb9e62bb`) and `unseen-sentence-v1` (`a5a9060499a4`), regenerated random-v1 CTC significance at n=1000, and added `docs/logs/SIGNIFICANCE_CTC_HARDSPLITS.md`.
- **Why:** Confirm whether the random-v1 joint-CTC chrF++ gain generalizes beyond the easiest split and firm up borderline CTC p-values.
- **Impact:** The chrF++ gain holds on both hard splits: signerindep-v1 10.361 vs 8.382 (p=0.000) and unseen-sentence-v1 10.943 vs 9.187 (p=0.000). Random-v1 remains chrF++-significant at n=1000 vs vanilla and TConv (both p=0.000), while random-v1 BLEU-4 remains not significant.
- **Commit:** (this commit)

## 2026-07-16 — Results overview reconciled with significance
- **What:** Updated `docs/logs/RESULTS_OVERVIEW.md` — added the CTC and BiLSTM runs to the random-v1 ablation table, rewrote the ablation decision, and corrected "Main Claims To Write" to match the Phase A/B/C paired-bootstrap results.
- **Why:** The overview still framed TConv as a useful win and BLEU-4 as a headline metric; significance testing contradicts both.
- **Impact:** TConv-alone is documented as *not* a supported win (BLEU-4 gain n.s. p=0.486); joint CTC/attention is the current best model-side change (chrF++ significantly > vanilla and > TConv, p=0.000, and it enables WS-E); BLEU-4 is flagged as noise-floor for this benchmark. Pending: confirm CTC on `signerindep-v1` / `unseen-sentence-v1`; re-run borderline comparisons at n=1000.
- **Commit:** (this commit)

## 2026-07-16 - BiLSTM encoder baseline
- **What:** Added `bornil/models/bilstm.py`, `configs/full_bilstm.yaml`, and `model.arch` selection in `scripts/train_sample.py`; completed BiLSTM random-v1 run `07709687f916`.
- **Why:** Phase C puts the classic WS-C T0.1 BiLSTM baseline on the leaderboard using the shared decoder contract.
- **Impact:** BiLSTM reached BLEU-4 0.621 / chrF++ 8.990 / ROUGE-L 0.064. BLEU-4 did not significantly beat the vanilla Transformer (`p=0.490`, n=200), and chrF++ was significantly lower (`p=0.000`).
- **Commit:** (this commit)

## 2026-07-16 - Joint CTC/attention ablation
- **What:** Added `configs/full_ctc.yaml`, wired `model.ctc_head` and `train.ctc_weight` through `scripts/train_sample.py`, and completed the TConv+CTC random-v1 run (`b0a590ea3317`).
- **Why:** Phase B tests joint CTC/attention as an encoder regularizer and WS-E alignment prerequisite.
- **Impact:** TConv+CTC reached BLEU-4 0.684 / chrF++ 10.913 / ROUGE-L 0.067. BLEU-4 did not significantly beat TConv-only (`p=0.880`, n=200), but chrF++ improved significantly (`p=0.000`).
- **Commit:** (this commit)

## 2026-07-16 - Significance & CIs added
- **What:** Added `scripts/results_significance.py` and `docs/logs/SIGNIFICANCE.md` with bootstrap confidence intervals and paired p-values for the saved random-v1 vanilla/TConv ablation runs.
- **Why:** Phase A requires CIs and paired significance before model-gain claims are used in the report.
- **Impact:** TConv's random-v1 BLEU-4 gain over vanilla is **not significant** at p<0.05 (`p=0.486`, bootstrap n=1000), so the paper should not claim a statistically supported TConv win yet; the EOS-penalty regression is significant.
- **Commit:** (this commit)

## 2026-07-16 — Qualitative example selection
- **What:** Added `scripts/select_qualitative_examples.py` and generated `docs/logs/QUALITATIVE_EXAMPLES.md` from per-example prediction scores.
- **Why:** The report needs representative prediction examples, not only corpus-level metric tables.
- **Impact:** Official and LTE12 runs now have automatic exact/best, representative, and failure examples ready for human review.
- **Commit:** (this commit)

## 2026-07-16 — Results section draft
- **What:** Added `docs/logs/RESULTS_SECTION_DRAFT.md` as report-ready prose for the official baselines, ablations, LTE12 diagnostic track, and decode stability result.
- **Why:** Convert experiment logs into a coherent paper/report narrative before starting more model training.
- **Impact:** The current results can now be copied into a manuscript draft with clear separation between official and diagnostic tracks.
- **Commit:** (this commit)

## 2026-07-16 — Result reporting structure
- **What:** Added `docs/logs/RESULTS_OVERVIEW.md` and guarded `build_baseline_table.py` against accidentally selecting filtered LTE12 runs by default.
- **Why:** The project now has official full-data results, full-data ablations, filtered LTE12 diagnostics, and decode diagnostics; these need separate report tables.
- **Impact:** The main baseline table remains the official unfiltered benchmark, while filtered LTE12 is clearly documented as a diagnostic track.
- **Commit:** (this commit)

## 2026-07-16 — Decode repetition guard
- **What:** Added opt-in `repetition_stop` decoding support for greedy and beam generation, plus `eval_checkpoint_decode.py --repetition-stop`.
- **Why:** LTE12 diagnostics still showed rare runaway repeated-token hypotheses even when corpus metrics improved.
- **Impact:** Existing official numbers are unchanged by default; checkpoints can now be re-decoded with a repetition guard to test whether those pathological rows can be clipped without retraining.
- **Commit:** (this commit)

## 2026-07-15 — LTE12 filtered TConv ablation
- **What:** Added `data.max_target_tokens` split filtering, `configs/full_tconv_lte12.yaml`, and `scripts/run_lte12_ablation.ps1`.
- **Why:** Target-length audit showed 84.6% of rows are <=12 tokens, while long article-like targets dominate the failure cases; this tests the same TConv model on the cleaner short/medium subset.
- **Impact:** Filtered experiments now preserve frozen split membership and remove long-target rows within each partition, so model, trivial floors, and diagnostics stay comparable.
- **Commit:** (this commit)

## 2026-07-15 — TConv length-pressure ablation
- **What:** Added optional `early_eos_penalty` training loss and `configs/full_tconv_len.yaml`.
- **Why:** TConv improved BLEU-4 but still under-generated and did not beat vanilla ROUGE-L; this ablation directly discourages premature EOS before running harder splits.
- **Impact:** Existing configs default to `early_eos_penalty: 0.0`; the `0.2` ablation collapsed output diversity and is documented as a failed setting in `docs/logs/TCONV_LENGTH_PRESSURE_RESULT.md`.
- **Commit:** (this commit)

## 2026-07-15 — Temporal-convolution encoder ablation
- **What:** Added optional temporal convolution layers before the Transformer encoder and a `configs/full_tconv.yaml` experiment config.
- **Why:** Diagnostics show the vanilla model under-generates and degrades sharply on longer references; a local temporal frontend is the first cheap model-side attempt to improve visual grounding.
- **Impact:** The default model remains checkpoint-compatible (`temporal_conv_layers: 0`), while `full_tconv.yaml` runs a stronger random-v1 baseline before deciding whether to evaluate all official splits.
- **Commit:** (this commit)

## 2026-07-15 — Eval-only decode ablation script
- **What:** Added `scripts/eval_checkpoint_decode.py`, which loads an existing run checkpoint and evaluates alternate decode settings without retraining. `evaluate()` now passes `length_penalty` through to beam search.
- **Why:** Let us test beam size, max decode length, and beam length penalty cheaply before spending GPU time on new model training.
- **Impact:** Existing full-dataset checkpoints can now produce decode-ablation prediction/metric files under `<run>/decode_ablations/`.
- **Commit:** (this commit)

## 2026-07-15 — Full official baseline suite + trivial floors
- **What:** Completed full-dataset SignTransformer baseline runs on `random-v1`, `unseen-sentence-v1`, and `signerindep-v1`; added `docs/logs/BASELINE_TABLE.md` and `docs/logs/TRIVIAL_FLOORS.md`.
- **Why:** Establish the paper-ready benchmark floor across all official splits before model ablations.
- **Impact:** The model beats trivial floors on BLEU-4 and ROUGE-L across all splits, while retrieval remains competitive on chrF++ for harder splits; the main diagnosis is weak visual grounding and signer/session generalization.
- **Commit:** (this commit)

## 2026-07-14 — Official split loading + greedy decode policy
- **What:** `train_sample.py` and `trivial_baselines.py` now resolve frozen split files from `splits/<kind>` for full-corpus runs, while sample/trial manifests fall back to generated local splits. Added `--split`, `--split-root`, `--no-frozen-splits`, and `--beam-size`; `configs/full.yaml` now uses official greedy decoding (`beam_size: 1`) with beam search reserved for explicit ablations.
- **Why:** The previous full run used an ad-hoc random split, not `splits/random-v1`; official benchmark numbers must be reproducible and decode policy must be explicit.
- **Impact:** Official `kp-218` reruns can now target frozen `random-v1`, `signerindep-v1`, and `unseen-sentence-v1`, with split source and decode mode saved in config/log rows.
- **Commit:** (this commit)

## 2026-07-14 — Milestone: full dataset extracted + first real baseline
- **What:** Full 100% run completed — all 17,988 clips extracted (512-frame ceiling), packed `(17988,512,258)`, and the SignTransformer baseline trained on GPU (35 min, random-v1 split, train 14390 / test 1799).
- **Result:** BLEU-4 **0.12**, chrF **10.61**, chrF++ **8.39**, ROUGE-L 0.034 (EXP data_full-4e112ae230b9).
- **Diagnosis:** outputs are fluent Bengali but not grounded to the signs (819/1799 unique, none empty) — decoder learned the language prior; visual grounding weak. Expected for a vanilla untuned model on 14k open-vocab pairs; not yet a tuned baseline.
- **Next:** trivial/retrieval floors for context; training improvements (warmup+scheduler, label smoothing, beam search); evaluate on signerindep-v1 + unseen-sentence-v1; then the model suite (WS-C/WS-D).
- **Commit:** (this commit)

## 2026-07-13 — GPU enabled + quieter extraction + --percent/run_trial
- **What:** Installed CUDA torch (2.6.0+cu124) — training now runs on the RTX 3070 (`device: auto`/`--device cuda`, verified). Added `--percent N` to `make_sample` and a one-command `scripts/run_trial.py` (sample→extract→pack→train). `bornil/quiet.py` silences benign cv2/mediapipe/ffmpeg/protobuf noise and redirects native fd-2 warnings to `<out_dir>/extract_stderr.log` so the console shows only the progress line.
- **Why:** User wanted GPU training, a percentage argument, and was (understandably) alarmed by the wall of harmless MediaPipe/ffmpeg warnings.
- **Impact:** `python scripts/run_trial.py --percent 2 --device cuda` runs the whole slice on GPU with a clean console. Those "errors" were all benign warnings (MediaPipe `inference_feedback_manager`, ffmpeg `matroska,webm Unknown-sized element`, TFLite/absl/protobuf notices) — extraction was working (`ok=N fail=0`). Full suite 59 passing on cu124.
- **Commit:** (this commit)

## 2026-07-13 — 5% trial pipeline: end-to-end vertical slice runs locally
- **What:** Greedy `generate()` on SignTransformer; `bornil/train/loop.py` (train_epoch + evaluate w/ locked metrics); trial scripts `make_sample`/`extract_sample`/`pack_sample`/`train_sample` + `configs/trial_5pct.yaml`; `bornil/mp_guard.py` (tensorflow-block so mediapipe imports); `requirements.txt` pinned; `.gitignore` for trial artifacts.
- **Why:** Prove the whole pipeline (sample→extract→pack→train→decode→score→log) on this PC before committing cloud compute to the full 17,988 clips (user request: 5% trial).
- **Impact:** **8-clip smoke passed end-to-end** — MediaPipe extraction (128,258) at ~6.7 s/clip, packing, training, greedy decode, metric scoring, run folder (config+hash, metrics.json, predictions.csv) + EXPERIMENTS row all working. Full suite **59 passing**. (Smoke BLEU=0 expected: 6 training clips.) Extraction rate ⇒ full 5% (~900 clips) ≈ 1.5–2 h.
- **Fix:** mediapipe/tensorflow DLL crash resolved — see FAIL-003.
- **Commit:** (this commit)

## 2026-07-13 — Model layer: tokenizer + SignDataset + Transformer baseline (WS-A/WS-C)
- **What:** Installed CPU torch 2.11 + sentencepiece. `bornil/data/tokenizer.py` (SentencePiece wrapper, BOS/EOS + special-strip; tested on real `bsl_sp.model`). `bornil/data/dataset.py` (`SignDataset` over packed features + `make_collate_fn`; tested on synthetic memmap). `bornil/models/transformer.py` (`SignTransformer` encoder-decoder SLT baseline, sinusoidal PE, causal + padding masks). `conftest.py` + `bornil/runtime.py` enforce torch-before-pandas import order.
- **Why:** First runnable SLT baseline (plan WS-C T0.2) and the data contract feeding it.
- **Impact:** **Overfit-tiny gate automated and passing** — the Transformer memorizes a synthetic clip→subword batch (loss < 0.05, exact teacher-forced reproduction), proving the arch/masking/loss are correct before any GPU/cloud run. Full suite **55 passing, exit 0**.
- **Fix:** resolved a Windows segfault (pandas-before-torch DLL conflict) — see FAIL-002.
- **Commit:** (this commit)

## 2026-07-13 — Phase 0: config system + keypoint-transform tests + disk triage
- **What:** `bornil/config.py` (YAML load/save, deep-merge, stable order-independent config hash) with 7 tests. Moved keypoint transforms to `bornil/data/transforms.py` (canonical) with 13 characterization tests locking normalization/selection behavior; `bsl_utils.py` is now a re-export shim so teammate notebooks keep working. Full suite: **42 passing**.
- **Why:** Config hash tags every future run (plan §4.3); characterization tests let us refactor WS-A + run normalization ablations without silently changing the model input contract.
- **Impact:** `bornil.config` + `bornil.data.transforms` ready for the training loop; input-contract behavior is now regression-guarded.
- **Disk triage:** D: hit 100% full. Root cause: `chunks/` = **172 GB** videos (on-disk here, gitignored) + **~86 GB dangling git objects** (only 72 of 7,476 objects reachable — leftover from large files staged-but-never-committed). User freed ~99 GB manually (now 79% used). **Still recommended:** `git reflog expire --expire=now --all && git prune --expire=now && git gc` to reclaim the ~86 GB (pending user go-ahead). Logged in FAILURES as FAIL-001.
- **Commit:** (this commit)

## 2026-07-13 — Phase 0 (WS-B spine): official splits + metric protocol
- **What:** Created `bornil/` package; `bornil/benchmark/splits.py` (random-v1, signerindep-v1 [chunk-as-session], unseen-sentence-v1 + leakage checks + file I/O) and `bornil/benchmark/metrics.py` (SacreBLEU BLEU-1..4 @ `intl` tokenizer, chrF/chrF++, ROUGE-L, bootstrap CI, NFC-normalized; optional BanglaBERT BERTScore). `scripts/make_splits.py`, 22 passing tests, `pyproject.toml`.
- **Why:** WS-B is the spine everything else scores against (plan §6). Split correctness = no leakage.
- **Impact:** Frozen split files under `splits/` (leakage-checked on all 17,988 clips: random 14390/1799/1799; signerindep 29/4/3 chunks = 14500/1988/1500; unseen-sentence with zero text overlap). Locked, reproducible metric protocol callable via `corpus_metrics()`.
- **Findings:** 36 chunks total; 17,928 unique normalized sentences (near-total uniqueness confirms open-vocabulary framing).
- **Env note:** dev machine has only numpy/pandas/pytest/pyyaml/sacrebleu — no torch/sentencepiece yet; MediaPipe features live off-repo. Model/data-dependent phases run in the cloud/teammate env.
- **Commit:** (this commit)

## 2026-07-13 — Repo synced + research program plan created
- **What:** Merged teammate's keypoint pipeline (`easha003/BdSL`) into `main`; wrote `docs/plan/2026-07-13-bornil-implementation-plan.md` and logging scaffold.
- **Why:** Establish the shared plan of record before Phase 0.
- **Impact:** Team has a single high-level→detailed plan; logging conventions defined.
- **Commit:** d9b2d65
