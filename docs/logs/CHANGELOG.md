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
