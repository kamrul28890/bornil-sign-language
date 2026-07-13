# Bornil BdSL — Research Program Implementation Plan

> **Status:** Plan of record. Living document — revise as phases complete; log every change in `docs/logs/CHANGELOG.md`, every run in `docs/logs/EXPERIMENTS.md`, every dead-end in `docs/logs/FAILURES.md`.
> **Author/owner:** Bornil team · **Created:** 2026-07-13 · **Primary venue target:** NeurIPS Datasets & Benchmarks track (2027 cycle, ~May 2027 deadline assumed).
> **Convention:** This plan discusses *approaches, libraries, and algorithms* — not code. Each phase ends with explicit **Validation** (how we know it worked) and **Failure modes → fixes**.

---

## 0. How to read this document

The plan is a **research program**, not a single feature. It flows **high-level → detailed**:

- **§1 Vision & scope** — what we are building and why, the NeurIPS target vs. the extended program.
- **§2 Program map** — the workstreams and how they connect (one picture).
- **§3 Timeline & compute strategy** — pacing on RTX 3070 + cloud.
- **§4 Foundations** — repo, config, experiment tracking, reproducibility.
- **§5–§10 The six workstreams** (map 1:1 to your six requirements), each with phases, tasks, deliverables, validation, and failure→fix notes.
- **§11 Analysis & ablations · §12 Paper & artifacts · §13 Risk register · §14 Glossary.**

**Requirement → workstream map** (so nothing you asked for is lost):

| Your requirement | Workstream | Section |
|---|---|---|
| (4) Data preprocessing pipeline — what/how we're getting | **WS-A Data** | §5 |
| (2) Dataset benchmarking | **WS-B Benchmark** | §6 |
| (1) Established / latest / transformer / video model training + eval | **WS-C SLT models** | §7 |
| (3) Novel approach | **WS-D Novel** | §8 |
| (5) Sentence↔video mapping (word, frame/clip connections) | **WS-E Alignment** | §9 |
| (6) Avatar-based animation + text→sign endpoint | **WS-F Production/Avatar** | §10 |

---

## 1. Vision & scope

**Vision.** Bornil is a large (17,988 clips / ~66.8 h), open-vocabulary, continuous **Bangla Sign Language (BdSL)** video corpus with sentence-level Bangla labels. We will turn it into (a) the **official benchmark** for gloss-free BdSL translation, (b) a **comprehensive model suite** (established → transformer → video → novel), (c) an **alignment layer** that induces word/clip correspondence from sentence-only labels, and (d) a **sign-language production** system that animates an avatar from arbitrary Bangla sentences, served behind a model endpoint.

**Task reality (important, sets expectations).** Bornil has **sentence-level labels only** — no glosses, no per-word timing — and **~100% unique sentences** (17,930 unique / 17,988 clips). So:
- The core recognition task is **gloss-free Sign Language Translation (SLT)**: video → Bangla sentence. **Gloss WER is not applicable** (no glosses); we correct the earlier research-doc's CSLR framing for Bornil specifically.
- Because sentences almost never repeat, models **cannot memorize** — the test set is inherently **open-vocabulary / unseen-sentence**. This is a genuine strength to advertise, and it also means **absolute BLEU will be modest** (single-digit to low-teens BLEU-4, as on How2Sign). The contribution is a **rigorous benchmark + relative gains**, not a BLEU record. Every claim uses **bootstrap confidence intervals**, not point estimates.

**Scope tiers.**
- **NeurIPS D&B core (must-ship):** WS-A (data) + WS-B (benchmark) + WS-C (baselines) + the best WS-D novel method + §11 analysis. This is one self-contained paper.
- **Extended program (parallel / follow-on):** WS-E alignment and WS-F avatar/production. WS-E is *also* a benchmark analysis contribution (interpretability of alignment) and the *bridge* to WS-F. WS-F is a strong demo + a candidate second paper (SLP). Both are in this plan because they share the data/alignment foundation, but they are **not gating** for the D&B submission.

---

## 2. Program map

```
                         ┌──────────────────────────────────────────┐
                         │  WS-A  DATA PIPELINE  (§5)                │
                         │  raw video → keypoints(+face/rgb/flow)    │
                         │  packed features · tokenizer · quality    │
                         └───────────────┬──────────────────────────┘
                                         │ representations + manifest
              ┌──────────────────────────┼───────────────────────────┐
              ▼                          ▼                            ▼
   ┌────────────────────┐   ┌──────────────────────────┐   ┌────────────────────┐
   │ WS-B BENCHMARK(§6) │   │ WS-C SLT MODELS (§7)     │   │ WS-E ALIGNMENT(§9) │
   │ splits · metrics · │◄──┤ established→transformer→ │──►│ word/clip↔frame    │
   │ eval harness ·     │   │ video · cross-dataset    │   │ pseudo-glosses     │
   │ datasheet/leaderbd │   └───────────┬──────────────┘   └─────────┬──────────┘
   └────────┬───────────┘               │                            │
            │                 ┌─────────▼──────────┐                 │
            │                 │ WS-D NOVEL (§8)    │                 │
            │                 │ SSL-KP · Mamba ·   │                 │
            │                 │ KP→BanglaLLM · fuse│                 │
            │                 └─────────┬──────────┘                 │
            ▼                           ▼                            ▼
   ┌───────────────────────────────────────────────┐     ┌────────────────────────┐
   │ §11 ANALYSIS + §12 PAPER/ARTIFACTS  (NeurIPS)  │     │ WS-F AVATAR/SLP (§10)  │
   └───────────────────────────────────────────────┘     │ text→pose→avatar·endpt │
                                                          └────────────────────────┘
```

Everything hangs off **WS-A**. The benchmark (**WS-B**) scores everything. Alignment (**WS-E**) both explains SLT models and feeds production (**WS-F**).

---

## 3. Timeline & compute strategy

**Compute reality:** dev/prototyping on a single **RTX 3070 (8 GB)**; full training + sweeps on **cloud (A100/H100 rentals)**. Keypoint models (218-dim input) are tiny and fit the 3070 comfortably; **video models (ViViT/VideoMAE) and large sweeps are cloud-only**.

**Golden rule (applies to every model phase):** *never* start a cloud run before an **overfit-tiny** sanity pass on the 3070 (see §4.6). This makes cloud spend buy signal, not debugging.

**Indicative pacing toward NeurIPS 2027 (~10 months).** Dates relative to "now" (2026-07).

| Months | Focus | Ships |
|---|---|---|
| 0–1 | §4 Foundations + WS-A audit + WS-B splits/harness | Repro infra, official splits v1, eval harness, datasheet draft |
| 1–3 | WS-C Tier 0/1 keypoint baselines (3070 dev → cloud) | Baseline leaderboard, first real numbers |
| 2–4 | WS-E alignment (runs alongside; reuses WS-C attentions/CTC) | Word/clip↔frame maps, pseudo-glosses |
| 3–5 | WS-D novel methods (SSL, Mamba, KP→LLM) | Headline method + ablations |
| 4–6 | WS-C Tier 2 video + cross-dataset (cloud) | ViViT/VideoMAE numbers, PHOENIX/CSL-Daily transfer |
| 6–8 | §11 full sweeps across all splits + analysis | Complete result tables w/ CIs |
| 8–10 | §12 paper writing, artifact release, fresh-clone repro | **NeurIPS D&B submission** |
| parallel / post | WS-F avatar/SLP + endpoint | Demo + SLP paper candidate |

Adjust if the target is a nearer venue — the phase *ordering* is fixed; only the calendar compresses.

---

## 4. Foundations (Phase 0 — do before any modeling)

**Goal:** make every later result reproducible, comparable, and loggable. This is the cheapest phase and the one that prevents the most wasted cloud spend.

### 4.1 Repository structure
Reorganize the merged repo into clear responsibilities (keep teammate's "store raw, transform at load" principle intact):
- `bornil/data/` — extraction, feature transforms (`bsl_extract.py`, `bsl_utils.py` move here), packing, dataset classes.
- `bornil/benchmark/` — split generation, metrics, eval harness, leaderboard writer.
- `bornil/models/` — one module per model family (encoders, decoders, heads) with a common interface.
- `bornil/align/` — WS-E alignment utilities.
- `bornil/production/` — WS-F pose generation + avatar + endpoint.
- `bornil/train/` — training loop(s), schedulers, checkpointing (framework choice in §4.4).
- `configs/` — one config per experiment (§4.3).
- `docs/` — this plan + logs (already created).
- `scripts/` — CLIs (extraction, packing, train, eval, sweep).

**Validation:** `import bornil` works; existing pipeline (extract → pack → dataloaders) runs end-to-end on a 20-clip smoke subset after the move, producing identical feature shapes `(T,258)`/`(T,218)` as before.

### 4.2 Environment & determinism
- Pin versions (already known-good: numpy 1.26, pandas 2.3, opencv 4.11, mediapipe 0.10.14, torch 2.x, sentencepiece 0.2.2). Add: HuggingFace `transformers`, `datasets`, `peft`, `sacrebleu`, `sacremoses`, a metrics stack (`chrF` via sacrebleu, `bert-score`), `mamba-ssm` (Tier-1), `torch-geometric` (ST-GCN), and an experiment tracker (§4.5).
- Determinism: fixed seeds, `torch.use_deterministic_algorithms(True)` where feasible, log CUDA/cuDNN/driver, record non-determinism sources (e.g., some CUDA kernels) explicitly.
- Two environment files: `env-3070.yml` (dev) and `env-cloud.yml` (adds mamba-ssm CUDA build, video libs).

**Validation:** a fixed-seed dummy training step yields identical loss on two consecutive runs on the same machine.

### 4.3 Config system
Adopt **Hydra + OmegaConf** (or a single dataclass-based config). Every run is fully specified by one composed config; the resolved config is **hashed** and stored with the run. No hyperparameter lives only in a notebook.

**Validation:** re-running from a stored config reproduces the run's first-epoch metrics within noise.

### 4.4 Training framework
Choose **PyTorch Lightning** (or a thin custom loop) for: mixed precision (bf16/fp16), gradient checkpointing, gradient accumulation (crucial on 8 GB), DDP for cloud multi-GPU, checkpoint/resume. One `LightningModule` interface all models implement, so the benchmark can swap architectures without touching the loop.

**Validation:** the same loop trains a trivial model on 3070 (accumulation on) and on cloud (DDP on) with matching per-step loss for step 0.

### 4.5 Experiment tracking + logging discipline
- **Tracker:** Weights & Biases (or MLflow if offline). Log config hash, git commit, split name, representation, metrics, curves, checkpoints.
- **Human logs (you asked for these):** every run also appends a row to `docs/logs/EXPERIMENTS.md`; every failure/dead-end an entry to `docs/logs/FAILURES.md`; every meaningful change a line to `docs/logs/CHANGELOG.md`. The tracker is for curves; the markdown logs are the narrative the paper is written from.

**Validation:** a run appears in the tracker *and* as an EXPERIMENTS.md row with a resolvable config hash and checkpoint path.

### 4.6 The "overfit-tiny" protocol (reused everywhere)
Before any full/cloud run of any model: train on **8–32 clips** until train loss → ~0 and the model reproduces those targets. If it can't overfit a handful of examples, the architecture/loss/data-plumbing is broken — fix before spending compute.

**Validation:** documented overfit-tiny pass (loss curve to ~0) precedes every EXPERIMENTS.md row for a real run.

**Phase 0 exit criteria:** repo reorganized + smoke test green; deterministic dummy run; config→run→log→checkpoint loop demonstrated end-to-end on the 3070.

---

## 5. WS-A — Data Preprocessing Pipeline (Requirement #4: what we're getting, how)

**Goal:** a documented, multi-representation, quality-audited feature layer that every model and the avatar system consume. Extends the teammate's pipeline rather than replacing it.

### 5.1 What we already have (inventory)
- **Raw:** 17,988 `.webm` clips, front-view, indoor, near-distance, real (non-green) backgrounds; per-clip metadata (`view, start_at, end_at, duration, lighting, camera_distance, is_green_background`), `frame_count`, `chunk_path` (36 chunks). Labels = Bangla sentences in `video_data.csv`.
- **Derived (teammate):** MediaPipe **Holistic** keypoints, uniformly sampled to **≤128 frames**, **258-dim/frame** (pose 33×4 + left hand 21×3 + right hand 21×3), stored **raw** per-clip `.npy`; packed memmap `(N,128,258)` + lengths; **load-time** transform to **218-dim** (shoulder-anchor normalization + upper-body-only pose landmarks 0–22); **SentencePiece BPE** tokenizer (~4k) over the Bangla text.

### 5.2 What we will add (representations we're getting, and how)
Each representation is a first-class, benchmarkable input. All extracted with the **same uniform-sampling and "store raw, transform at load"** discipline.

1. **Keypoints — current (kp-258 → kp-218).** Keep as the primary, cheap representation. *How:* existing `bsl_extract.py` (MediaPipe Holistic, `model_complexity=1`).
2. **Keypoints + face (kp+face).** **Face is currently OFF** (`refine_face_landmarks=False`, face not in the 258). Facial expression carries **grammatical** meaning in sign languages (negation, question, intensity), so this is a real gap. *How:* re-run Holistic with face landmarks (468 face pts → select a compact subset: brows, eyes, mouth, ~40–70 pts) into an extended raw vector; keep face optional at load. **This is both a representation and an ablation** ("does face help BdSL SLT?").
3. **Cropped-hand RGB (hand-patches).** Hands are the highest-information region and MediaPipe often *misses* them. *How:* use hand-landmark bounding boxes to crop fixed-size hand patches per frame; store as small image tensors or pre-extracted CNN features (e.g., a frozen ResNet/EfficientNet). Feeds two-stream and video models.
4. **Full-frame RGB video (rgb-video).** For ViViT/VideoMAE (Tier-2). *How:* decode + resize + uniform-sample frames on the fly (or pre-extract patch/embedding tensors on cloud to cut I/O). Store as pre-extracted backbone features where possible (the research-doc's "pre-extract offline" advice).
5. **Optical flow / motion (optional).** Motion is discriminative for signs. *How:* RAFT or TV-L1 flow on sampled frames, or simply **keypoint velocities/accelerations** (finite differences of kp) — the latter is nearly free and often as useful. Add velocity/accel channels as a load-time transform.

### 5.3 Quality audit (doubles as analysis + QC)
Compute and store per-clip and corpus-level statistics:
- **Detection rates:** % frames with pose / left-hand / right-hand detected (non-zero-filled). Hand-miss rate is the key SL quality signal.
- **Zero-filled fraction** per clip; distribution across chunks (does quality vary by session?).
- **Temporal coverage:** clips whose true frame count ≫ 128 (undersampled) — quantify information loss.
- **Text stats:** length distribution (chars/subwords), vocabulary growth, domain tags (news/sports/conversation/literary — infer via keywords or a classifier), duplicate/near-duplicate sentences.
- **Signer/session proxy:** treat **`chunk` as session** (your decision). Verify whether visual appearance clusters by chunk (embed a few frames per clip with a face/appearance model, cluster, measure chunk purity) to justify chunk-as-session for the signer-independent split.

**Deliverables:** `bornil/data/` refactor; extractors for kp+face, hand-patches, rgb features, velocity channels; a `quality_report.md` + figures; a `manifest_v2.csv` carrying representation paths, detection rates, chunk/session id, domain tag, length bins.

**Validation:**
- Feature-parity: refactored kp extraction reproduces existing `.npy` bitwise (or within float tolerance) on 100 clips.
- Every representation loads to the documented shape; a load-time transform round-trips (raw → transform → expected dim).
- Detection-rate numbers are stable across two runs; chunk-purity clustering reported with a number (e.g., "chunk explains X% of appearance-cluster variance").

**Failure modes → fixes:**
- *MediaPipe misses hands on fast signs* → add temporal interpolation for short gaps; report residual miss-rate; consider higher `model_complexity=2` on a subset to bound the ceiling. Log to FAILURES.
- *Face extraction ~doubles raw size/time* → extract face only for the subset needed for the face ablation, not all 17,988, if compute-bound.
- *rgb pre-extraction I/O bottleneck* → pre-extract backbone features to memmap on cloud once; never decode video in the hot training loop.

---

## 6. WS-B — Dataset Benchmarking (Requirement #2)

**Goal:** turn Bornil into a *citeable, reproducible benchmark*: fixed splits, a locked metric protocol, an eval harness, a leaderboard, and dataset documentation to D&B-track standard.

### 6.1 Task definition (frozen)
Gloss-free **SLT**: given a clip's frame sequence (any representation), output the Bangla sentence. Report translation metrics; **no gloss WER**. Secondary optional signal: character/subword error rate as a diagnostic, clearly labeled as *not* the primary metric.

### 6.2 Official splits (versioned, released as files)
1. **`random-v1`** — stratified random 80/10/10 (comparable to Arib et al. and the teammate's current split; re-seed and freeze).
2. **`signerindep-v1` (chunk-as-session)** — train/val/test with **disjoint chunks**, so test "signers/sessions" are unseen. This is the headline generalization split. (Documented caveat: chunk≈session is a proxy pending true signer IDs; §5.3 clustering justifies it.)
3. **`unseen-sentence-v1`** — guarantee zero sentence overlap between train and test (near-automatic here; formalize and verify). Tests open-vocabulary translation.
4. **Analysis strata (not separate training splits):** by sentence length, by hand-detection quality, by domain, by clip duration — for §11 breakdowns.

**Leakage checks:** verify no `recording` UUID appears in two splits; no chunk crosses the signer-independent boundary; near-duplicate sentences don't straddle train/test in `unseen-sentence`.

### 6.3 Metric protocol (locked, scriptable)
- **SacreBLEU** BLEU-1..4 with a **fixed, declared tokenizer** for Bangla (record the exact `--tokenize` setting; Bangla needs a consistent Unicode-normalized, script-aware tokenization — declare it so numbers are comparable across papers).
- **chrF and chrF++** (character-level; **critical** for Bangla's rich morphology and the subword regime).
- **ROUGE-L**, **BERTScore** with **BanglaBERT**.
- **Bootstrap 95% CIs** on BLEU-4/chrF (resample test set); report CIs everywhere. Add **paired significance tests** when comparing two models on the same split.
- **Unicode normalization** (NFC) applied identically to hypotheses/references before scoring — a silent source of BdSL/Bangla metric noise.

### 6.4 Eval harness + trivial baselines
- One `evaluate(predictions, split)` entry point that emits all metrics + CIs + a per-example CSV (for error analysis) from a **frozen** reference set.
- **Trivial baselines to validate the harness** (must run *before* any real model): (a) most-frequent-sentence, (b) length-matched random sentence, (c) **nearest-neighbor retrieval** (retrieve train sentence whose video embedding is closest — a surprisingly strong, informative floor). Real models must beat these.

### 6.5 Artifacts (part of the contribution)
- **Datasheet for Datasets** (Gebru et al.) — provenance, collection, consent, demographics (as known), intended use, limitations.
- **Croissant** metadata (NeurIPS D&B expects machine-readable dataset metadata).
- **Split files** (UUID lists per split/version), the **locked eval script**, a **leaderboard** (`docs/logs/EXPERIMENTS.md` table + a rendered page), model **checkpoints**, and a **reproduction README**.

**Validation:**
- Metric protocol reproduces identical scores on a fixed (hypothesis, reference) fixture across machines (this is the anti-"unreproducible BLEU" guarantee).
- Trivial baselines produce sane, low scores; retrieval floor is documented.
- Split files pass all leakage checks programmatically.
- Datasheet + Croissant validate against their schemas.

**Failure modes → fixes:**
- *BLEU varies by environment* → the locked tokenizer + SacreBLEU signature string eliminates it; store the signature with every score.
- *chunk-as-session too coarse/fine* → report how many chunks/sessions and clips-per-chunk; if a chunk mixes signers, note it as a limitation and/or refine with the appearance-clustering from §5.3.
- *Retrieval floor beats weak models* → that's a finding, not a bug — it means those models don't learn beyond dataset priors; report it.

---

## 7. WS-C — SLT Model Suite (Requirement #1: established / latest / transformer / video + eval)

**Goal:** a broad, apples-to-apples suite of translation models, all trained/evaluated through the §4 loop and §6 harness, on all §6.2 splits. Tiered by compute so the 3070 does maximal work before cloud.

**Common interface.** Every model = **encoder** (over a representation) + **decoder/head** producing Bangla subwords. Two decoding families: **autoregressive attention decoder** (BLEU-oriented, fluent) and **CTC head** (alignment-friendly, feeds WS-E). Where sensible, train **joint CTC + attention** (hybrid loss, as in speech translation) — CTC regularizes and yields alignments for free.

### 7.1 Tier 0 — keypoint baselines (3070-prototypable)
- **T0.1 BiLSTM encoder + attention decoder** — the classic SLT baseline; the teammate's DataLoaders drop straight in. Also a **BiLSTM + CTC** variant.
- **T0.2 Transformer encoder–decoder** ("Sign Language Transformers", Camgöz 2020, gloss-free variant) over kp-218. Sinusoidal/learned temporal positions; subword targets.

### 7.2 Tier 1 — keypoint, stronger/graph/latest (mostly 3070, some cloud)
- **T1.1 ST-GCN / spatiotemporal graph encoder** over the hand+body skeleton (PyTorch Geometric or a standard ST-GCN impl) + attention decoder. **Directly subsumes/compares Arib et al.'s ST-GCN-LSTM+Transformer hybrid** — we reimplement their family so our comparison is honest.
- **T1.2 Two-stream** (keypoints ⊕ cropped-hand-RGB features from §5.2) with cross-stream fusion — tests whether appearance rescues MediaPipe hand misses.
- **T1.3 Conformer encoder** (convolution-augmented transformer; strong on temporal signals) + attention/CTC — a "latest established" point.

### 7.3 Tier 2 — video backbones (cloud)
- **T2.1 ViViT** and **T2.2 VideoMAE-v2** as clip encoders (HuggingFace), + a Bangla text decoder. Use **pre-extracted features** (§5.2.4) to keep training tractable; finetune with gradient checkpointing + bf16 on A100.
- **T2.3 (optional) I3D/SlowFast** feature encoder + decoder as a CNN-video reference.

### 7.4 Cross-dataset (you allowed this)
Reimplement the data adapter for **PHOENIX-2014T** and **CSL-Daily** (keypoint versions via MediaPipe on their frames, or published keypoints) so the *same* models run there. Purpose: show our methods generalize and situate Bornil numbers against known scales. (How2Sign optional — larger, ASL.)

**Training + evaluation (every model):**
1. Overfit-tiny on 3070 (§4.6). 2. Small-subset dev run (curves sane, val > trivial). 3. Full train (cloud if Tier ≥1.5). 4. Evaluate through §6 harness on **all** splits. 5. Log EXPERIMENTS row + checkpoint + leaderboard update. 6. Error-analysis CSV.

**Validation (per model):**
- Beats the retrieval/trivial floor on `random-v1` with non-overlapping CI.
- Reports numbers on all three official splits; the `signerindep-v1` drop vs `random-v1` is measured (generalization gap is itself a result).
- Reproduces within CI on a re-run from stored config.

**Failure modes → fixes:**
- *Model collapses to frequent n-grams / low chrF* → check label smoothing, decoder LR, subword vocab size; try CTC-aux; inspect attention (WS-E) for degenerate alignment. Log FAIL entry.
- *Transformer overfits tiny data / underfits long clips* → stronger dropout/aug (§11 aug), longer warmup; for long clips revisit 128-frame cap (windowing).
- *Video models OOM on 3070* → they're cloud-only by design; if cloud-limited, use frozen pre-extracted features + a light decoder.
- *ST-GCN reimpl doesn't match Arib et al.* → treat their reported numbers as external reference; document our config; our claim is relative-to-our-suite, cross-checked against their paper.

---

## 8. WS-D — Novel Approach (Requirement #3)

**Goal:** the headline methodological contribution. We **build all three candidate directions**, compare them head-to-head in the suite, and the **best-performing becomes the paper's flagship**; the comparison itself is a contribution. All are keypoint-first (3070-prototypable), cloud for full training.

### 8.1 D1 — Self-supervised keypoint pretraining (primary novelty bet)
**Idea:** exploit the fact that **you own Bornil** (and its unlabeled video beyond the 17,988) with **masked / contrastive keypoint pretraining**, then finetune for SLT. Directions:
- **Masked Keypoint Modeling (MKM):** randomly mask spatial (joints) and temporal (frames) spans; reconstruct — a MAE for skeletons.
- **Motion-contrastive:** contrast temporally-augmented views (speed perturb, jitter within physiological range) → representation invariant to signer/tempo (InfoNCE).
- **Finetune:** pretrained encoder + attention/CTC decoder.
**Why it can win / why it's novel:** first SSL-for-keypoints on BdSL; a **low-resource narrative** (labels are scarce, motion is abundant); cleanly differentiates from Arib et al. (supervised only).

### 8.2 D2 — Mamba/SSM temporal SLT
**Idea:** replace the temporal Transformer with a **selective state-space model** (`mamba-ssm`), linear-time in sequence length → can consume **full-length clips** (not just 128 frames), directly attacking the undersampling loss quantified in §5.3.
**Why it can win / why it's novel:** SSMs are barely explored for SL; the **long-form efficiency** angle is a clean story and lets us test whether the 128-cap was hurting.

### 8.3 D3 — Keypoint → Bangla-LLM decoder
**Idea:** keypoint encoder emits **soft-prompt tokens** into a pretrained **Bangla seq2seq/LLM** (BanglaT5 / mBART / a Bangla-capable LLM) adapted with **LoRA/QLoRA** (PEFT). The language prior yields fluent, grammatical Bangla.
**Why it can win / why it's novel:** brings strong Bangla language priors to gloss-free SLT; VLM-style framing; likely best *fluency*/BLEU if visual encoder is decent. Dev via LoRA on a small model (3070); full training on cloud.

### 8.4 D4 — Best-of fusion (only if warranted)
If D1–D3 are complementary (e.g., SSL encoder + LLM decoder + Mamba temporal), combine the winning pieces. **YAGNI guard:** only pursue if ablations show additive gains; don't build the union speculatively.

**Validation:**
- **D1:** pretrained-then-finetuned beats from-scratch identical architecture on `signerindep-v1` with non-overlapping CI; report data-efficiency curve (BLEU vs. #labeled clips) — the low-resource money plot.
- **D2:** full-length Mamba beats the 128-frame Transformer *specifically on long clips* (length-stratified analysis).
- **D3:** higher chrF/BLEU and human-rated fluency than non-LLM decoders at matched encoder.
- **All:** every gain survives bootstrap CI + paired test; every negative result logged to FAILURES (negative results are paper material).

**Failure modes → fixes:**
- *SSL pretraining doesn't transfer* → vary masking ratio/objective; try motion-contrastive vs MKM; check the encoder isn't too small; if still flat, report as an honest negative (valuable for the community) and lean on D2/D3.
- *Mamba unstable / no CUDA build on 3070* → prototype on cloud; use a smaller SSM; compare against a linear-attention Transformer as a fallback "efficient long-seq" baseline.
- *LLM decoder hallucinates fluent-but-wrong Bangla* → higher visual conditioning weight, CTC-aux grounding, retrieval-augmented decoding; report adequacy vs fluency separately.

---

## 9. WS-E — Sentence ↔ Video Mapping / Alignment (Requirement #5)

**Goal:** from **sentence-only** labels, induce **which video segments correspond to which words/units** — at two granularities: **word/unit-level** and **frame/clip-level**. This yields interpretability for the benchmark **and** the raw material (word↔sign-clip inventory) for WS-F avatar production.

### 9.1 Why this is non-trivial here
No glosses, no timings, ~100% unique sentences → alignment must be **learned weakly/unsupervised**. We attack it from multiple angles and cross-validate them against each other (agreement between independent methods = confidence, since we lack ground-truth alignment).

### 9.2 Approaches (multiple, as you asked)
1. **CTC forced alignment (frame↔subword).** A CTC-trained encoder (from WS-C) gives per-frame subword posteriors; CTC forced alignment assigns frame spans to subwords → **frame-level** map. Cheapest, comes free with CTC models.
2. **Attention alignment (word↔frame).** Decoder cross-attention in attention models indicates which frames each output token attends to; aggregate to **word/subword ↔ frame** heatmaps. Cross-check vs CTC.
3. **Contrastive word↔clip correspondence.** A dual encoder (sign-segment encoder ↔ Bangla word/phrase encoder, InfoNCE) learns a shared space; retrieval gives **word ↔ clip-segment** matches. Enables a **sign lexicon** (word → representative clips).
4. **Unsupervised temporal segmentation.** Changepoint detection on motion (keypoint velocity minima ≈ sign boundaries), slot-attention, or clustering of sub-sequences → segment the clip into **candidate sign units** without labels; then label units via (3).
5. **Pseudo-gloss induction.** Combine (1)/(3)/(4) to assign a discrete **pseudo-gloss** id per segment → a gloss-like intermediate the community lacks for Bornil. Feeds both analysis and a two-stage SLT variant (video→pseudo-gloss→text).

### 9.3 Two granularities, explicitly
- **Word / unit level:** "sentence → ordered list of sign units, each a (word/phrase, clip-segment) pair." Built from (3)+(4)+(5).
- **Frame / clip level:** "each frame → subword/word responsibility." Built from (1)+(2).

**Deliverables:** alignment tools in `bornil/align/`; a **sign lexicon** (word → clips/pose-segments); pseudo-gloss annotations for the corpus; alignment-quality figures; a two-stage SLT baseline using pseudo-glosses (feeds WS-C comparison).

**Validation (no gold alignment, so triangulate):**
- **Cross-method agreement:** CTC vs attention alignment agree above chance (report an agreement metric); segmentation boundaries coincide with motion changepoints.
- **Human spot-check:** native BdSL signer verifies word↔clip matches on a small sample (report precision@k for the lexicon).
- **Downstream utility:** pseudo-gloss two-stage SLT is competitive with end-to-end on some split (if it helps, that's a result; if not, documented).
- **Sanity:** aligned spans respect monotonic-ish ordering for a mostly-monotonic language pair; flag heavy reordering as a linguistic finding (BdSL vs Bangla word order).

**Failure modes → fixes:**
- *Attention is diffuse/uninterpretable* → use CTC alignment as primary; add attention-guidance loss; try Gaussian/monotonic attention.
- *Segmentation over-/under-segments* → tune changepoint sensitivity against average signs-per-second priors; validate against clip duration vs sentence length.
- *Lexicon noisy for rare words* → restrict avatar use (WS-F) to high-confidence, frequent units; fall back to generative pose for the rest.

---

## 10. WS-F — Avatar-Based Animation + Text→Sign Endpoint (Requirement #6)

**Goal:** given an arbitrary **Bangla sentence**, generate a **BdSL animation on an avatar**, served behind a **model endpoint**. This is **Sign Language Production (SLP)** — the reverse of WS-C — and it reuses WS-E's word↔clip/pose inventory and the pose data from WS-A.

**Scope note:** this is the extended-program headline (great demo + candidate SLP paper). It is **not gating** for the NeurIPS D&B submission, but it's planned here because it shares the data/alignment foundation and you want the proposal now.

### 10.1 Two production strategies (build the simpler first)
1. **Retrieval + concatenative synthesis (baseline, ship first).**
   - Text → sign units via WS-E sign lexicon (word/phrase → representative **pose segment**).
   - Concatenate pose segments; **smooth transitions** (motion blending / DTW-based time alignment / inbetweening) to avoid teleporting joints.
   - Render pose sequence on an avatar (§10.2).
   - *Pros:* faithful, uses real signing; *Cons:* limited to lexicon coverage, blending artifacts, no co-articulation.
2. **Generative pose synthesis (advanced).**
   - **Text → pose sequence** directly: **Progressive Transformers** (Saunders et al., seq2seq text→pose) or a **motion-diffusion model** (MDM-style, diffusion over pose sequences conditioned on Bangla text embeddings).
   - Produces continuous, co-articulated signing; generalizes beyond the lexicon.
   - *Cons:* harder, needs the pose data at scale (we have it from WS-A), evaluation is subtle.

### 10.2 Avatar rendering
- **Skeleton/pose target:** MediaPipe/3D keypoints → drive a rig. Options: **SMPL-X** body+hands+face parametric model (rich hands/face, research-standard) or a lighter **2D/3D stick-to-mesh** avatar; render in **Blender** (offline, high quality) or **Three.js/Unity** (interactive, web-friendly for the endpoint/demo).
- Recommendation: prototype with a lightweight web avatar (Three.js) for the endpoint demo; use SMPL-X + Blender for paper-quality figures/videos.

### 10.3 The endpoint
- **Serve:** a **FastAPI** service: `POST /sign {bangla_sentence}` → returns a **pose sequence** (JSON/npz) and/or a **rendered video/animation** (server-side render or client-side Three.js playback).
- **Pipeline behind it:** sentence → (retrieval or generative) pose → smoothing → render. Cache frequent sentences/units.
- **Packaging:** containerize; document latency; expose a minimal web demo page.

**Validation:**
- **Back-translation:** run a WS-C SLT model on the **generated** avatar video/pose → measure BLEU/chrF against the input sentence (the standard SLP metric). Higher = more intelligible production.
- **Pose fidelity:** DTW distance / **MPJPE** between generated and held-out real pose for sentences that exist in data.
- **Realism:** **FVD** (Fréchet Video Distance) for rendered video (generative track).
- **Human study:** native BdSL signers rate **intelligibility + naturalness** (Likert), plus comprehension accuracy (can they read it back?). This is the metric that actually matters.
- **Endpoint:** returns valid pose/animation for held-out sentences within a stated latency budget; graceful fallback when lexicon coverage is low.

**Failure modes → fixes:**
- *Concatenation looks robotic* → stronger blending, velocity-continuity constraints, learned inbetweening; move toward generative for smoothness.
- *Generative pose is mushy/averaged* → diffusion (multimodal) over regression; condition on stronger Bangla text embeddings (BanglaBERT/BanglaT5); add hand-shape-specific losses (hands are where meaning lives).
- *Lexicon gaps for open vocabulary* → hybrid: retrieve where confident, generate elsewhere; report coverage.
- *Avatar hands unreadable* → SMPL-X hands or a hand-focused rig; hands are non-negotiable for legibility.

---

## 11. Analysis & Ablations (feeds the paper's story)

Run across the frozen splits, all with CIs:
- **Representation ablation:** kp-218 vs kp-258 vs kp+face vs two-stream vs video — *what does face/appearance/motion buy?*
- **Normalization ablation:** shoulder-anchor on/off, upper-body select on/off (cheap thanks to "transform at load") — validates the teammate's design choices with numbers.
- **Temporal budget:** 64 vs 128 vs full-length (Mamba) — quantify the undersampling cost from §5.3.
- **Decoder family:** CTC vs attention vs joint vs LLM-decoder.
- **Generalization gap:** `random` vs `signerindep` vs `unseen-sentence` for every model — the benchmark's core message.
- **Data efficiency:** BLEU vs #labeled clips (the SSL money plot from §8).
- **Cross-dataset transfer:** train Bornil → test/adapt PHOENIX/CSL-Daily and vice-versa (does BdSL share structure with DGS/CSL?).
- **Difficulty factors:** correlate BLEU with hand-detection rate, clip length, sentence length, domain — *what makes a clip hard?*
- **Error taxonomy:** manual categorization of failure types from the per-example CSVs.

**Validation:** every ablation cell is an EXPERIMENTS row with CI; conclusions are only drawn where CIs separate; ablations that *don't* move the needle are reported (prevents overclaiming).

---

## 12. Paper & Artifacts (NeurIPS D&B) + Reproducibility

- **Paper sections:** intro (BdSL gap + open-vocabulary framing), dataset (datasheet summary), benchmark (splits/metrics/protocol), baseline suite (WS-C), novel method (winning WS-D), analysis (§11), alignment interpretability (WS-E, condensed), limitations (signer-proxy, absolute-BLEU, ethics), broader impact (Deaf accessibility). WS-F as a demo/appendix or forward-looking section.
- **Artifacts:** dataset access + datasheet + Croissant, split files, locked eval script, code, checkpoints, leaderboard, reproduction README, demo (WS-F).
- **Reproducibility gate (hard):** a **fresh clone**, following the README, reproduces the **headline number within CI** on the released split using a released checkpoint *and* by retraining from config. Nothing ships that fails this.
- **Ethics/consent:** confirm signer consent for release; document demographics as known; involve native BdSL signers in evaluation (WS-E spot-checks, WS-F human study).

**Validation:** independent team member (or fresh environment) runs the repro gate green before submission; all paper numbers trace to EXPERIMENTS rows.

---

## 13. Risk register (top risks, owner-visible)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Signer IDs unrecoverable → weak signer split | Med | High | chunk-as-session proxy + appearance-clustering justification (§5.3); document as limitation |
| Absolute BLEU low → "looks weak" to reviewers | High | Med | frame as open-vocab benchmark + relative gains + CIs; retrieval floor context |
| Arib et al. seen as prior benchmark | Med | Med | position as *official* dataset paper; reimplement their family; broader suite + novel method |
| Cloud budget overrun | Med | High | overfit-tiny gate; pre-extract features; 3070 does all dev; sweep only promising configs |
| SSL novelty doesn't pan out | Med | Med | three novel bets (D1–D3); negative results are publishable; best-of wins |
| Face/video extraction cost | Med | Low | extract on subsets for ablation; pre-extract once |
| WS-F scope creep pulls focus from D&B | Med | High | WS-F explicitly non-gating; sequence after submission |
| MediaPipe hand-miss ceiling | Med | Med | two-stream RGB rescue; interpolation; report residual |

---

## 14. Glossary

- **SLT** — Sign Language Translation (video → spoken/written-language sentence).
- **CSLR** — Continuous Sign Language Recognition (video → gloss sequence). *Not our primary task* (no glosses in Bornil).
- **SLP** — Sign Language Production (text → sign). WS-F.
- **Gloss** — a written word-label for a single sign; Bornil has none.
- **Keypoints** — MediaPipe pose/hand/(face) landmark coordinates per frame.
- **kp-258 / kp-218** — raw vs upper-body-normalized per-frame feature widths.
- **CTC** — Connectionist Temporal Classification (alignment-free seq loss; yields frame alignments).
- **SSM / Mamba** — selective state-space sequence model (linear-time long sequences).
- **chrF/chrF++** — character-n-gram F-score MT metrics (robust for morphologically rich Bangla).
- **Datasheet / Croissant** — dataset documentation standard / machine-readable dataset metadata (D&B track expectations).
- **Overfit-tiny** — sanity protocol: overfit a handful of examples before any real/cloud run.

---

## 15. Immediate next actions (Phase 0 kickoff)

1. Reorganize repo into `bornil/` packages (§4.1); green the smoke test.
2. Stand up config + tracker + training-loop skeleton (§4.3–4.5); demonstrate the config→run→log→checkpoint loop on the 3070.
3. Generate `manifest_v2.csv` + quality report (§5.3) and freeze **official splits v1** (§6.2).
4. Lock the metric protocol + eval harness + trivial baselines (§6.3–6.4).
5. First real model: T0.2 Transformer enc-dec — overfit-tiny → dev run → first leaderboard row.

> After Phase 0, WS-C and WS-E proceed in parallel; WS-D follows once a solid attention/CTC baseline exists; WS-B artifacts finalize alongside §11; WS-F starts after the D&B core is stable.
