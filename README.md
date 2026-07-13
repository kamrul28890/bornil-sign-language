# BSL — Bangla Sign Language → Text

A keypoint-based pipeline for translating **Bangla Sign Language (BSL)** videos
into Bangla text. Sign clips are converted into MediaPipe pose/hand keypoint
sequences, paired with sentence labels, and turned into model-ready batches for
a sequence-to-sequence recognition/translation model.

> **Status:** the full data pipeline (extraction → features → tokenized targets →
> batches) is complete and verified on all **17,988** clips. The recognition
> model is the next step.

---

## Dataset

- **17,988** short `.webm` clips — each a single signer performing **one Bangla
  sentence** (front view, mostly indoor).
- **17,930** unique sentences, open-domain (news, sports, everyday speech).
- Labels and metadata live in [`video_data.csv`](video_data.csv):

  | column | description |
  |---|---|
  | `text` | Bangla sentence (the label) |
  | `recording` | clip filename, `<uuid>.webm` |
  | `metadata` | JSON: view, duration, lighting, camera distance, … |
  | `frame_count` | number of frames in the clip |
  | `chunk_path` | folder holding the clip (`chunks/chunk_k`) |

The label↔video link is **clip-level**: one clip ↔ one sentence, joined by the
UUID. There is no frame-to-word alignment — the model is expected to learn it
(via CTC or attention).

> ⚠️ The videos (~180 GB) and the generated feature files are **not** in this
> repo — only the pipeline code and the trained tokenizer are versioned. Point
> the path constants at your local copy of the dataset.

---

## Pipeline

Two phases, built around one principle: **store raw features, transform at
load** — so normalization and landmark choices can change without re-running
the expensive MediaPipe pass.

```
Phase A — preprocessing   (bsl_preprocessing_clean.ipynb + bsl_extract.py)
  video_data.csv + chunks/*.webm
        │   MediaPipe Holistic; 128 frames uniformly sampled across each clip
        ▼
  features_mediapipe_v2/<uuid>.npy        raw (T≤128, 258) float32
        +  processed_manifest.csv         (text ↔ feature, per clip)

Phase B — modeling data   (bsl_modeling.ipynb + bsl_utils.py)
  packed memmap (one file, via pack_features.py)
        │   prepare_features: normalize (shoulder anchor) + upper-body select
        ▼   (T, 218)                      SentencePiece BPE targets (~4k)
  DataLoaders  →  (B, 128, 218) inputs + subword-id targets  →  [model — TODO]
```

### Feature representation

Each frame is **258** raw values from MediaPipe Holistic:

| part | landmarks × dims | size |
|---|---|---|
| pose | 33 × (x, y, z, visibility) | 132 |
| left hand | 21 × (x, y, z) | 63 |
| right hand | 21 × (x, y, z) | 63 |

At **load time** (`bsl_utils.prepare_features`) these become the **218-dim**
model input:

- **normalized** — each frame is anchored to the shoulder midpoint and scaled by
  shoulder width, making it invariant to where the signer stands and how large
  they appear;
- **upper-body only** — pose landmarks 23–32 (hips/legs/feet) are dropped; they
  have ≤ 0.08 mean visibility and sit off-frame (extrapolated noise).

The stored `.npy` files stay raw (258); the 258 → 218 selection and
normalization happen at load, so they can be changed without re-extraction.

### Targets

Bangla text is tokenized with a **SentencePiece BPE** model
([`bsl_sp.model`](bsl_sp.model), ~4000 subwords). Word-level targets are
unlearnable here — 25.5k unique words, 58% appearing only once — so frequent,
reusable subword units are used instead.

---

## Repository

| file | role |
|---|---|
| [`bsl_preprocessing_clean.ipynb`](bsl_preprocessing_clean.ipynb) | Phase A — videos → keypoint features + manifest |
| [`bsl_extract.py`](bsl_extract.py) | MediaPipe extraction (uniform 128-frame sampling, 258-dim) — shared source |
| [`run_extraction.py`](run_extraction.py) | parallel, resumable full-dataset extraction |
| [`pack_features.py`](pack_features.py) | consolidate per-clip features into one memmap file |
| [`bsl_utils.py`](bsl_utils.py) | load-time normalization + upper-body landmark selection |
| [`bsl_modeling.ipynb`](bsl_modeling.ipynb) | Phase B — packed features + BPE tokenizer + DataLoaders |
| [`bsl_sp.model`](bsl_sp.model) / [`bsl_sp.vocab`](bsl_sp.vocab) | trained SentencePiece tokenizer |
| [`video_data.csv`](video_data.csv) | master index of clips + labels |

---

## Setup

Python 3.10. Install dependencies:

```bash
pip install numpy pandas opencv-python mediapipe torch sentencepiece tqdm
```

Tested with: numpy 1.26, pandas 2.3, opencv 4.11, mediapipe 0.10.14,
torch 2.11, sentencepiece 0.2.2.

---

## Usage

Set the dataset and output paths in the config cell at the top of each notebook
(and the matching constants in the scripts), then:

1. **Extract features** — run `bsl_preprocessing_clean.ipynb` top to bottom, or
   process the whole dataset in parallel:
   ```bash
   python run_extraction.py           # resumable — skips clips already done
   ```
   Produces per-clip `.npy` features and `processed_manifest.csv`.

2. **Pack features** — consolidate the 17,988 files into one memmap file for
   fast, low-overhead loading during training:
   ```bash
   python pack_features.py            # adds a pack_idx column to the manifest
   ```

3. **Build batches / train** — run `bsl_modeling.ipynb`: it loads the packed
   features, tokenizes text with SentencePiece, splits 80/10/10, and exposes
   `train_loader` / `val_loader` / `test_loader` yielding `(B, 128, 218)` inputs
   with subword-id targets. `INPUT_DIM = 218`, `VOCAB_SIZE = 4000`.

---

## Design notes

- **Store raw, transform at load** — on-disk features are raw 258-dim; `bsl_utils`
  applies normalization + landmark selection at load time, so the scheme can
  change without a ~45-hour re-extraction.
- **Uniform temporal sampling** — clips longer than 128 frames are sampled evenly
  across the whole clip, not truncated to the first few seconds.
- **Single source of truth** — extraction lives only in `bsl_extract.py`,
  transforms only in `bsl_utils.py`; the model reads its input width from
  `feature_dim()`, never a hard-coded literal.
- **Weakly-aligned targets** — one keypoint sequence ↔ one sentence with no
  per-frame labels, which suits a CTC or attention sequence model.

---

## Roadmap

- [x] Keypoint extraction over the full dataset
- [x] Shoulder-anchored normalization + upper-body landmark selection
- [x] SentencePiece subword tokenizer
- [x] Packed features + DataLoaders
- [ ] Baseline model (BiLSTM / Transformer encoder + CTC or attention decoder)
- [ ] Training / evaluation loop with metrics (WER / BLEU)
