# Bornil — Bangla Sign Language → Text

A gloss-free, keypoint-based research pipeline for translating **Bangla Sign
Language (BdSL)** video into Bangla text. Sign clips are converted into
MediaPipe pose/hand keypoint sequences, paired with open-vocabulary sentence
labels, and used to train and benchmark sequence-to-sequence translation
models — with an accompanying frozen benchmark, baseline suite, novel-method
program, and alignment/interpretability workstream aimed at a NeurIPS
Datasets & Benchmarks submission.

> **Status:** the data pipeline, frozen benchmark, and baseline model suite
> (WS-A/B/C) are complete. The current best model — TConv + joint CTC/attention
> — significantly beats the vanilla baseline and clears the retrieval floor on
> every official split and metric. A novel-method comparison (WS-D) and an
> alignment/interpretability workstream (WS-E) are in progress. Full plan:
> [`docs/plan/2026-07-13-bornil-implementation-plan.md`](docs/plan/2026-07-13-bornil-implementation-plan.md).

---

## Dataset

- **17,988** short `.webm` clips — each a single signer performing **one Bangla
  sentence** (front view, mostly indoor).
- **17,928** unique normalized sentences, open-domain (news, sports, everyday
  speech) — near-total uniqueness confirms this is an open-vocabulary
  translation task, not closed-set gloss recognition.
- Labels and metadata live in [`video_data.csv`](video_data.csv):

  | column | description |
  |---|---|
  | `text` | Bangla sentence (the label) |
  | `recording` | clip filename, `<uuid>.webm` |
  | `metadata` | JSON: view, duration, lighting, camera distance, … |
  | `frame_count` | number of frames in the clip |
  | `chunk_path` | folder holding the clip (`chunks/chunk_k`) |

The label↔video link is **clip-level**: one clip ↔ one sentence, joined by the
UUID. There is no frame-to-word alignment — models learn it (via CTC or
attention), and inducing it after the fact is its own workstream (WS-E, see
below).

> ⚠️ The raw videos (~180 GB), extracted features, and every model's trained
> outputs are **not** in this repo — only code, the trained tokenizer, and the
> frozen split files are versioned. See
> [`docs/DATA_AND_OUTPUTS.md`](docs/DATA_AND_OUTPUTS.md) for what's excluded,
> why, and where to find results/examples without needing the heavy files.

---

## Benchmark & results

Three frozen, leakage-checked splits (`splits/`) test different
generalization axes:

| Split | What it holds out |
|---|---|
| `random-v1` | nothing extra — random 80/10/10 |
| `signerindep-v1` | whole signer sessions (`chunk_path`) |
| `unseen-sentence-v1` | whole normalized-sentence groups (zero text overlap train/test) |

Metrics (`bornil/benchmark/metrics.py`) are locked: SacreBLEU BLEU-1–4, chrF,
chrF++, ROUGE-L, bootstrap CIs, optional BanglaBERT BERTScore. **chrF++ is the
discriminating metric** here — BLEU-4 sits at the noise floor for this
open-vocabulary benchmark.

**Headline result:** TConv2k5 + joint CTC/attention (best model,
chrF++ 10.913 on `random-v1`) significantly beats both the vanilla Transformer
baseline and clears the retrieval floor on all three splits, every metric. The
strongest grounding evidence is an input ablation: deranging only the
video↔text pairing costs −4.71 chrF++ with the input distribution held exactly
constant — but the grounding is measurably **static-pose, not motion**
(shuffling frame order costs only −0.88 chrF++), which is the main open
weakness the later workstreams (WS-D/WS-E) probe.

Full tables, significance tests, and narrative:
[`docs/logs/RESULTS_OVERVIEW.md`](docs/logs/RESULTS_OVERVIEW.md). Every run
ever trained, with metrics and a checkpoint pointer:
[`docs/logs/EXPERIMENTS.md`](docs/logs/EXPERIMENTS.md). Real sample
predictions (exact matches, typical cases, failure modes):
[`docs/logs/QUALITATIVE_EXAMPLES.md`](docs/logs/QUALITATIVE_EXAMPLES.md).
Dead ends and negative results, with root-cause analysis (they're publishable,
not hidden): [`docs/logs/FAILURES.md`](docs/logs/FAILURES.md).

---

## Pipeline

Two phases, built around one principle: **store raw features, transform at
load** — so normalization and landmark choices can change without re-running
the expensive MediaPipe pass.

```
Phase A — extraction   (bsl_extract.py, run via scripts/extract_sample.py or run_extraction.py)
  video_data.csv + chunks/*.webm
        │   MediaPipe Holistic, up to 512 frames/clip
        ▼
  per-clip raw (T≤512, 258) float32 features + a manifest

Phase B — packing + modeling data   (scripts/pack_sample.py, bornil/data/)
  packed memmap (one file, low-overhead loading)
        │   bornil.data.transforms.prepare_features:
        │   shoulder-anchored normalize + upper-body select
        ▼   (T, 218)                      SentencePiece BPE targets (~4k)
  bornil.data.dataset.SignDataset  →  (B, T, 218) inputs + subword-id targets
        │   bornil.benchmark.splits (frozen random-v1 / signerindep-v1 / unseen-sentence-v1)
        ▼
  scripts/train_sample.py  →  trained model + metrics + predictions
```

### Feature representation

Each frame is **258** raw values from MediaPipe Holistic:

| part | landmarks × dims | size |
|---|---|---|
| pose | 33 × (x, y, z, visibility) | 132 |
| left hand | 21 × (x, y, z) | 63 |
| right hand | 21 × (x, y, z) | 63 |

At **load time** (`bornil.data.transforms.prepare_features`) these become the
**218-dim** model input:

- **normalized** — each frame is anchored to the shoulder midpoint and scaled
  by shoulder width, making it invariant to where the signer stands and how
  large they appear;
- **upper-body only** — pose landmarks 23–32 (hips/legs/feet) are dropped;
  they have ≤ 0.08 mean visibility and sit off-frame (extrapolated noise).

The stored features stay raw (258); the 258 → 218 selection and normalization
happen at load, so the scheme can change without re-extraction.

### Targets

Bangla text is tokenized with a **SentencePiece BPE** model
([`bsl_sp.model`](bsl_sp.model), ~4000 subwords). Word-level targets are
unlearnable here — 25.5k unique words, 58% appearing only once — so frequent,
reusable subword units are used instead.

---

## Repository

| path | role |
|---|---|
| [`bornil/`](bornil/) | the project package — `data/` (dataset, transforms, tokenizer, splits), `models/` (SLT architectures), `train/` (training loop, checkpoints), `benchmark/` (locked splits + metrics), `align/` (WS-E forced alignment + motion agreement), `pretrain/` (WS-D self-supervised pretraining) |
| [`scripts/`](scripts/) | CLI entry points: sampling/extraction/packing, training (`train_sample.py`), evaluation and ablations, significance testing, results reporting |
| [`configs/`](configs/) | one YAML per experiment (official baselines, ablations, novel-method runs) |
| [`splits/`](splits/) | frozen split files (`random-v1`, `signerindep-v1`, `unseen-sentence-v1`) — the released, reproducible benchmark partition |
| [`tests/`](tests/) | full test suite (pytest) — mirrors `bornil/`'s structure |
| [`docs/plan/`](docs/plan/) | the master research plan plus one detailed, checkbox-tracked implementation plan per workstream |
| [`docs/logs/`](docs/logs/) | `RESULTS_OVERVIEW.md` (report-facing summary), `EXPERIMENTS.md` (every run), `CHANGELOG.md` (what changed and why), `FAILURES.md` (negative results, logged as data) |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | design specs (one per novel-method sub-project, written before implementation) |
| [`docs/handoff/`](docs/handoff/) | self-contained prompts for handing a plan off to an autonomous coding agent |
| [`bsl_extract.py`](bsl_extract.py) / [`bsl_utils.py`](bsl_utils.py) | shared MediaPipe extraction routine and a backward-compatibility re-export shim (canonical logic now lives in `bornil.data.transforms`) |
| [`bsl_sp.model`](bsl_sp.model) / [`bsl_sp.vocab`](bsl_sp.vocab) | trained SentencePiece tokenizer |
| [`video_data.csv`](video_data.csv) | master index of clips + labels |

`bsl_preprocessing_clean.ipynb`, `bsl_modeling.ipynb`, `run_extraction.py`,
and `pack_features.py` are the original full-dataset extraction/packing
scripts from before the `bornil/` package existed; the full dataset has
already been extracted and packed using them, and they're kept for
provenance. New work uses the `scripts/*.py` + `bornil/` path above.

---

## Setup

Python 3.10 (Windows dev environment; see `requirements.txt` for exact
pins and the reasoning behind them — this project has hit real
version-compatibility failures, documented in
[`docs/logs/FAILURES.md`](docs/logs/FAILURES.md)).

```bash
pip install -r requirements.txt
# torch: install a build matching your hardware, separately —
#   GPU (NVIDIA, recommended): pip install torch --index-url https://download.pytorch.org/whl/cu124
#   CPU only:                   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Tested with torch 2.6.0+cu124 (RTX 3070) and 2.11.0+cpu. On Windows, `import
torch` must happen before `import pandas` in any entry point (see
`bornil/runtime.py`) — a real DLL-load-order crash this project hit early on.

Run the full test suite: `pytest` (or `.venv/Scripts/python -m pytest -q`).

---

## Usage

1. **Reproduce a trial run end-to-end** (sample → extract → pack → train, on
   a small percentage of the dataset — the fast way to verify the whole
   pipeline works in a new environment):
   ```bash
   python scripts/run_trial.py --percent 2 --device cuda
   ```
2. **Train an official full-dataset config** (once the full packed dataset
   exists locally — see [`docs/DATA_AND_OUTPUTS.md`](docs/DATA_AND_OUTPUTS.md)):
   ```bash
   python scripts/train_sample.py --config configs/full_ctc.yaml --device cuda
   ```
   Every config in [`configs/`](configs/) is a real, reproducible experiment
   that has an entry in [`docs/logs/EXPERIMENTS.md`](docs/logs/EXPERIMENTS.md).
3. **Explore results without training anything** — start at
   [`docs/logs/RESULTS_OVERVIEW.md`](docs/logs/RESULTS_OVERVIEW.md).

---

## Design notes

- **Store raw, transform at load** — on-disk features are raw 258-dim;
  `bornil.data.transforms` applies normalization + landmark selection at load
  time, so the scheme can change without a costly re-extraction.
- **Frozen splits, locked metrics** — every claim in this project is scored
  against versioned split files and a metric protocol that's never quietly
  changed underneath a comparison.
- **Single source of truth per concern** — e.g. the model contract lives in
  `bornil/models/build.py`, never duplicated per-script; the model always
  reads its input width from `feature_dim()`, never a hard-coded literal.
  (A drifted duplicate of the model-construction logic once silently broke
  every CTC checkpoint reload — see `docs/logs/FAILURES.md` — which is why
  this rule is enforced structurally now, not just by convention.)
- **Negative results are logged as data, not hidden** — every dead end,
  failed gate, or result that didn't support its hypothesis gets a
  `FAILURES.md` entry with root-cause analysis. Several already inform this
  project's Limitations narrative.
- **Weakly-aligned targets** — one keypoint sequence ↔ one sentence with no
  per-frame labels; WS-E induces frame↔subword alignment post hoc via CTC
  forced alignment, validated against an independent motion signal rather
  than assumed correct.
