# WS-D / D1 — Self-Supervised Keypoint Pretraining (Masked Keypoint Modeling): Design

> Status: approved in brainstorming session 2026-07-25. Follow-on to
> `docs/plan/2026-07-13-bornil-implementation-plan.md` §8.1 (D1). Feeds the
> WS-D novel-method comparison (D1 vs D2 vs D3), which itself feeds the
> NeurIPS D&B paper's headline method.

## Goal

Pretrain the encoder half of the current best model (TConv2k5 + Transformer,
`data_full/runs/b0a590ea3317`, chrF++ 10.913) via masked keypoint modeling
(MKM) on the 14,390 `random-v1` train-split clips, using no text labels.
Finetune from that checkpoint and show it beats the identical architecture
trained from scratch — both on a single held-out comparison and on a
data-efficiency curve (the plan's "low-resource money plot": finetune
chrF++/BLEU-4 vs. % of labeled clips used).

## Context / scope correction from the master plan

`docs/plan/2026-07-13-bornil-implementation-plan.md` §8.1 motivates D1 with
"you own Bornil (and its unlabeled video beyond the 17,988)". This does **not**
hold: `chunks/` contains exactly 17,988 `.webm` files, one per labeled clip —
verified by direct count during brainstorming (2026-07-25). There is no extra
unlabeled video. **D1 is reframed**: SSL pretraining happens on the *same*
train-split clips, without touching their text labels, then the identical
clips (now with labels) are used to finetune. The value proposition is a
better inductive bias / representation-learning story, not a data-abundance
story. This must be stated explicitly wherever D1 results are written up —
do not let the "unlabeled video" framing survive into the paper unchanged.

**Explicitly out of scope for this spec** (deferred to follow-on specs if
there's appetite after this ships): motion-contrastive pretraining (the
plan's second D1 direction), D2 (Mamba/SSM), D3 (KP→Bangla-LLM). Each of
those is an independent sub-project with its own spec → plan → implementation
cycle. Do not build them here (YAGNI, and per this repo's own convention of
keeping workstreams additive and independently gated).

## Architecture

### Where masking happens (the one subtle decision in this spec — read before implementing)

`bornil/models/transformer.py`'s `SignTransformer._project_src` does:
`input_proj (Linear) -> temporal_frontend (TConv, kernel=5) -> PositionalEncoding`,
then feeds the Transformer encoder.

Mask-token substitution must happen **in `d_model` space, immediately after
`input_proj`, before `temporal_frontend`** — not on raw input features, and
not after the TConv.

**Why this matters:** TConv has kernel=5 (receptive field spans neighboring
frames). If masking were applied *after* the TConv, an unmasked neighbor's
TConv output would already have mixed in a "masked" position's true value
through the conv's receptive field — leaking the reconstruction target into
context the transformer can see, weakening the pretext task near every mask
boundary. Masking *before* the TConv means the TConv itself only ever
consumes the mask-substituted sequence, so both the TConv and the Transformer
encoder are forced to learn to fill in from context. This also means both
parts of the encoder actually get pretrained, not just the Transformer half.

### Masking strategy

- **Span masking**, not i.i.d. per-frame masking: sample contiguous temporal
  spans to mask (harder to solve via local interpolation than isolated
  single-frame gaps; matches VideoMAE / SkeletonMAE practice).
- **Mask ratio:** target ~50–70% of *valid* (non-padded) frames per clip.
  Pick one value (recommend starting at 0.6) and record it in the run config;
  do not grid-search this before confirming the core pretrain→finetune win —
  that is a follow-on tuning pass, not part of the first result.
- **Padding is never masked or scored.** Use the existing `src_lengths` /
  `src_key_padding_mask` machinery (`SignTransformer.src_padding_mask`) —
  do not reimplement padding logic.
- Mask token: a single learnable `nn.Parameter(d_model)`, broadcast to
  replace the projected+pre-TConv feature vector at every masked frame
  position, same value reused across all masked positions (standard MAE
  practice — the position is recovered via positional encoding + context,
  not by which mask token variant was used).

### Reconstruction target and loss

- Target: the same per-frame keypoint features that go into `input_proj`
  (i.e., reconstruct in the normalized keypoint feature space the dataset
  already uses — do not invent a new normalization).
- Reconstruction head: a small module (`MKMHead`, linear or 1-hidden-layer
  MLP) mapping encoder output `(B, S, d_model)` back to `(B, S, input_dim)`.
- Loss: L1 (or MSE — pick one, record it; L1 is the more common MAE choice
  and more robust to outlier keypoints from noisy MediaPipe frames) computed
  **only at masked, non-padded positions**. Mean-reduce over masked positions
  and batch.

## Components

New package `bornil/pretrain/` (mirrors the existing `bornil/align/` package
added for WS-E — same additive-package convention, avoids duplicating
model-construction logic the way `eval_checkpoint_decode.py` once drifted
from the real builder, per `docs/logs/FAILURES.md` context around
`bornil/models/build.py`).

- **`bornil/pretrain/__init__.py`** — docstring only.
- **`bornil/pretrain/masking.py`** — pure functions, no model/torch-module
  state:
  - `sample_span_mask(seq_len: int, lengths: Tensor, ratio: float, min_span: int, max_span: int, generator=None) -> BoolTensor (B, S)` —
    `True` at masked positions. Never masks position `>= lengths[b]`. Spans
    are contiguous; total masked fraction per clip should land close to
    `ratio` (greedy sampling of spans until the ratio is hit or exceeded,
    then trim the last span so it doesn't overshoot past `lengths[b]`).
- **`bornil/pretrain/mkm.py`**:
  - `class MKMHead(nn.Module)` — `__init__(d_model, input_dim)`,
    `forward(memory) -> (B, S, input_dim)`.
  - `mkm_loss(pred: Tensor, target: Tensor, mask: BoolTensor) -> Tensor` —
    masked-mean L1, returns a scalar. Raise or return `0.0` cleanly if a
    batch has zero masked positions (edge case worth a test).
  - `class MaskedSignEncoder(nn.Module)` — wraps a `SignTransformer` instance
    (built via `bornil.models.build.build_model`, so it never drifts from the
    real construction contract) plus a learnable mask token and an
    `MKMHead`. `forward(src, src_lengths, mask) -> (B, S, input_dim)`
    reproduces `SignTransformer._project_src` up to the mask-token
    substitution point, applies the mask, continues through
    `temporal_frontend` + positional encoding + `transformer.encoder`, then
    the `MKMHead`. This requires either (a) a small refactor of
    `SignTransformer._project_src` to expose a pre-TConv hook, or (b)
    duplicating the three lines of that method here with a comment pointing
    back to it. Prefer (a): add a `SignTransformer.encode_from_projected(...)`
    or split `_project_src` into `project(src)` and `apply_temporal(s)` so
    `MaskedSignEncoder` calls the same code path as the real model instead of
    a parallel copy. Keep the change to `SignTransformer` minimal and
    covered by the existing model tests (no behavior change for normal
    forward/encode/ctc_logits calls — add a regression test asserting that).
- **`scripts/pretrain_mkm.py`** — CLI, structured like `train_sample.py`:
  - Args: `--config` (yaml: encoder hyperparams matching a real model config
    so the pretrained checkpoint is architecture-compatible with finetuning;
    `mask_ratio`, `min_span`, `max_span`; standard train args: epochs, lr,
    batch size, device, seed).
  - Loads `random-v1` **train split only** (never val/test — leaking eval
    clips into pretraining, even unlabeled, would contaminate the later
    finetune-eval comparison).
  - Trains `MaskedSignEncoder`, logs mean masked-reconstruction loss per
    epoch, saves `data_full/runs/<hash>/encoder_pretrain.pt` (encoder +
    TConv frontend weights only — not the mask token or `MKMHead`, which are
    pretraining-only) plus `config.json` (config hash discipline, same as
    every other run) and an `EXPERIMENTS.md`-style row noting it's a
    pretraining run (no BLEU/chrF — log reconstruction loss curve instead).
  - **Gate, mirroring WS-E's A0 pattern:** if masked-reconstruction loss does
    not decrease over the run (or plateaus at the same level as a
    frame-shuffled-input control), stop, log to `docs/logs/FAILURES.md`, and
    do not proceed to finetuning — a non-decreasing loss means the pretext
    task isn't teaching anything and finetuning from it would just be a
    slower random init.
- **`scripts/train_sample.py`** — one new optional flag: `--init-encoder
  <path to encoder_pretrain.pt>`. When set, after building the model via
  `build_model`, load matching encoder-submodule weights
  (`input_proj`, `temporal_frontend`, `transformer.encoder`) from the
  checkpoint via `load_state_dict(..., strict=False)` with an explicit
  assertion that the expected keys were actually present and loaded (don't
  silently no-op on a key mismatch — that exact silent-failure class of bug
  already bit this repo once, see `docs/logs/FAILURES.md` build_model
  history). Decoder and CTC head remain randomly initialized (no text
  supervision exists at pretrain time). Log `init_encoder=<path>` in the run
  config/EXPERIMENTS row so every finetuned run's provenance is traceable.

## Data flow

1. Pretrain once on `splits/random-v1/train.txt` (14,390 clips, no labels
   used) → `data_full/runs/<pretrain-hash>/encoder_pretrain.pt`.
2. **Core-claim comparison** — two arms, same architecture, same split, same
   hyperparameters except init, on `signerindep-v1` (the hardest split —
   matches the plan's own validation criterion):
   - Arm A (new): `train_sample.py --config configs/full_ctc.yaml
     --init-encoder data_full/runs/<pretrain-hash>/encoder_pretrain.pt
     --split signerindep-v1`.
   - Arm B (reused, not rerun): the existing from-scratch
     `40cffb9e62bb` — already `configs/full_ctc.yaml` on 100%-label
     `signerindep-v1`.
3. **Data-efficiency curve** — filter `random-v1` (chosen over
   `signerindep-v1` for comparability with the largest body of existing
   baseline coverage) train split to 10/25/50/100% of labeled clips
   (stratified random subsample, fixed seed, a new fraction-based filter —
   see below), and run both arms (pretrained-init, from-scratch) at each
   fraction. **Only the from-scratch 100% point reuses an existing run**
   (`b0a590ea3317` is exactly `configs/full_ctc.yaml` on 100%-label
   `random-v1` from scratch) — the pretrained-init arm has never been run
   before at any fraction, including 100%, so all 4 of its points are new.
   That is **7 new finetune runs** for the curve (pretrained ×
   {10,25,50,100}%, scratch × {10,25,50}%) plus 1 reused point. The fraction
   filter is new code (`filter_split_by_fraction`, alongside
   `filter_split_by_recordings` in `bornil/benchmark/splits.py`): the
   existing LTE12 mechanism filters by *target length*, not by *fraction of
   clips*, so it is reused as a pattern (resolve the frozen split first,
   then filter inside each partition) but is a different function with a
   different selection rule. Unlike the LTE12 filter, only the **train**
   partition is filtered — val/test stay at full size in every run so
   evaluation stays comparable across the whole curve.
4. **Total new training jobs for this spec: 1 pretrain run + 1 signerindep-v1
   finetune run + 7 curve finetune runs = 9.**
5. All finetune runs go through the existing `results_significance.py` /
   `RESULTS_OVERVIEW.md` / `CHANGELOG.md` pipeline unchanged — no new eval
   machinery.

## Validation / success criteria

1. **Pretraining sanity gate:** reconstruction loss decreases and is finite;
   contrast against a frame-shuffled-input control (loss should be *higher*
   or reconstruction *harder* under shuffle if the model is using real
   temporal structure — mirrors A0's frame-shuffle contrast from WS-E).
2. **Core claim:** pretrained-then-finetuned chrF++ significantly exceeds
   from-scratch on `signerindep-v1` (or the chosen curve split), non-
   overlapping 95% CI, two-sided paired bootstrap n=1000 — same protocol as
   every other model comparison in `docs/logs/SIGNIFICANCE_*.md`.
3. **Data-efficiency curve:** plot/table chrF++ (and BLEU-4 as secondary) vs.
   label fraction for both arms. The SSL story is supported if the gap
   between arms widens as label fraction shrinks.
4. Every run gets a normal `EXPERIMENTS.md` row.

## Failure modes → what to do

- **Reconstruction loss doesn't decrease (gate fails):** stop before
  finetuning, log to `FAILURES.md`, do not proceed to Arm A. This is the
  cheap failure to catch early.
- **Pretrained-then-finetuned does not beat from-scratch:** this is a valid,
  loggable negative result (repo convention — see FAIL-004, FAIL-005). Log
  it plainly. **Do not** retroactively raise the mask ratio, change the loss
  function, or otherwise tune against the eval split to force a win — that
  would invalidate the significance test. Any such tuning must happen on a
  held-out-from-the-comparison config choice made *before* looking at
  finetune results, or in a clearly-labeled follow-on experiment.
- **Data-efficiency curve is flat / no widening gap:** still report it — a
  flat curve where the single-point comparison wins is itself informative
  (says the gain isn't specifically a low-resource effect) and should be
  written up honestly, not omitted.

## Testing (TDD, matching repo convention — write failing tests first)

- `tests/pretrain/test_masking.py` — `sample_span_mask`: ratio approximately
  respected, never masks `>= lengths[b]`, masked positions form contiguous
  spans, deterministic under a fixed generator/seed.
- `tests/pretrain/test_mkm.py` — `mkm_loss`: masked-only reduction (loss
  changes only when masked-position values differ; unmasked-position
  differences don't move it), correct scalar shape, zero-masked-positions
  edge case handled cleanly.
- `tests/pretrain/test_masked_encoder.py` — `MaskedSignEncoder` on tiny
  synthetic input: output shape `(B, S, input_dim)`; an
  **overfit-tiny gate** (this repo's standing pattern from
  `docs/plan/2026-07-13-bornil-implementation-plan.md` §4.6) — the module
  should drive masked-reconstruction loss below a small threshold on a
  single repeated tiny batch within a bounded number of steps, proving the
  masking → encode → reconstruct path is wired correctly before any real
  training run.
- `tests/models/test_transformer.py` — regression test: after refactoring
  `_project_src`, `SignTransformer.forward`/`encode`/`ctc_logits` produce
  bit-identical output to before the refactor on a fixed seed (no behavior
  change for the existing model).
- `tests/train/test_train_sample_init_encoder.py` (or extend the existing
  train_sample test file) — `--init-encoder` actually changes encoder
  weights vs. a fresh random init (compare a couple of parameter tensors by
  value), and leaves decoder/CTC-head weights untouched by the loaded
  checkpoint (still randomly initialized, not the pretrain run's values).

## Compute budget (for planning, not a hard spec)

Pretraining is one run over 14,390 clips with a lightweight head — expect
roughly TConv+CTC-scale training time (that run was ~45 min for a full
official split) times more epochs (SSL pretraining conventionally wants more
epochs than the ~20–30 used for supervised finetuning here since there's no
early-stopping-on-metric signal); budget a few hours on the RTX 3070 and
confirm empirically rather than assuming. The finetune arms are each roughly
TConv+CTC-scale (~45 min – a few hours depending on split size and
`ctc_weight`/epoch count already used elsewhere) — the fractional-label runs
(10/25/50%) should be proportionally *faster* per epoch (fewer train batches)
but may need more epochs to converge; don't assume linear scaling, measure
the first one and extrapolate. Total for this spec: 1 pretrain run + 8
finetune runs (9 training jobs total, §Data flow) — plan for a full working
day of GPU time, not minutes. This workload is exactly why the implementation
plan must be resumable per-task: expect this to span a multi-hour unattended
session.
