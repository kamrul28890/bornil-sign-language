# WS-D / D1 — Masked Keypoint Modeling (SSL Pretraining) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Design doc:
> `docs/superpowers/specs/2026-07-25-ws-d-d1-ssl-pretraining-design.md` — read it
> once before starting; this plan operationalizes it and does not repeat its
> reasoning. Follow-on to `docs/plan/2026-07-13-bornil-implementation-plan.md`
> §8.1 (D1).

**Goal:** pretrain the TConv2k5+Transformer encoder via masked keypoint
modeling (MKM) on the unlabeled `random-v1` train clips, then show
pretrained-then-finetuned beats from-scratch on `signerindep-v1` (paired
bootstrap, p<0.05) and produce a data-efficiency curve on `random-v1`
(10/25/50/100% labeled clips).

**Architecture:** mask-token substitution happens in `d_model` space right
after `input_proj`, before the TConv frontend (`temporal_frontend`) — see
design doc "Architecture" section for why (conv-boundary leak otherwise). A
new `bornil/pretrain/` package holds the masking, loss, and wrapper-encoder
code; `scripts/pretrain_mkm.py` is the pretraining CLI; `scripts/train_sample.py`
gets one new flag (`--init-encoder`) to warm-start from a pretrain checkpoint,
and one new flag (`--label-fraction`) to run the data-efficiency curve.

**Tech Stack:** pure PyTorch (no new dependencies). Reuses `build_model`,
`SignDataset`, `LengthBucketBatchSampler`, `resolve_split`, `config_hash` —
every new script is additive, nothing in the existing training/eval path
changes behavior for existing configs.

## Global Constraints

- **Windows import order:** every script entry point imports `bornil.runtime`
  before pandas (FAIL-002). Use `.venv/Scripts/python`. PowerShell has no
  `&&` — chain with `;`, or use one of `python -c` / a single command per
  line. If you are Codex running in a bash-compatible shell, `&&` is fine
  there — check which shell you actually have before assuming.
- **No behavior change to existing configs.** `configs/full_ctc.yaml` and
  every other existing config must produce byte-identical training behavior
  after this plan lands. Task 3's refactor is a pure decomposition — verify
  with the existing golden tests (`tests/models/test_decode_golden.py`), not
  just a new unit test.
- **Do not modify the locked metric protocol** (`bornil/benchmark/metrics.py`).
  This plan is additive: a new `bornil/pretrain/` package, one new function in
  `bornil/benchmark/splits.py`, two new CLI flags on `scripts/train_sample.py`,
  and one new script.
- **Frozen splits only for real runs.** Every real (non-tiny-synthetic) run
  must resolve `splits/<kind>/` via `resolve_split` — never
  `--no-frozen-splits` outside of tests.
- **Logging discipline:** every real training run gets an `EXPERIMENTS.md` row
  (already automatic via `_append_experiment_row` in `train_sample.py`/your
  new `pretrain_mkm.py`). Meaningful changes → `docs/logs/CHANGELOG.md`.
  Dead-ends, failed gates, or negative results → `docs/logs/FAILURES.md`. Do
  not skip these — they are how the next person (human or agent) picks up
  this work without re-deriving it.
- **TDD:** write the failing test before the implementation, for every task
  that has one. Do not write implementation code first and backfill tests.
- **Commit after every task**, not just at the end. Use the exact commit
  message given in each task's Step 5/final step unless noted otherwise.
- **This plan is resumable.** Before starting work in a fresh session, read
  this file top to bottom and find the first unchecked `- [ ]` box. Do not
  redo a task whose boxes are already checked without first verifying its
  tests still pass (`git log` should show its commit). If a task's checkboxes
  are partially checked, resume from the first unchecked step in that task —
  do not restart the task from Step 1.
- **Long-running steps are marked ⏱.** These are real GPU training jobs, not
  quick commands — expect anywhere from ~10 minutes to a few hours each (see
  the design doc's "Compute budget" section). Run them with
  `run_in_background`-equivalent behavior if your harness supports it, or in
  a way that lets you keep working / check back rather than blocking
  synchronously the whole time. Do not reduce `epochs` or dataset size on
  these to make them faster — that invalidates the run.

---

## Task 1: Span masking (`bornil/pretrain/masking.py`)

**Files:**
- Create: `bornil/pretrain/__init__.py`
- Create: `bornil/pretrain/masking.py`
- Test: `tests/pretrain/test_masking.py`

**Interfaces:**
- Produces: `sample_span_mask(seq_len: int, lengths: torch.Tensor, ratio: float, min_span: int = 3, max_span: int = 15, generator: torch.Generator = None) -> torch.BoolTensor` shaped `(B, S)` where `B = lengths.shape[0]`, `S = seq_len`. `True` = masked. Never masks position `>= lengths[b]`. Masked positions per row form contiguous spans (sampled greedily until the row's masked fraction reaches `ratio`, trimming the last span so it never crosses `lengths[b]`).

- [x] **Step 1: Write the failing tests**

```python
# tests/pretrain/test_masking.py
import torch

from bornil.pretrain.masking import sample_span_mask


def test_output_shape_and_dtype():
    lengths = torch.tensor([10, 6, 10])
    mask = sample_span_mask(seq_len=10, lengths=lengths, ratio=0.5,
                            generator=torch.Generator().manual_seed(0))
    assert mask.shape == (3, 10)
    assert mask.dtype == torch.bool


def test_never_masks_beyond_length():
    lengths = torch.tensor([4, 10])
    mask = sample_span_mask(seq_len=10, lengths=lengths, ratio=0.9,
                            min_span=1, max_span=2,
                            generator=torch.Generator().manual_seed(1))
    assert not mask[0, 4:].any()


def test_ratio_is_approximately_respected():
    lengths = torch.full((1,), 1000, dtype=torch.long)
    mask = sample_span_mask(seq_len=1000, lengths=lengths, ratio=0.6,
                            min_span=3, max_span=15,
                            generator=torch.Generator().manual_seed(2))
    frac = mask[0].float().mean().item()
    assert 0.5 <= frac <= 0.7


def test_masked_positions_form_contiguous_spans():
    lengths = torch.tensor([200])
    mask = sample_span_mask(seq_len=200, lengths=lengths, ratio=0.4,
                            min_span=5, max_span=5,
                            generator=torch.Generator().manual_seed(3))
    row = mask[0].tolist()
    # count contiguous True-runs; with min_span==max_span==5 every run's
    # length must be a multiple of 5 give or take the final trim.
    runs = []
    cur = 0
    for v in row:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    assert all(r <= 5 for r in runs)
    assert len(runs) > 1


def test_deterministic_with_seeded_generator():
    lengths = torch.tensor([50])
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    m1 = sample_span_mask(seq_len=50, lengths=lengths, ratio=0.5, generator=g1)
    m2 = sample_span_mask(seq_len=50, lengths=lengths, ratio=0.5, generator=g2)
    assert torch.equal(m1, m2)


def test_zero_length_row_is_never_masked():
    lengths = torch.tensor([0, 10])
    mask = sample_span_mask(seq_len=10, lengths=lengths, ratio=0.5,
                            generator=torch.Generator().manual_seed(4))
    assert not mask[0].any()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_masking.py -v`
Expected: FAIL — `No module named 'bornil.pretrain'`.

- [x] **Step 3: Implement the module**

```python
# bornil/pretrain/__init__.py
"""WS-D D1: self-supervised keypoint pretraining (masked keypoint modeling)."""
```

```python
# bornil/pretrain/masking.py
"""Span masking for masked keypoint modeling (WS-D D1).

Masks contiguous temporal spans rather than i.i.d. frames — a masked span
can't be filled in by simple local interpolation the way an isolated masked
frame could, matching VideoMAE/SkeletonMAE practice. Never masks padded
positions (index >= that row's true length).
"""
from __future__ import annotations

import torch


def sample_span_mask(seq_len: int, lengths: torch.Tensor, ratio: float,
                     min_span: int = 3, max_span: int = 15,
                     generator: torch.Generator = None) -> torch.Tensor:
    """Bool mask (B, S), True = masked. Greedily samples spans per row until
    `ratio` of that row's valid (non-padded) length is masked, trimming the
    final span so it never crosses `lengths[b]`."""
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    if min_span < 1 or max_span < min_span:
        raise ValueError("require 1 <= min_span <= max_span")
    B = lengths.shape[0]
    mask = torch.zeros(B, seq_len, dtype=torch.bool)
    for b in range(B):
        valid = int(lengths[b].item())
        if valid <= 0:
            continue
        target = int(round(ratio * valid))
        if target <= 0:
            continue
        masked = 0
        guard = 0
        while masked < target and guard < 10_000:
            guard += 1
            span = int(torch.randint(min_span, max_span + 1, (1,),
                                     generator=generator).item())
            span = min(span, valid)
            start = int(torch.randint(0, valid - span + 1, (1,),
                                      generator=generator).item())
            end = start + span
            newly = (~mask[b, start:end]).sum().item()
            if newly == 0:
                continue
            if masked + newly > target:
                # trim from the end of this span so we don't overshoot
                keep = target - masked
                mask[b, start:start + keep] = True
                masked += keep
                break
            mask[b, start:end] = True
            masked += newly
    return mask
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_masking.py -v`
Expected: 6 passed.

Implementation note (2026-07-25): the supplied Step 3 code allowed adjacent
sampled spans to merge, which failed Step 1's `r <= max_span` assertion.
The implementation therefore prefers non-touching spans, with a bounded
fallback so high requested ratios still terminate.

- [x] **Step 5: Commit**

```bash
git add bornil/pretrain/__init__.py bornil/pretrain/masking.py tests/pretrain/test_masking.py
git commit -m "feat(pretrain): span-masking primitive for masked keypoint modeling"
```

---

## Task 2: Reconstruction head + masked loss (`bornil/pretrain/mkm.py`)

**Files:**
- Create: `bornil/pretrain/mkm.py`
- Test: `tests/pretrain/test_mkm.py`

**Interfaces:**
- Consumes: nothing from Task 1 yet (this task is pure loss/head machinery; masking is wired in at Task 4).
- Produces:
  - `class MKMHead(nn.Module)` — `__init__(self, d_model: int, input_dim: int)`; `forward(memory: Tensor (B,S,d_model)) -> Tensor (B,S,input_dim)`.
  - `mkm_loss(pred: Tensor, target: Tensor, mask: torch.BoolTensor) -> Tensor` — scalar. Masked-positions-only mean L1 loss (`pred`, `target` both `(B,S,input_dim)`, `mask` `(B,S)`). Returns `torch.tensor(0.0)` (not NaN, not a Python float — keep it a tensor so `.backward()` still works, e.g. via `pred.sum() * 0.0`) when `mask` has zero True entries anywhere in the batch.

- [x] **Step 1: Write the failing tests**

```python
# tests/pretrain/test_mkm.py
import torch

from bornil.pretrain.mkm import MKMHead, mkm_loss


def test_mkm_head_output_shape():
    head = MKMHead(d_model=16, input_dim=218)
    memory = torch.randn(2, 9, 16)
    out = head(memory)
    assert out.shape == (2, 9, 218)


def test_loss_is_zero_for_perfect_prediction_on_masked_positions():
    pred = torch.randn(2, 5, 3)
    target = pred.clone()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1:3] = True
    loss = mkm_loss(pred, target, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_loss_ignores_unmasked_position_errors():
    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    target[0, 3] = 100.0          # huge error at an UNMASKED position
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True             # only position 0 is masked, and pred==target there
    loss = mkm_loss(pred, target, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_loss_reflects_masked_position_error():
    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    target[0, 0] = torch.tensor([1.0, 1.0])   # error at the masked position
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True
    loss = mkm_loss(pred, target, mask)
    assert loss.item() > 0.0


def test_zero_masked_positions_returns_zero_tensor_not_nan():
    pred = torch.randn(2, 5, 3, requires_grad=True)
    target = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    loss = mkm_loss(pred, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()   # must not raise
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_mkm.py -v`
Expected: FAIL — `No module named 'bornil.pretrain.mkm'`.

- [x] **Step 3: Implement**

```python
# bornil/pretrain/mkm.py
"""Masked keypoint modeling: reconstruction head + masked loss (WS-D D1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MKMHead(nn.Module):
    """Linear projection from encoder output back to per-frame keypoint space."""

    def __init__(self, d_model: int, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, input_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.proj(memory)


def mkm_loss(pred: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    """Masked-mean L1 reconstruction loss. `mask` True = position contributes."""
    if not mask.any():
        return pred.sum() * 0.0
    diff = (pred - target).abs()
    per_frame = diff.mean(dim=-1)          # (B, S)
    return per_frame[mask].mean()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_mkm.py -v`
Expected: 5 passed.

- [x] **Step 5: Commit**

```bash
git add bornil/pretrain/mkm.py tests/pretrain/test_mkm.py
git commit -m "feat(pretrain): MKM reconstruction head + masked L1 loss"
```

---

## Task 3: Refactor `SignTransformer._project_src` (expose the mask injection point)

**Files:**
- Modify: `bornil/models/transformer.py`
- Test: `tests/models/test_transformer.py` (add one test)

**Read first:** `bornil/models/transformer.py` lines 96–100 (the current `_project_src`). This task is a **pure decomposition — no behavior change**. Do not alter `input_proj`, `temporal_frontend`, or `pos` themselves.

**Interfaces:**
- Produces: `SignTransformer.project(src) -> Tensor (B,S,d_model)` (input projection only, pre-TConv, pre-positional-encoding — **this is where Task 4 injects the mask token**); `SignTransformer.apply_temporal(s) -> Tensor (B,S,d_model)` (TConv frontend if configured, then positional encoding). `_project_src` becomes `apply_temporal(project(src))` and must keep its exact existing signature and behavior (every existing caller — `forward`, `encode` — is unaffected).

- [x] **Step 1: Write the failing test**

```python
# add to tests/models/test_transformer.py
def test_project_and_apply_temporal_compose_to_project_src():
    model = SignTransformer(
        input_dim=218, vocab_size=20, d_model=64, nhead=4,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128,
        dropout=0.0, pad_id=0, temporal_conv_layers=2, temporal_conv_kernel=5,
    )
    src = torch.randn(2, 15, 218)
    composed = model.apply_temporal(model.project(src))
    direct = model._project_src(src)
    assert torch.allclose(composed, direct)
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_transformer.py::test_project_and_apply_temporal_compose_to_project_src -v`
Expected: FAIL — `AttributeError: 'SignTransformer' object has no attribute 'project'`.

- [x] **Step 3: Refactor `_project_src`**

In `bornil/models/transformer.py`, replace:

```python
    def _project_src(self, src: torch.Tensor) -> torch.Tensor:
        s = self.input_proj(src) * math.sqrt(self.d_model)
        if not isinstance(self.temporal_frontend, nn.Identity):
            s = self.temporal_frontend(s.transpose(1, 2)).transpose(1, 2)
        return self.pos(s)
```

with:

```python
    def project(self, src: torch.Tensor) -> torch.Tensor:
        """Input projection only (pre-TConv, pre-positional-encoding).

        This is the mask-token injection point for masked keypoint modeling
        (bornil.pretrain.mkm.MaskedSignEncoder) — masking here, before the
        TConv frontend, means the TConv itself only ever sees the
        mask-substituted sequence, so a masked position's true value can't
        leak into a neighboring unmasked position's TConv output.
        """
        return self.input_proj(src) * math.sqrt(self.d_model)

    def apply_temporal(self, s: torch.Tensor) -> torch.Tensor:
        """TConv frontend (if configured) + positional encoding."""
        if not isinstance(self.temporal_frontend, nn.Identity):
            s = self.temporal_frontend(s.transpose(1, 2)).transpose(1, 2)
        return self.pos(s)

    def _project_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.apply_temporal(self.project(src))
```

- [x] **Step 4: Run tests to verify they pass, and confirm no regression**

Run these three commands and confirm all pass with unchanged pass counts vs. before this task:

```bash
.venv/Scripts/python -m pytest tests/models/test_transformer.py -v
.venv/Scripts/python -m pytest tests/models/test_ctc_head.py -v
.venv/Scripts/python -m pytest tests/models/test_decode_golden.py -v
```

Expected: all pass, including the new `test_project_and_apply_temporal_compose_to_project_src`. `test_decode_golden.py` passing unchanged is the real regression guard here — it pins exact decode output, so if the refactor had changed numerics it would fail.

- [x] **Step 5: Commit**

```bash
git add bornil/models/transformer.py tests/models/test_transformer.py
git commit -m "refactor(models): split SignTransformer._project_src into project()+apply_temporal()"
```

---

## Task 4: `MaskedSignEncoder` (wires masking + the refactored encoder + MKM head)

**Files:**
- Modify: `bornil/pretrain/mkm.py` (add to it)
- Test: `tests/pretrain/test_masked_encoder.py`

**Read first:** `bornil/models/build.py` (`build_model`) — `MaskedSignEncoder` wraps a `SignTransformer` built through this, never constructs one by hand, so a pretrain checkpoint is always architecture-compatible with a normal finetune run built the same way.

**Interfaces:**
- Consumes: `SignTransformer.project`/`apply_temporal` (Task 3), `sample_span_mask` (Task 1), `MKMHead`/`mkm_loss` (Task 2).
- Produces: `class MaskedSignEncoder(nn.Module)`:
  - `__init__(self, model: SignTransformer, mask_ratio: float = 0.6, min_span: int = 3, max_span: int = 15)`
  - `forward(self, src: Tensor (B,S,input_dim), src_lengths: Tensor (B,), generator: torch.Generator = None) -> tuple[Tensor (B,S,input_dim), BoolTensor (B,S)]` — returns `(reconstruction, mask)`.
  - `loss(self, src, src_lengths, generator=None) -> Tensor` — convenience: `forward` then `mkm_loss(pred, src, mask)` (the reconstruction target is the same `src` that was masked — MAE reconstructs its own input).

- [ ] **Step 1: Write the failing test**

```python
# tests/pretrain/test_masked_encoder.py
import torch

from bornil.models.transformer import SignTransformer
from bornil.pretrain.mkm import MaskedSignEncoder


def _tiny_model():
    torch.manual_seed(0)
    return SignTransformer(
        input_dim=12, vocab_size=10, d_model=32, nhead=4,
        num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, pad_id=0, temporal_conv_layers=2, temporal_conv_kernel=5,
    )


def test_forward_output_shapes():
    enc = MaskedSignEncoder(_tiny_model(), mask_ratio=0.5, min_span=2, max_span=4)
    src = torch.randn(3, 20, 12)
    lengths = torch.tensor([20, 15, 20])
    pred, mask = enc(src, lengths, generator=torch.Generator().manual_seed(0))
    assert pred.shape == (3, 20, 12)
    assert mask.shape == (3, 20)
    assert not mask[1, 15:].any()   # padded region never masked


def test_loss_is_finite_and_positive_before_training():
    torch.manual_seed(0)
    enc = MaskedSignEncoder(_tiny_model(), mask_ratio=0.5, min_span=2, max_span=4)
    src = torch.randn(2, 20, 12)
    lengths = torch.tensor([20, 20])
    loss = enc.loss(src, lengths, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_overfit_tiny():
    """Repeated tiny batch: masked-reconstruction loss must drop sharply.
    This is this repo's standing overfit-tiny gate (plan §4.6) — proves the
    masking -> encode -> reconstruct path is wired correctly before any real
    training run."""
    torch.manual_seed(0)
    enc = MaskedSignEncoder(_tiny_model(), mask_ratio=0.5, min_span=2, max_span=4)
    src = torch.randn(2, 20, 12)
    lengths = torch.tensor([20, 20])
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    first_loss = None
    for step in range(200):
        opt.zero_grad()
        loss = enc.loss(src, lengths)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss * 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_masked_encoder.py -v`
Expected: FAIL — `ImportError: cannot import name 'MaskedSignEncoder'`.

- [ ] **Step 3: Implement**

Append to `bornil/pretrain/mkm.py` (keep the existing `MKMHead`/`mkm_loss` above this):

```python
from bornil.pretrain.masking import sample_span_mask


class MaskedSignEncoder(nn.Module):
    """Wraps a SignTransformer for masked-keypoint-modeling pretraining.

    Mask-token substitution happens at `model.project(src)` — in d_model
    space, before the TConv frontend (see SignTransformer.project's
    docstring for why). Only the encoder half of `model` is ever used here;
    `model`'s decoder/CTC head exist (build_model always constructs them)
    but receive no gradient during pretraining — only the extracted encoder
    submodules get saved (see bornil/pretrain/checkpoint.py, Task 5).
    """

    def __init__(self, model, mask_ratio: float = 0.6, min_span: int = 3,
                max_span: int = 15):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.max_span = max_span
        self.mask_token = nn.Parameter(torch.zeros(model.d_model))
        input_dim = model.input_proj.in_features
        self.head = MKMHead(model.d_model, input_dim)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor,
               generator: torch.Generator = None):
        mask = sample_span_mask(src.size(1), src_lengths, self.mask_ratio,
                                self.min_span, self.max_span,
                                generator=generator).to(src.device)
        s = self.model.project(src)
        s = torch.where(mask.unsqueeze(-1), self.mask_token, s)
        s = self.model.apply_temporal(s)
        src_kpm = self.model.src_padding_mask(src_lengths, src.size(1))
        memory = self.model.transformer.encoder(s, src_key_padding_mask=src_kpm)
        pred = self.head(memory)
        return pred, mask

    def loss(self, src: torch.Tensor, src_lengths: torch.Tensor,
             generator: torch.Generator = None) -> torch.Tensor:
        pred, mask = self.forward(src, src_lengths, generator=generator)
        return mkm_loss(pred, src, mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_masked_encoder.py -v`
Expected: 3 passed. If `test_overfit_tiny` is flaky (doesn't clear the 10% threshold), raise the step count or lr slightly — do not weaken the assertion threshold below 10% without first checking whether the mask is actually being applied (a common bug here is broadcasting the mask wrong and silently masking nothing).

- [ ] **Step 5: Commit**

```bash
git add bornil/pretrain/mkm.py tests/pretrain/test_masked_encoder.py
git commit -m "feat(pretrain): MaskedSignEncoder wiring masking into the shared encoder"
```

---

## Task 5: Encoder-only checkpoint save/load (`bornil/pretrain/checkpoint.py`)

**Files:**
- Create: `bornil/pretrain/checkpoint.py`
- Test: `tests/pretrain/test_checkpoint.py`

**Why a separate module:** the pretrain checkpoint must contain only the
transferable encoder submodules (`input_proj`, `temporal_frontend`,
`transformer.encoder`) — not the mask token or `MKMHead` (pretraining-only,
meaningless to a finetuned model with a text decoder), and not the
decoder/CTC head (never trained during pretraining, would silently overwrite
a freshly-initialized finetune decoder with garbage if loaded).

**Interfaces:**
- Produces:
  - `ENCODER_KEY_PREFIXES = ("input_proj.", "temporal_frontend.", "transformer.encoder.")`
  - `encoder_state_dict(model) -> dict` — filters `model.state_dict()` to keys starting with one of `ENCODER_KEY_PREFIXES`.
  - `save_encoder_checkpoint(path, model) -> None`.
  - `load_encoder_checkpoint(model, path, device) -> set[str]` — loads via `load_state_dict(sd, strict=False)`, returns the set of keys actually loaded, and **raises `ValueError` if that set is empty** (silent no-op on a full key mismatch is exactly the class of bug `bornil/models/build.py` was written to prevent for the main checkpoint path — do not reintroduce it here).

- [ ] **Step 1: Write the failing tests**

```python
# tests/pretrain/test_checkpoint.py
import os

import torch

from bornil.models.transformer import SignTransformer
from bornil.pretrain.checkpoint import (encoder_state_dict, load_encoder_checkpoint,
                                        save_encoder_checkpoint)


def _tiny_model(seed):
    torch.manual_seed(seed)
    return SignTransformer(
        input_dim=12, vocab_size=10, d_model=32, nhead=4,
        num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, pad_id=0, temporal_conv_layers=2, temporal_conv_kernel=5,
    )


def test_encoder_state_dict_excludes_decoder_and_heads():
    model = _tiny_model(0)
    sd = encoder_state_dict(model)
    assert any(k.startswith("input_proj.") for k in sd)
    assert any(k.startswith("transformer.encoder.") for k in sd)
    assert not any(k.startswith("transformer.decoder.") for k in sd)
    assert not any(k.startswith("tok_embed.") for k in sd)
    assert not any(k.startswith("out.") for k in sd)


def test_save_and_load_roundtrip_changes_weights(tmp_path):
    src_model = _tiny_model(0)
    dst_model = _tiny_model(1)   # different seed -> different weights
    path = os.path.join(tmp_path, "encoder_pretrain.pt")
    save_encoder_checkpoint(path, src_model)

    before = dst_model.input_proj.weight.clone()
    loaded_keys = load_encoder_checkpoint(dst_model, path, device="cpu")
    after = dst_model.input_proj.weight

    assert not torch.allclose(before, after)
    assert torch.allclose(after, src_model.input_proj.weight)
    assert any(k.startswith("input_proj.") for k in loaded_keys)


def test_load_raises_on_empty_checkpoint(tmp_path):
    path = os.path.join(tmp_path, "empty.pt")
    torch.save({}, path)
    model = _tiny_model(0)
    try:
        load_encoder_checkpoint(model, path, device="cpu")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_decoder_and_heads_unaffected_by_load(tmp_path):
    """This is the property scripts/train_sample.py's --init-encoder flag
    (Task 8) relies on: warm-starting the encoder must never touch the
    decoder or CTC head, which stay randomly initialized for finetuning."""
    src_model = _tiny_model(0)
    dst_model = _tiny_model(1)
    path = os.path.join(tmp_path, "encoder_pretrain.pt")
    save_encoder_checkpoint(path, src_model)

    decoder_before = [p.clone() for p in dst_model.transformer.decoder.parameters()]
    out_before = dst_model.out.weight.clone()
    load_encoder_checkpoint(dst_model, path, device="cpu")
    decoder_after = list(dst_model.transformer.decoder.parameters())

    assert all(torch.equal(b, a) for b, a in zip(decoder_before, decoder_after))
    assert torch.equal(out_before, dst_model.out.weight)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_checkpoint.py -v`
Expected: FAIL — `No module named 'bornil.pretrain.checkpoint'`.

- [ ] **Step 3: Implement**

```python
# bornil/pretrain/checkpoint.py
"""Encoder-only checkpoints for masked-keypoint-modeling pretraining.

Only the transferable encoder submodules are saved/loaded — not the mask
token, MKM head, or the (untouched, randomly-initialized) decoder/CTC head.
"""
from __future__ import annotations

import os
from typing import Set

import torch

ENCODER_KEY_PREFIXES = ("input_proj.", "temporal_frontend.", "transformer.encoder.")


def encoder_state_dict(model) -> dict:
    return {k: v for k, v in model.state_dict().items()
           if k.startswith(ENCODER_KEY_PREFIXES)}


def save_encoder_checkpoint(path, model) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    torch.save(encoder_state_dict(model), path)


def load_encoder_checkpoint(model, path, device) -> Set[str]:
    sd = torch.load(path, map_location=device, weights_only=True)
    result = model.load_state_dict(sd, strict=False)
    loaded = set(sd.keys()) - set(result.unexpected_keys)
    if not loaded:
        raise ValueError(f"no matching encoder keys loaded from {path!r} "
                         f"(missing={result.missing_keys}, "
                         f"unexpected={result.unexpected_keys})")
    return loaded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/pretrain/test_checkpoint.py -v`
Expected: 4 passed. (This file's `test_decoder_and_heads_unaffected_by_load`
is what makes a dedicated test unnecessary in Task 8 — see that task's
note.)

- [ ] **Step 5: Commit**

```bash
git add bornil/pretrain/checkpoint.py tests/pretrain/test_checkpoint.py
git commit -m "feat(pretrain): encoder-only checkpoint save/load"
```

---

## Task 6: Label-fraction split filter (`bornil/benchmark/splits.py`)

**Files:**
- Modify: `bornil/benchmark/splits.py`
- Test: `tests/benchmark/test_splits.py` (add to existing file — check it exists first: `tests/benchmark/`)

**Why this is a new function, not a reuse of the LTE12 filter:** the existing
`filter_split_by_recordings`/target-length filtering in `scripts/train_sample.py`
(`_apply_target_length_filter`) selects rows by a text-length predicate
applied to *every* partition. The data-efficiency curve needs something
different: randomly subsample a *fraction* of the **train** partition only,
leaving val/test at full size so every point on the curve is evaluated
identically. Reuse the pattern (resolve frozen split, then filter inside
partitions) via `filter_split_by_recordings`'s existing style, not its logic.

**Interfaces:**
- Produces: `filter_split_by_fraction(split: Split, fraction: float, seed: int = 42) -> Split` — returns a new dict; `train` is a fixed-seed random subsample of `round(fraction * len(train))` (minimum 1) recordings, `val`/`test` are copied unchanged. `fraction == 1.0` returns train unchanged (same members, not necessarily same list object). Raises `ValueError` if `fraction` is not in `(0, 1]`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/benchmark/test_splits.py
from bornil.benchmark.splits import filter_split_by_fraction


def test_filter_by_fraction_shrinks_train_only():
    split = {"train": [f"r{i}" for i in range(100)], "val": ["v0", "v1"], "test": ["t0"]}
    out = filter_split_by_fraction(split, fraction=0.1, seed=0)
    assert len(out["train"]) == 10
    assert out["val"] == split["val"]
    assert out["test"] == split["test"]
    assert set(out["train"]) <= set(split["train"])


def test_filter_by_fraction_one_returns_full_train():
    split = {"train": [f"r{i}" for i in range(20)], "val": [], "test": []}
    out = filter_split_by_fraction(split, fraction=1.0, seed=0)
    assert set(out["train"]) == set(split["train"])
    assert len(out["train"]) == 20


def test_filter_by_fraction_is_deterministic():
    split = {"train": [f"r{i}" for i in range(50)], "val": [], "test": []}
    a = filter_split_by_fraction(split, fraction=0.3, seed=5)
    b = filter_split_by_fraction(split, fraction=0.3, seed=5)
    assert a["train"] == b["train"]


def test_filter_by_fraction_rejects_out_of_range():
    split = {"train": ["a"], "val": [], "test": []}
    for bad in (0.0, -0.1, 1.1):
        try:
            filter_split_by_fraction(split, fraction=bad)
            assert False, f"expected ValueError for fraction={bad}"
        except ValueError:
            pass


def test_filter_by_fraction_never_returns_empty_train():
    split = {"train": ["only-one"], "val": [], "test": []}
    out = filter_split_by_fraction(split, fraction=0.01, seed=0)
    assert len(out["train"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/benchmark/test_splits.py -k fraction -v`
Expected: FAIL — `ImportError: cannot import name 'filter_split_by_fraction'`.

- [ ] **Step 3: Implement**

Add to `bornil/benchmark/splits.py`, near `filter_split_by_recordings`:

```python
def filter_split_by_fraction(split: Split, fraction: float, seed: int = 42) -> Split:
    """Randomly subsample the train partition to `fraction` of its clips.

    val/test are returned unchanged (as new lists, same contents) so every
    point on a data-efficiency curve evaluates against the same held-out
    data. Unlike `filter_split_by_recordings`, this filters by a random
    fraction, not by a predicate on the manifest.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    train = list(split.get("train", []))
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = max(1, int(round(fraction * len(train))))
        keep_idx = sorted(rng.permutation(len(train))[:n].tolist())
        train = [train[i] for i in keep_idx]
    return {
        "train": train,
        "val": list(split.get("val", [])),
        "test": list(split.get("test", [])),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/benchmark/test_splits.py -v`
Expected: all pass (existing tests unaffected + 5 new ones).

- [ ] **Step 5: Commit**

```bash
git add bornil/benchmark/splits.py tests/benchmark/test_splits.py
git commit -m "feat(benchmark): filter_split_by_fraction for the data-efficiency curve"
```

---

## Task 7: `scripts/pretrain_mkm.py` CLI + config

**Files:**
- Create: `scripts/pretrain_mkm.py`
- Create: `configs/pretrain_mkm.yaml`

**Read first:** `scripts/train_sample.py` in full — copy its data-loading
structure (`resolve_split`, `SignDataset`, `LengthBucketBatchSampler`,
`make_collate_fn`, device resolution, config-hash run-dir convention,
`EXPERIMENTS.md` row appending) rather than reinventing it. This script
differs from `train_sample.py` only in: no test-split decode/scoring (there
is no text supervision to score), the loss is `MaskedSignEncoder.loss`
instead of the seq2seq/CTC loss, and it always trains on `train` only
(never evaluates on val/test — pretraining has no labels to score against,
and touching val/test here would contaminate the later finetune-eval
comparison per the design doc).

**No unit test for this task** — it is orchestration glue over already-tested
pieces (Tasks 1–6) and the existing data pipeline; it is validated by the
smoke run in Step 3 below and the real run in Task 9. This matches this
repo's existing convention (`train_sample.py` itself has no direct unit
test either — see `tests/train/`, which tests `train_epoch`/`evaluate`, not
the CLI).

- [ ] **Step 1: Write the config**

```yaml
# configs/pretrain_mkm.yaml
trial:
  name: data_full_pretrain_mkm
  seed: 42
  out_dir: data_full
data:
  manifest: video_data.csv
  tokenizer: bsl_sp.model
  max_frames: 512
  normalize: true
  upper_body: true
split:
  kind: random-v1
  fracs: [0.8, 0.1, 0.1]
model:
  arch: transformer
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.1
  temporal_conv_layers: 2
  temporal_conv_kernel: 5
  ctc_head: false
pretrain:
  mask_ratio: 0.6
  min_span: 3
  max_span: 15
  epochs: 100
  batch_size: 32
  lr: 0.0005
  weight_decay: 0.0001
  clip_grad: 1.0
  device: auto
```

Note: `model.*` here matches `configs/full_ctc.yaml`'s `model.*` block
**except `ctc_head: false`** (pretraining never uses the CTC head — it isn't
part of `ENCODER_KEY_PREFIXES` and gets no gradient either way, so leaving it
`true` would just waste memory building an unused head; keep it `false` for
clarity). `d_model`/`nhead`/layer counts/TConv settings must stay identical
to `configs/full_ctc.yaml` so `encoder_pretrain.pt` is shape-compatible with
a `--init-encoder` finetune run built from that config.

- [ ] **Step 2: Implement the script**

```python
# scripts/pretrain_mkm.py
"""Masked keypoint modeling pretraining (WS-D D1).

Pretrains the encoder half of a SignTransformer (input_proj + TConv frontend
+ Transformer encoder) on unlabeled random-v1 train-split clips, saving an
encoder-only checkpoint scripts/train_sample.py can warm-start from via
--init-encoder.

    python scripts/pretrain_mkm.py --config configs/pretrain_mkm.yaml --device cuda
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # import torch before pandas (FAIL-002)

import argparse
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bornil.config import config_hash, load_config
from bornil.benchmark.splits import resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.models.build import build_model
from bornil.pretrain.checkpoint import save_encoder_checkpoint
from bornil.pretrain.mkm import MaskedSignEncoder
from scripts.train_sample import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_mkm.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (overrides config)")
    ap.add_argument("--split-root", default="splits")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["pretrain"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["pretrain"]["device"] = args.device
    out_dir = cfg["trial"]["out_dir"]
    device = resolve_device(cfg["pretrain"]["device"])

    man = pd.read_csv(os.path.join(out_dir, "sample_manifest.csv"))
    features = np.load(os.path.join(out_dir, "packed.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    tok = Tokenizer(cfg["data"]["tokenizer"])
    # Targets are loaded (dataset contract requires them) but never used in
    # the pretraining loss -- no text supervision happens here.
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]), split_root=args.split_root,
    )
    cfg["split"]["source"] = split_source

    rows = man[man["recording"].isin(set(sp["train"]))]
    pack_idx = rows["pack_idx"].tolist()
    targets = [targets_by_rec[r] for r in rows["recording"]]
    ds = SignDataset(features, lengths, pack_idx, targets,
                     normalize=cfg["data"]["normalize"], upper_body=cfg["data"]["upper_body"])
    collate = make_collate_fn(tok.pad_id)
    bs = LengthBucketBatchSampler(ds.item_lengths(), cfg["pretrain"]["batch_size"],
                                  shuffle=True, seed=cfg["trial"]["seed"])
    loader = DataLoader(ds, batch_sampler=bs, collate_fn=collate)

    dev_desc = device
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        dev_desc = f"cuda ({p.name}, {p.total_memory/1e9:.1f} GB)"
    print(f"device={dev_desc} | split={cfg['split']['kind']} train-only ({split_source}) "
          f"| clips={len(sp['train'])}", flush=True)

    backbone = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    enc = MaskedSignEncoder(
        backbone, mask_ratio=cfg["pretrain"]["mask_ratio"],
        min_span=cfg["pretrain"]["min_span"], max_span=cfg["pretrain"]["max_span"],
    ).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["pretrain"]["lr"],
                            weight_decay=cfg["pretrain"]["weight_decay"])

    h = config_hash(cfg)
    run_dir = os.path.join(out_dir, "runs", h)
    os.makedirs(run_dir, exist_ok=True)

    epochs = cfg["pretrain"]["epochs"]
    t0 = time.time()
    losses = []
    for ep in range(1, epochs + 1):
        enc.train()
        total, nb = 0.0, 0
        from tqdm import tqdm
        for x, _y, x_lengths, _y_lengths in tqdm(loader, desc=f"epoch {ep}/{epochs}",
                                                  unit="batch", leave=False, dynamic_ncols=True):
            x = x.to(device)
            x_lengths = x_lengths.to(device)
            opt.zero_grad()
            loss = enc.loss(x, x_lengths)
            loss.backward()
            if cfg["pretrain"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), cfg["pretrain"]["clip_grad"])
            opt.step()
            total += float(loss.item())
            nb += 1
        mean_loss = total / max(nb, 1)
        losses.append(mean_loss)
        elapsed = (time.time() - t0) / 60
        print(f"epoch {ep:3d}/{epochs} | recon_loss {mean_loss:.4f} | elapsed {elapsed:.1f}m",
              flush=True)

    save_encoder_checkpoint(os.path.join(run_dir, "encoder_pretrain.pt"), backbone)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "pretrain_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"recon_loss_by_epoch": losses, "train_min": (time.time() - t0) / 60}, f, indent=2)

    _append_pretrain_row(cfg, h, losses, device, (time.time() - t0) / 60, len(sp["train"]))
    print(f"run saved -> {run_dir}")
    print(f"encoder checkpoint -> {os.path.join(run_dir, 'encoder_pretrain.pt')}")


def _append_pretrain_row(cfg, h, losses, device, mins, n_train):
    row = (
        f"\n### EXP-{cfg['trial']['name']}-{h}  ({date.today().isoformat()})\n"
        f"- **Model:** MKM pretrain (encoder-only) d{cfg['model']['d_model']} "
        f"L{cfg['model']['num_encoder_layers']}   **Phase/Task:** {cfg['trial']['name']} (WS-D D1)\n"
        f"- **Data/Split:** {cfg['trial']['name']} {cfg['split']['kind']} train-only "
        f"(train {n_train}); source={cfg['split'].get('source', 'unknown')}\n"
        f"- **Config hash:** {h}   **Seed:** {cfg['trial']['seed']}\n"
        f"- **Compute:** {device}   **Train time:** {mins:.1f} min\n"
        f"- **Pretrain extras:** mask_ratio={cfg['pretrain']['mask_ratio']} "
        f"min_span={cfg['pretrain']['min_span']} max_span={cfg['pretrain']['max_span']}\n"
        f"- **Recon loss:** first={losses[0]:.4f} last={losses[-1]:.4f}\n"
        f"- **Checkpoint:** {cfg['trial']['out_dir']}/runs/{h}/encoder_pretrain.pt\n"
    )
    with open(os.path.join("docs", "logs", "EXPERIMENTS.md"), "a", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test on real data at low epoch count**

```bash
.venv/Scripts/python scripts/pretrain_mkm.py --config configs/pretrain_mkm.yaml --epochs 2 --device cuda
```

Expected: runs to completion in a few minutes (2 epochs over 14,390 clips),
prints `epoch 1/2 | recon_loss ...` then `epoch 2/2 | recon_loss ...` with
**epoch 2's loss lower than epoch 1's** (sanity, not the real gate — the
real gate with the full epoch count is Task 9), writes
`data_full/runs/<hash>/encoder_pretrain.pt` + `config.json` +
`pretrain_metrics.json`, and appends an `EXPERIMENTS.md` row. If this
crashes, fix it before moving on — do not proceed to Task 8 with a broken
CLI. **Do not commit the smoke-run's `data_full/runs/<hash>/` output or its
`EXPERIMENTS.md` row** — `data_full/runs/` is gitignored already (matches
every other run's convention), but the `EXPERIMENTS.md` append is not:
revert it with `git checkout -- docs/logs/EXPERIMENTS.md` after confirming
the smoke test worked, since the real Task 9 run will append the row that
actually belongs in history.

- [ ] **Step 4: Commit the script and config**

```bash
git checkout -- docs/logs/EXPERIMENTS.md
git add scripts/pretrain_mkm.py configs/pretrain_mkm.yaml
git commit -m "feat(pretrain): pretrain_mkm.py CLI for masked keypoint modeling"
```

---

## Task 8: `--init-encoder` and `--label-fraction` flags on `train_sample.py`

**Files:**
- Modify: `scripts/train_sample.py`

**Read first:** the full file (already read for Task 7). You are adding two
independent, composable CLI flags — most finetune runs in this plan use one
or the other, not both, but nothing prevents combining them.

**No new test file** — mirrors Task 7's reasoning (orchestration glue over
Task 5's already-tested `load_encoder_checkpoint` and Task 6's already-tested
`filter_split_by_fraction`); validated by the real runs in Tasks 9–10. The
specific property that matters for `--init-encoder` (encoder weights change,
decoder/CTC-head weights don't) is already covered by
`test_decoder_and_heads_unaffected_by_load` in Task 5 — that test exercises
`load_encoder_checkpoint` directly, which is the entire body of the
`--init-encoder` branch added here, so a second CLI-level test would just be
re-testing the same function through a heavier path (full config load, full
`build_model`) without adding coverage. Do not skip Task 5's test thinking
"I'll cover it here instead" — this task deliberately relies on it already
existing.

- [ ] **Step 1: Add `--init-encoder`**

In `scripts/train_sample.py`, add the import:

```python
from bornil.pretrain.checkpoint import load_encoder_checkpoint
```

Add the CLI arg (near the other `ap.add_argument` calls):

```python
    ap.add_argument("--init-encoder", default=None,
                    help="path to an encoder_pretrain.pt from pretrain_mkm.py; "
                         "warm-starts input_proj/temporal_frontend/transformer.encoder")
```

After the line `model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)`, add:

```python
    if args.init_encoder:
        loaded = load_encoder_checkpoint(model, args.init_encoder, device)
        print(f"init_encoder: loaded {len(loaded)} tensors from {args.init_encoder}", flush=True)
        cfg.setdefault("train", {})["init_encoder"] = args.init_encoder
```

(Putting `init_encoder` into `cfg["train"]` means `config_hash(cfg)` differs
from the same config without it — pretrained and from-scratch runs land in
different `run_dir`s automatically, no collision risk.)

In `_append_experiment_row`, extend the "Training extras" line so pretrain
provenance is visible in `EXPERIMENTS.md` without opening `config.json`:

```python
    f"- **Training extras:** early_eos_penalty={cfg['train'].get('early_eos_penalty', 0.0)} "
    f"ctc_weight={cfg['train'].get('ctc_weight', 0.0)} "
    f"init_encoder={cfg['train'].get('init_encoder', 'none')} "
    f"label_fraction={cfg.get('split', {}).get('label_fraction', 1.0)}\n"
```

(replace the existing single-line f-string for "Training extras" with this
one — same key, longer value.)

- [ ] **Step 2: Add `--label-fraction`**

Add the CLI arg:

```python
    ap.add_argument("--label-fraction", type=float, default=None,
                    help="subsample the train split to this fraction of clips "
                         "(data-efficiency curve; val/test unaffected)")
    ap.add_argument("--label-fraction-seed", type=int, default=42)
```

Add the import:

```python
from bornil.benchmark.splits import filter_split_by_fraction
```

Right after the existing `sp, split_source = _apply_target_length_filter(cfg, man, sp, split_source)` line, add:

```python
    if args.label_fraction is not None:
        sp = filter_split_by_fraction(sp, args.label_fraction, seed=args.label_fraction_seed)
        split_source = f"{split_source}|label_fraction={args.label_fraction}"
        cfg.setdefault("split", {})["label_fraction"] = args.label_fraction
        print(f"label_fraction={args.label_fraction} -> train clips={len(sp['train'])}", flush=True)
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: same pass count as before this task plus no failures — this task
adds no new tests, so the count should be identical to Task 6's post-commit
count (confirm by comparing against the previous task's test run output; if
you didn't capture it, run the full suite now and treat this number as the
new baseline for the rest of the plan).

- [ ] **Step 4: Smoke-test both flags on real data at low epoch count**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/pretrain_mkm.yaml --epochs 1 --device cuda
```

This will fail on purpose — `configs/pretrain_mkm.yaml` has no `train:`
block (it uses `pretrain:`), which is expected; it's the wrong config for
this script. Instead smoke-test against a real training config:

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --epochs 1 --split random-v1 --label-fraction 0.05 --out-dir smoke_label_frac --device cuda
```

Expected: prints `label_fraction=0.05 -> train clips=...` (roughly 5% of
14,390 ≈ 720), trains 1 epoch, completes. Then smoke-test `--init-encoder`
using the checkpoint Task 7's smoke test produced (if you reverted
`EXPERIMENTS.md` but kept `data_full/runs/<hash>/encoder_pretrain.pt` — if
you deleted it, rerun `pretrain_mkm.py --epochs 2` once more first):

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --epochs 1 --split random-v1 --init-encoder data_full/runs/<pretrain-smoke-hash>/encoder_pretrain.pt --out-dir smoke_init_encoder --device cuda
```

Expected: prints `init_encoder: loaded N tensors from ...` with N > 0,
trains 1 epoch, completes. **Delete `smoke_label_frac/` and
`smoke_init_encoder/` run output and any stray smoke `EXPERIMENTS.md`
rows after confirming both work** (`git checkout -- docs/logs/EXPERIMENTS.md`
again if either smoke run appended to it) — these are throwaway
verification runs, not real results.

- [ ] **Step 5: Commit**

```bash
git checkout -- docs/logs/EXPERIMENTS.md
git add scripts/train_sample.py
git commit -m "feat(train): --init-encoder and --label-fraction flags"
```

---

## Task 9: Real pretraining run (gate check)

**Files:** none (this task runs code, doesn't write it).

- [ ] **Step 1: ⏱ Run real pretraining**

```bash
.venv/Scripts/python scripts/pretrain_mkm.py --config configs/pretrain_mkm.yaml --device cuda
```

This trains 100 epochs over 14,390 clips — expect on the order of an hour or
more (measure the smoke test's per-epoch time from Task 7 Step 3 and
extrapolate; do not assume). Let it run to completion.

- [ ] **Step 2: Check the gate — reconstruction loss decreased and is finite**

Read `data_full/runs/<pretrain-hash>/pretrain_metrics.json`. Compare
`recon_loss_by_epoch[0]` (or an early-epoch value, e.g. index 4, since epoch
1 can be noisy) against the last value.

- **If the loss decreased substantially (not just noise) and is finite:**
  gate passes, continue to Step 3.
- **If the loss did not decrease, is flat, or is NaN/inf:** **STOP. Do not
  proceed to Task 10 or 11.** Log to `docs/logs/FAILURES.md` using the
  existing template (see FAIL-004/FAIL-005 for the exact format this repo
  uses), with a hypothesis (e.g., mask ratio too aggressive, learning rate
  wrong, mask token not actually reaching the encoder — check by printing
  `enc.mask_token.grad` after one backward pass; a `None` or all-zero
  gradient means the mask substitution isn't wired into the graph, most
  likely a `torch.where` broadcasting bug). Fix the root cause, rerun this
  task from Step 1, and only continue once the gate passes. Note this in the
  plan by appending a note under this task's checkboxes documenting what
  happened.

- [ ] **Step 3: Log the pretraining run**

Add a `docs/logs/CHANGELOG.md` entry (newest-first, matching the file's
existing format exactly — copy the header/What/Why/Impact/Commit structure
from any recent entry) summarizing: config hash, epoch count, first vs. last
reconstruction loss, wall-clock time, and the checkpoint path. State plainly
whether the gate passed.

- [ ] **Step 4: Commit**

```bash
git add docs/logs/EXPERIMENTS.md docs/logs/CHANGELOG.md
git commit -m "feat(pretrain): MKM pretraining run on random-v1 train split"
```

(If the gate failed and you logged to `FAILURES.md` instead, commit that
file too, with a commit message reflecting the negative result, e.g.
`docs(failures): MKM pretraining gate failed -- <one-line reason>`, and
**stop the plan here** — do not execute Tasks 10–12 against a
pretrain checkpoint that failed its own sanity gate.)

---

## Task 10: Core-claim comparison on `signerindep-v1`

**Files:** none (runs code).

- [ ] **Step 1: ⏱ Finetune with pretrained-init on `signerindep-v1`**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split signerindep-v1 --init-encoder data_full/runs/<pretrain-hash>/encoder_pretrain.pt --device cuda
```

Replace `<pretrain-hash>` with Task 9's actual run hash. Expect roughly
TConv+CTC-scale training time (the existing `40cffb9e62bb` reference run's
`EXPERIMENTS.md` row has its exact train time — check it and expect
something in that neighborhood). Confirm on completion:
`docs/logs/EXPERIMENTS.md` has a new row, and note its config hash — call it
`<arm-a-hash>`.

- [ ] **Step 2: Significance test vs. the existing from-scratch run**

```bash
.venv/Scripts/python scripts/results_significance.py --run "scratch=data_full/runs/40cffb9e62bb" --run "pretrained=data_full/runs/<arm-a-hash>" --baseline scratch --metrics bleu4 chrf++ --n 1000 --out docs/logs/SIGNIFICANCE_MKM_VS_SCRATCH_SIGNERINDEP.md
```

- [ ] **Step 3: Diagnostics**

```bash
.venv/Scripts/python scripts/prediction_diagnostics.py --pred data_full/runs/<arm-a-hash>/predictions.csv --out-json data_full/runs/<arm-a-hash>/diagnostics.json
```

Read the resulting `diagnostics.json` (unique-hypothesis rate, mean
hyp/ref length, repeated-hypothesis rate) — same fields used for every other
model's Table 5 entry in `docs/logs/RESULTS_OVERVIEW.md`.

- [ ] **Step 4: Record the verdict — do not skip this regardless of outcome**

Whether chrF++ significantly beats scratch or not, this is a real result.
Add it to `docs/logs/RESULTS_OVERVIEW.md` (a new row in Table 2, or a new
small table if you judge Table 2's shape doesn't fit a pretrain/scratch pair
cleanly — use your judgment, but do not omit the result because it's
negative). Add a `docs/logs/CHANGELOG.md` entry. If chrF++ does **not**
significantly beat scratch, also log to `docs/logs/FAILURES.md` (repo
convention — see the design doc's "Failure modes" section: **do not**
retroactively tune the mask ratio or any other pretraining hyperparameter
against this eval split to try to force a win; that invalidates the
significance test).

- [ ] **Step 5: Commit**

```bash
git add docs/logs/EXPERIMENTS.md docs/logs/RESULTS_OVERVIEW.md docs/logs/CHANGELOG.md docs/logs/SIGNIFICANCE_MKM_VS_SCRATCH_SIGNERINDEP.md
git commit -m "feat(pretrain): MKM vs scratch core-claim comparison on signerindep-v1"
```

(Add `docs/logs/FAILURES.md` to the commit too if you logged a negative
result there.)

---

## Task 11: Data-efficiency curve on `random-v1`

**Files:** none (runs code).

**Order matters less than completeness here** — run all 7 in whichever
order is convenient, but do not report partial curve results as if they
were the whole curve. If interrupted mid-task, resume by checking
`docs/logs/EXPERIMENTS.md` for which of the 7 config/fraction combinations
already have a row before rerunning any of them (rerunning is wasted GPU
time, not incorrect — `config_hash` makes each fraction+arm combination a
distinct run_dir, so accidental reruns don't collide, but avoid them anyway).

- [ ] **Step 1: ⏱ Pretrained-init runs at 10%, 25%, 50%, 100% labels**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 0.10 --init-encoder data_full/runs/<pretrain-hash>/encoder_pretrain.pt --device cuda
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 0.25 --init-encoder data_full/runs/<pretrain-hash>/encoder_pretrain.pt --device cuda
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 0.50 --init-encoder data_full/runs/<pretrain-hash>/encoder_pretrain.pt --device cuda
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 1.00 --init-encoder data_full/runs/<pretrain-hash>/encoder_pretrain.pt --device cuda
```

Each smaller-fraction run should train faster per epoch (fewer batches) —
verify empirically rather than assuming, per the design doc's compute-budget
note; if a small-fraction run looks like it's undertraining (loss still
dropping fast at the config's fixed epoch count), that's a real finding to
note in the write-up, not something to silently fix by changing epochs
mid-curve (changing epochs per-fraction would confound the comparison — if
you do decide it's necessary, apply the same epoch policy to both arms at
that fraction, and say so explicitly in the write-up).

- [ ] **Step 2: ⏱ From-scratch runs at 10%, 25%, 50% labels (100% is `b0a590ea3317`, already done)**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 0.10 --device cuda
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 0.25 --device cuda
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --split random-v1 --label-fraction 0.50 --device cuda
```

- [ ] **Step 3: Collect all 8 curve points into one table**

You now have 8 (fraction, arm) → chrF++/BLEU-4 points: pretrained ×
{10,25,50,100}% (4 new runs from Step 1) and scratch × {10,25,50,100}% (3
new runs from Step 2 plus the existing `b0a590ea3317` at 100%). Read each
run's `metrics.json` and tabulate.

- [ ] **Step 4: Significance per fraction (optional but recommended)**

For each fraction, run `results_significance.py` pretrained vs. scratch at
that fraction (same pattern as Task 10 Step 2, 4 invocations). This tells
you not just the curve shape but whether each individual gap is real.

- [ ] **Step 5: Write up the curve**

Add a new table to `docs/logs/RESULTS_OVERVIEW.md` (e.g. "Table 6:
Data-Efficiency Curve (WS-D D1)") with columns `Label fraction | Arm | Run |
Test rows | BLEU-4 | chrF++ | vs-other-arm p-value`. State the verdict
plainly: does the pretrained-vs-scratch gap widen as label fraction shrinks
(supports the low-resource SSL story), stay constant (SSL helps but isn't
specifically a low-resource effect), or not show a consistent gap at all
(the honest negative — see the design doc's "Failure modes" section, report
it, don't omit it). Add a `docs/logs/CHANGELOG.md` entry summarizing the
whole curve in one entry (not one per run — that would be 7 near-duplicate
entries; the design doc's own compute-budget section anticipated this as
one deliverable).

- [ ] **Step 6: Commit**

```bash
git add docs/logs/EXPERIMENTS.md docs/logs/RESULTS_OVERVIEW.md docs/logs/CHANGELOG.md docs/logs/SIGNIFICANCE_MKM_CURVE_*.md
git commit -m "feat(pretrain): MKM data-efficiency curve on random-v1"
```

---

## Task 12: Final verification + report

**Files:** none.

- [ ] **Step 1: Full test suite**

```bash
.venv/Scripts/python -m pytest -q
```

Expected: all pass, exit 0. If anything fails, fix it before declaring this
plan complete — do not report success with a failing suite.

- [ ] **Step 2: Confirm every real run in this plan has an `EXPERIMENTS.md` row**

Count: 1 pretrain run (Task 9) + 1 signerindep-v1 comparison run (Task 10) +
7 curve runs (Task 11) = 9 new rows total, plus whatever smoke-test rows
you correctly reverted in Tasks 7–8 (confirm those did **not** end up
committed — `git log --oneline -- docs/logs/EXPERIMENTS.md` should show
clean, real-run-only commits).

- [ ] **Step 3: Final report**

Produce a short summary (this is the one deliverable meant for a human to
read without opening every file) covering:
- Pretraining gate result (pass/fail, loss numbers).
- Core-claim result on `signerindep-v1`: chrF++ pretrained vs. scratch,
  p-value, verdict.
- Data-efficiency curve: the full 8-point table and whether the gap widens,
  stays flat, or is inconsistent at low label fractions.
- Any `docs/logs/FAILURES.md` entries added by this plan.
- Full list of commits made under this plan (`git log --oneline` since the
  plan's Task 1 commit).
- Full suite status (pass count).
- **Explicit statement of whether D1 is a supported win for the WS-D
  comparison** (per the master plan §8, D1/D2/D3 get compared head-to-head
  to pick the paper's flagship method) — this plan's job is to produce that
  verdict, not to guarantee a positive one.
