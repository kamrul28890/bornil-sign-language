# WS-D / D2 — Pure-PyTorch Selective-SSM (Mamba) Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Design doc:
> `docs/superpowers/specs/2026-07-28-ws-d-d2-mamba-ssm-design.md` — read it
> once before starting; this plan operationalizes it and does not repeat its
> reasoning. Follow-on to `docs/plan/2026-07-13-bornil-implementation-plan.md`
> §8.2 (D2). D1 (SSL pretraining) closed negative — see
> `docs/logs/FAILURES.md` FAIL-006/FAIL-007 — this is independent of that
> work and does not depend on it.

**Goal:** implement a Mamba-style selective state-space encoder in pure
PyTorch (no `mamba-ssm` dependency — confirmed unbuildable on this Windows
machine, see design doc) and test it against the current best encoder
(TConv2k5+Transformer+CTC, `b0a590ea3317`, chrF++ 10.913) on `random-v1`,
both overall and on a length-stratified cut.

**Architecture:** a full Mamba block (RMSNorm → gated linear projection →
depthwise causal conv → input-dependent selective scan → gate → output
projection → residual), stacked 4 deep, feeding the same shared Transformer
decoder + optional CTC head every other encoder in this repo uses. The
selective scan is a sequential Python loop over timesteps (not a parallel
associative scan) — simpler to implement and test correctly; see the design
doc's "Compute budget" for the explicit uncertainty this introduces, which
Task 6 measures directly.

**Tech Stack:** pure PyTorch (`torch.nn.RMSNorm`, confirmed available in
this repo's torch 2.6.0+cu124). No new dependencies.

## Global Constraints

- **Windows import order:** every script entry point imports `bornil.runtime`
  before pandas (FAIL-002). Use `.venv/Scripts/python`. PowerShell has no
  `&&` — chain with `;`, or run one command per line.
- **No behavior change to any existing model, config, or script.** This
  plan is additive: one new module (`bornil/models/mamba.py`), one new
  script (`scripts/length_stratified_report.py`), one new config
  (`configs/full_mamba.yaml`), and one dispatch branch added to
  `bornil/models/build.py`'s existing `if/elif` chain (mirrors how
  `"bilstm"` and `"stgcn"` were added — do not restructure that function).
- **Do not modify the locked metric protocol** (`bornil/benchmark/metrics.py`).
- **Frozen splits only for real runs** — never `--no-frozen-splits` outside tests.
- **Logging discipline:** every real training run gets an `EXPERIMENTS.md`
  row (automatic via `train_sample.py`'s existing `_append_experiment_row`
  — `SignMamba` needs no changes there, it already reads `cfg["model"].get("arch")`
  generically... **check this**: `_append_experiment_row` in
  `scripts/train_sample.py` currently has an `if/elif` for `arch == "bilstm"`
  / `arch == "stgcn"` / else (transformer-labeled) — Task 5 adds a
  `"mamba"` branch there too, otherwise a Mamba run would be mislabeled
  "SignTransformer" in `EXPERIMENTS.md`). Meaningful changes →
  `docs/logs/CHANGELOG.md`. Negative or surprising results → `docs/logs/FAILURES.md`,
  using the exact template already in that file (see FAIL-005/006/007).
  **A negative result is a completed task, not a failed one** — do not tune
  `d_state`/`expand`/`conv_kernel`/`num_mamba_layers` against eval results to
  manufacture a win; those are fixed once (Task 5) and not revisited based
  on Task 7/8's outcome.
- **TDD:** write the failing test before the implementation, for every task
  that has one.
- **Commit after every task.**
- **This plan is resumable.** Before starting work in a fresh session, read
  this file top to bottom and find the first unchecked `- [ ]` box, and
  cross-check against `git log --oneline` — a task with a matching commit
  already landed even if a checkbox wasn't ticked; trust the commit history
  over the checkboxes if they ever disagree. Tick checkboxes as you go and
  include that edit in each task's commit.
- **Long-running steps are marked ⏱.** Task 6's smoke test is quick (a
  couple minutes); Task 6's full run and Task 7 could be anywhere from
  ~45 minutes to several hours — **this is exactly what Task 6's timing
  measurement exists to pin down before you commit to the full run.** Do
  not reduce epoch count or dataset size on the real run to make it faster.

---

## Task 1: Selective scan core recurrence (`bornil/models/mamba.py`)

**Files:**
- Create: `bornil/models/mamba.py`
- Test: `tests/models/test_selective_scan.py`

**Read first:** the design doc's "Architecture" section for the exact
recurrence. Restated here precisely since this is the one piece of math in
the whole plan: for batch `b`, channel `c` (of `d_inner` channels), state
index `n` (of `d_state`):

```
h[b,c,n]_t = exp(delta[b,t,c] * A[c,n]) * h[b,c,n]_{t-1} + delta[b,t,c] * B[b,t,n] * x[b,t,c]
y[b,t,c]   = sum_n( C[b,t,n] * h[b,c,n]_t ) + D[c] * x[b,t,c]
```

`h` starts at zero. This is a simplified Euler-style discretization (`Δ·B`
directly, not the Mamba paper's exact zero-order-hold term) — deliberate
simplification for implementation simplicity, noted here so it's never
mistaken for an exact reproduction of the paper's math.

**Interfaces:**
- Produces: `selective_scan(x: Tensor(B,T,d_inner), delta: Tensor(B,T,d_inner), A: Tensor(d_inner,d_state), B: Tensor(B,T,d_state), C: Tensor(B,T,d_state), D: Tensor(d_inner,)) -> Tensor(B,T,d_inner)` — pure function, sequential Python loop over `T` internally, fully differentiable (plain tensor ops, no `.item()`/no-grad inside the loop).

- [x] **Step 1: Write the failing tests**

```python
# tests/models/test_selective_scan.py
import math

import torch

from bornil.models.mamba import selective_scan


def _brute_force_reference(x, delta, A, B, C, D):
    """Independent, unvectorized reference (plain Python floats, a second
    code path from `selective_scan`) for a single batch element.

    x, delta: (T, d_inner). A: (d_inner, d_state). B, C: (T, d_state).
    D: (d_inner,). Returns a (T, d_inner) list of lists of floats.
    """
    T, d_inner = x.shape
    d_state = A.shape[1]
    h = [[0.0] * d_state for _ in range(d_inner)]
    y = []
    for t in range(T):
        y_t = []
        for c in range(d_inner):
            dt = delta[t, c].item()
            xc = x[t, c].item()
            y_c = D[c].item() * xc
            for n in range(d_state):
                dA = math.exp(dt * A[c, n].item())
                dBx = dt * B[t, n].item() * xc
                h[c][n] = dA * h[c][n] + dBx
                y_c += C[t, n].item() * h[c][n]
            y_t.append(y_c)
        y.append(y_t)
    return y


def test_matches_independent_brute_force_reference():
    torch.manual_seed(0)
    T, d_inner, d_state = 6, 4, 3
    x = torch.randn(1, T, d_inner)
    delta = torch.rand(1, T, d_inner) * 0.5 + 0.01   # positive, like softplus output
    A = -torch.rand(d_inner, d_state) - 0.1           # negative, like -exp(A_log)
    B = torch.randn(1, T, d_state)
    C = torch.randn(1, T, d_state)
    D = torch.randn(d_inner)

    y = selective_scan(x, delta, A, B, C, D)
    ref = _brute_force_reference(x[0], delta[0], A, B[0], C[0], D)

    assert y.shape == (1, T, d_inner)
    for t in range(T):
        for c in range(d_inner):
            assert abs(y[0, t, c].item() - ref[t][c]) < 1e-4


def test_zero_delta_freezes_state():
    torch.manual_seed(1)
    T, d_inner, d_state = 5, 2, 3
    x = torch.randn(1, T, d_inner)
    delta = torch.zeros(1, T, d_inner)   # delta==0 everywhere
    A = -torch.rand(d_inner, d_state) - 0.1
    B = torch.randn(1, T, d_state)
    C = torch.randn(1, T, d_state)
    D = torch.zeros(d_inner)   # isolate the state contribution (no skip term)

    y = selective_scan(x, delta, A, B, C, D)
    # delta==0 -> dA=exp(0)=1, dBx=0 -> h never changes from its zero init
    # -> C . h stays 0 at every t (D is zeroed out above).
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_gradients_are_finite():
    torch.manual_seed(2)
    T, d_inner, d_state = 8, 4, 3
    x = torch.randn(2, T, d_inner, requires_grad=True)
    delta_raw = torch.randn(2, T, d_inner, requires_grad=True)
    delta = torch.nn.functional.softplus(delta_raw)
    A = (-torch.rand(d_inner, d_state) - 0.1).requires_grad_()
    B = torch.randn(2, T, d_state, requires_grad=True)
    C = torch.randn(2, T, d_state, requires_grad=True)
    D = torch.randn(d_inner, requires_grad=True)

    y = selective_scan(x, delta, A, B, C, D)
    y.sum().backward()

    for t in (x, delta_raw, A, B, C, D):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/models/test_selective_scan.py -v`
Expected: FAIL — `No module named 'bornil.models.mamba'`.

- [x] **Step 3: Implement**

```python
# bornil/models/mamba.py
"""Pure-PyTorch Mamba-style selective-SSM encoder (WS-D D2, master plan §8.2).

The official `mamba-ssm` package ships custom CUDA kernels with no Windows
wheels; building it from source needs the CUDA Toolkit + MSVC, both absent
on this dev machine (verified during design). This module reimplements the
selective-scan recurrence in plain PyTorch: a Python loop over timesteps,
vectorized across batch/channel/state at each step. Simpler to get right
and test than a parallel associative scan; clips cap at 512 frames and this
is a research comparison, not a speed benchmark (see docs/logs/CHANGELOG.md
and the design doc's "Compute budget" for the actual measured cost).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def selective_scan(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                   B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Sequential selective-SSM scan.

    x, delta: (B_batch, T, d_inner). A: (d_inner, d_state) (should be
    negative for a stable decay). B, C: (B_batch, T, d_state). D: (d_inner,).
    Returns y: (B_batch, T, d_inner).
    """
    B_batch, T, d_inner = x.shape
    d_state = A.shape[1]
    h = x.new_zeros(B_batch, d_inner, d_state)
    ys = []
    for t in range(T):
        dt = delta[:, t, :]                          # (B_batch, d_inner)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))          # (B_batch, d_inner, d_state)
        dBx = dt.unsqueeze(-1) * B[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1)
        h = dA * h + dBx
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D.unsqueeze(0) * x[:, t, :]
        ys.append(y_t)
    return torch.stack(ys, dim=1)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_selective_scan.py -v`
Expected: 3 passed.

- [x] **Step 5: Commit**

```bash
git add bornil/models/mamba.py tests/models/test_selective_scan.py
git commit -m "feat(models): pure-PyTorch selective-SSM scan (WS-D D2)"
```

---

## Task 2: `MambaBlock` (the full block around the scan)

**Files:**
- Modify: `bornil/models/mamba.py` (add to it)
- Test: `tests/models/test_mamba_block.py`

**Interfaces:**
- Consumes: `selective_scan` (Task 1).
- Produces: `class MambaBlock(nn.Module)`:
  - `__init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_kernel: int = 4)`
  - `forward(self, x: Tensor(B,T,d_model), pad_mask: Tensor(B,T) bool = None) -> Tensor(B,T,d_model)` — `pad_mask` True at padded positions; residual-added internally (caller passes the block's input, gets the post-residual output back).

- [x] **Step 1: Write the failing tests**

```python
# tests/models/test_mamba_block.py
import torch

from bornil.models.mamba import MambaBlock


def test_output_shape():
    block = MambaBlock(d_model=32, d_state=8, expand=2, conv_kernel=4)
    x = torch.randn(3, 20, 32)
    out = block(x)
    assert out.shape == (3, 20, 32)


def test_padding_never_affects_real_positions():
    """Two batches identical except for their padded tail must produce
    identical output at every real (non-padded) position -- this is the
    property the whole delta=0 padding design depends on; test the full
    block, not just the scan in isolation."""
    torch.manual_seed(0)
    block = MambaBlock(d_model=16, d_state=8, expand=2, conv_kernel=4)
    block.eval()   # no dropout in this block, but eval() is cheap insurance

    x = torch.randn(1, 20, 16)
    real_len = 12
    x_a = x.clone()
    x_b = x.clone()
    x_b[:, real_len:] = torch.randn(1, 20 - real_len, 16)   # different padding garbage

    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, real_len:] = True

    out_a = block(x_a, pad_mask=mask)
    out_b = block(x_b, pad_mask=mask)

    assert torch.allclose(out_a[:, :real_len], out_b[:, :real_len], atol=1e-5)


def test_residual_present():
    """With all weights zeroed except would still return the residual input
    (sanity that this is a residual block, not a bare replacement)."""
    torch.manual_seed(1)
    block = MambaBlock(d_model=8, d_state=4, expand=2, conv_kernel=4)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.randn(2, 10, 8)
    out = block(x)
    # zeroed in_proj -> z=0 -> SiLU(0)=0 -> gated branch is exactly 0 ->
    # out_proj(0)=0 (zeroed bias/weight) -> output collapses to the residual.
    assert torch.allclose(out, x, atol=1e-5)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/models/test_mamba_block.py -v`
Expected: FAIL — `ImportError: cannot import name 'MambaBlock'`.

- [x] **Step 3: Implement**

Append to `bornil/models/mamba.py`:

```python
class MambaBlock(nn.Module):
    """One Mamba block: RMSNorm -> gated proj -> causal depthwise conv ->
    input-dependent selective scan -> gate -> output proj -> residual.

    No TConv frontend, no positional encoding needed elsewhere in the model
    that uses this block -- the scan's recurrence is inherently sequential/
    order-aware (see SignMamba below)."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                conv_kernel: int = 4):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                              groups=d_inner, padding=conv_kernel - 1)
        self.delta_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)
        # A_log parameterizes A = -exp(A_log) < 0, a stable decay, initialized
        # to the standard Mamba S4D-real init (rows 1..d_state, log-spaced).
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        xz = self.in_proj(h)
        x_in, z = xz.chunk(2, dim=-1)
        # Causal depthwise conv: symmetric padding=(k-1) then keep only the
        # first T outputs -- output position i only ever sees input <= i.
        x_conv = self.conv(x_in.transpose(1, 2))[..., :x_in.size(1)].transpose(1, 2)
        x_conv = torch.nn.functional.silu(x_conv)

        delta = torch.nn.functional.softplus(self.delta_proj(x_conv))
        if pad_mask is not None:
            delta = delta.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        Bt = self.B_proj(x_conv)
        Ct = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)

        y = selective_scan(x_conv, delta, A, Bt, Ct, self.D)
        y = y * torch.nn.functional.silu(z)
        out = self.out_proj(y)
        return residual + out
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_mamba_block.py -v`
Expected: 3 passed.

- [x] **Step 5: Commit**

```bash
git add bornil/models/mamba.py tests/models/test_mamba_block.py
git commit -m "feat(models): MambaBlock (gated selective-SSM block)"
```

---

## Task 3: `SignMamba` (full model, decode-side copied from `SignSTGCN`)

**Files:**
- Modify: `bornil/models/mamba.py` (add to it)
- Test: `tests/models/test_mamba.py`

**Read first:** `bornil/models/stgcn.py`'s `SignSTGCN` class in full. Its
`_decode_memory`, `forward`, `ctc_logits`, `forward_with_ctc`, `generate`,
`beam_search` methods are **copied verbatim below** — this is deliberate,
not laziness: every encoder in this codebase shares this exact decoder
plumbing by design (see `bornil/models/transformer.py`,
`bornil/models/bilstm.py`, `bornil/models/stgcn.py` — all four now,
including this one, have the same six methods with the same bodies, only
`encode()` differs per architecture).

**Interfaces:**
- Consumes: `MambaBlock` (Task 2).
- Produces: `class SignMamba(nn.Module)`:
  - `__init__(self, input_dim: int, vocab_size: int, d_model: int = 256, nhead: int = 8, num_mamba_layers: int = 4, d_state: int = 16, expand: int = 2, conv_kernel: int = 4, num_decoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1, pad_id: int = 0, use_ctc: bool = False)`
  - `encode(self, src, src_key_padding_mask=None) -> Tensor(B,S,d_model)`
  - `src_padding_mask`, `_decode_memory`, `forward`, `ctc_logits`, `forward_with_ctc`, `generate`, `beam_search` — identical contract to `SignSTGCN`.

- [x] **Step 1: Write the failing tests**

```python
# tests/models/test_mamba.py
"""Tests for SignMamba (WS-D D2). Mirrors tests/models/test_stgcn.py."""
import torch

from bornil.models.mamba import SignMamba


def _tiny_model(vocab=20, use_ctc=False):
    torch.manual_seed(0)
    return SignMamba(
        input_dim=12, vocab_size=vocab, d_model=32, nhead=4,
        num_mamba_layers=2, d_state=8, expand=2, conv_kernel=4,
        num_decoder_layers=2, dim_feedforward=64, dropout=0.0, pad_id=0,
        use_ctc=use_ctc,
    )


def test_forward_output_shape():
    model = _tiny_model()
    src = torch.randn(2, 15, 12)
    tgt_in = torch.randint(1, 20, (2, 6))
    logits = model(src, tgt_in)
    assert logits.shape == (2, 6, 20)


def test_encode_output_shape():
    model = _tiny_model()
    src = torch.randn(2, 15, 12)
    memory = model.encode(src)
    assert memory.shape == (2, 15, 32)


def test_ctc_logits_requires_use_ctc():
    model = _tiny_model(use_ctc=False)
    src = torch.randn(1, 10, 12)
    try:
        model.ctc_logits(src)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_ctc_logits_shape_when_enabled():
    model = _tiny_model(use_ctc=True)
    src = torch.randn(2, 10, 12)
    logits = model.ctc_logits(src)
    assert logits.shape == (2, 10, 20)


def test_generate_returns_token_lists():
    model = _tiny_model()
    src = torch.randn(2, 10, 12)
    out = model.generate(src, bos_id=1, eos_id=2, max_len=5)
    assert len(out) == 2
    assert all(isinstance(o, list) for o in out)


def test_overfit_tiny():
    """Standing overfit-tiny gate (plan §4.6): the model must memorize a
    couple of (keypoint-sequence -> subword-sequence) pairs, proving the
    encoder->decoder->loss path is wired correctly before any GPU run."""
    torch.manual_seed(0)
    model = _tiny_model(vocab=15)
    src = torch.randn(2, 12, 12)
    tgt = torch.randint(1, 15, (2, 7))
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    first_loss = None
    for _ in range(150):
        opt.zero_grad()
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, 15), tgt_out.reshape(-1))
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss * 0.1
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/models/test_mamba.py -v`
Expected: FAIL — `ImportError: cannot import name 'SignMamba'`.

- [x] **Step 3: Implement**

Append to `bornil/models/mamba.py`. Add these imports at the top of the
file alongside the existing ones:

```python
from bornil.models.transformer import PositionalEncoding, causal_mask
```

```python
class SignMamba(nn.Module):
    """Selective-SSM encoder + shared Transformer decoder (WS-D D2)."""

    def __init__(self, input_dim: int, vocab_size: int, d_model: int = 256,
                nhead: int = 8, num_mamba_layers: int = 4, d_state: int = 16,
                expand: int = 2, conv_kernel: int = 4,
                num_decoder_layers: int = 4, dim_feedforward: int = 1024,
                dropout: float = 0.1, pad_id: int = 0, use_ctc: bool = False):
        super().__init__()
        if num_mamba_layers < 1:
            raise ValueError("num_mamba_layers must be >= 1")
        self.pad_id = pad_id
        self.d_model = d_model
        self.use_ctc = use_ctc

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand=expand, conv_kernel=conv_kernel)
            for _ in range(num_mamba_layers)
        ])
        self.encoder_norm = nn.RMSNorm(d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.ctc_head = nn.Linear(d_model, vocab_size) if use_ctc else None

    def src_padding_mask(self, src_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        ar = torch.arange(seq_len, device=src_lengths.device)[None, :]
        return ar >= src_lengths[:, None]

    def encode(self, src: torch.Tensor,
              src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(src) * math.sqrt(self.d_model)
        for block in self.blocks:
            x = block(x, pad_mask=src_key_padding_mask)
        return self.encoder_norm(x)

    def _decode_memory(self, memory: torch.Tensor, tgt_in: torch.Tensor,
                       src_key_padding_mask: torch.Tensor = None,
                       tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        t = self.pos(self.tok_embed(tgt_in) * math.sqrt(self.d_model))
        h = self.decoder(
            t, memory,
            tgt_mask=causal_mask(tgt_in.size(1)).to(tgt_in.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask)
        return self._decode_memory(
            memory, tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def ctc_logits(self, src: torch.Tensor,
                   src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        if self.ctc_head is None:
            raise RuntimeError("SignMamba built with use_ctc=False has no CTC head")
        return self.ctc_head(self.encode(src, src_key_padding_mask))

    def forward_with_ctc(self, src: torch.Tensor, tgt_in: torch.Tensor,
                         src_key_padding_mask: torch.Tensor = None,
                         tgt_key_padding_mask: torch.Tensor = None):
        if self.ctc_head is None:
            raise RuntimeError("SignMamba built with use_ctc=False has no CTC head")
        memory = self.encode(src, src_key_padding_mask)
        logits = self._decode_memory(
            memory, tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits, self.ctc_head(memory)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int,
                max_len: int = 128, src_key_padding_mask: torch.Tensor = None,
                repetition_stop: int = 0):
        from bornil.models.decode import greedy_decode

        self.eval()
        memory = self.encode(src, src_key_padding_mask)

        def step_fn(ys):
            ys = ys.to(src.device)
            return self._decode_memory(
                memory, ys, src_key_padding_mask=src_key_padding_mask)[:, -1]

        return greedy_decode(step_fn, bos_id, eos_id, src.size(0), src.device,
                            max_len=max_len, repetition_stop=repetition_stop)

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, bos_id: int, eos_id: int,
                   beam_size: int = 5, max_len: int = 128,
                   length_penalty: float = 1.0,
                   src_key_padding_mask: torch.Tensor = None,
                   repetition_stop: int = 0):
        from bornil.models.decode import beam_decode

        self.eval()
        results = []
        for b in range(src.size(0)):
            s = src[b:b + 1]
            mask = None if src_key_padding_mask is None else src_key_padding_mask[b:b + 1]
            memory = self.encode(s, mask)

            def step_fn_single(ys, memory=memory, mask=mask):
                ys = ys.to(src.device)
                return self._decode_memory(
                    memory, ys, src_key_padding_mask=mask)[:, -1][0]

            results.append(beam_decode(
                step_fn_single, bos_id, eos_id, beam_size=beam_size,
                max_len=max_len, length_penalty=length_penalty,
                repetition_stop=repetition_stop))
        return results
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_mamba.py -v`
Expected: 6 passed. If `test_overfit_tiny` doesn't clear the threshold,
raise the step count before weakening the assertion — check first that
gradients are actually flowing into the `MambaBlock` parameters (a common
bug class here is an accidental `.detach()` or a `torch.no_grad()` context
leaking from `generate`/`beam_search` into `encode`, which those methods
must never do since `forward`/`ctc_logits` need gradients).

- [x] **Step 5: Commit**

```bash
git add bornil/models/mamba.py tests/models/test_mamba.py
git commit -m "feat(models): SignMamba (selective-SSM encoder + shared decoder)"
```

---

## Task 4: `build_model` dispatch for `arch == "mamba"`

**Files:**
- Modify: `bornil/models/build.py`
- Test: `tests/models/test_build.py` (add to existing file)

**Read first:** `bornil/models/build.py` in full — this task adds one more
`if arch == "mamba":` branch, mirroring the existing `"bilstm"`/`"stgcn"`
branches exactly in style.

**Interfaces:**
- Consumes: `SignMamba` (Task 3).
- Produces: `build_model(cfg, vocab_size, pad_id)` now also accepts
  `cfg["model"]["arch"] == "mamba"`.

- [x] **Step 1: Write the failing test**

```python
# add to tests/models/test_build.py
def test_build_model_mamba():
    from bornil.models.mamba import SignMamba
    cfg = {
        "model": {
            "arch": "mamba", "d_model": 16, "nhead": 2,
            "num_mamba_layers": 2, "d_state": 4, "expand": 2, "conv_kernel": 4,
            "num_decoder_layers": 1, "dim_feedforward": 32, "dropout": 0.0,
            "ctc_head": True,
        },
        "data": {"upper_body": True},
    }
    model = build_model(cfg, vocab_size=10, pad_id=0)
    assert isinstance(model, SignMamba)
    assert model.use_ctc is True
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_build.py::test_build_model_mamba -v`
Expected: FAIL — `ValueError: unknown model.arch 'mamba'; ...`.

- [x] **Step 3: Implement**

In `bornil/models/build.py`, update `SUPPORTED_ARCHS` and add the branch
(placed before the final `raise ValueError`, alongside the `"stgcn"`
branch):

```python
SUPPORTED_ARCHS = ("transformer", "bilstm", "stgcn", "mamba")
```

```python
    if arch == "mamba":
        from bornil.models.mamba import SignMamba
        return SignMamba(
            input_dim=input_dim, vocab_size=vocab_size,
            d_model=m["d_model"], nhead=m["nhead"],
            num_mamba_layers=int(m.get("num_mamba_layers", 4)),
            d_state=int(m.get("d_state", 16)),
            expand=int(m.get("expand", 2)),
            conv_kernel=int(m.get("conv_kernel", 4)),
            num_decoder_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
            pad_id=pad_id, use_ctc=use_ctc,
        )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/models/test_build.py -v`
Expected: all pass (existing tests + 1 new).

- [x] **Step 5: Commit**

```bash
git add bornil/models/build.py tests/models/test_build.py
git commit -m "feat(models): build_model dispatch for arch=mamba"
```

---

## Task 5: `configs/full_mamba.yaml` + `EXPERIMENTS.md` labeling

**Files:**
- Create: `configs/full_mamba.yaml`
- Modify: `scripts/train_sample.py` (`_append_experiment_row`)

**Why the second file:** `_append_experiment_row` in `scripts/train_sample.py`
currently has an `if/elif` that labels a run `"BiLSTM"`, `"ST-GCN"`, or
falls through to `"SignTransformer"` for anything else — a Mamba run would
silently be mislabeled `"SignTransformer"` in `EXPERIMENTS.md` without this
change.

- [x] **Step 1: Write the config**

```yaml
# configs/full_mamba.yaml - pure-PyTorch selective-SSM encoder + joint CTC/attention on random-v1.
trial:
  name: data_full_mamba
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
  arch: mamba
  d_model: 256
  nhead: 8
  num_mamba_layers: 4
  d_state: 16
  expand: 2
  conv_kernel: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.1
  ctc_head: true
train:
  epochs: 60
  batch_size: 32
  lr: 0.0005
  weight_decay: 0.0001
  clip_grad: 1.0
  warmup_steps: 0
  label_smoothing: 0.1
  ctc_weight: 0.3
  beam_size: 1
  max_decode_len: 256
  device: auto
```

(Matches `configs/full_ctc.yaml`'s `data`/`split`/`train` blocks exactly —
only `model.*` differs, per the design doc.)

- [x] **Step 2: Update `_append_experiment_row`**

In `scripts/train_sample.py`, find:

```python
    if arch == "bilstm":
        model_name = "BiLSTM"
        enc_layers = cfg["model"].get("enc_layers", "?")
        phase_task = "WS-C T0.1"
    elif arch == "stgcn":
        model_name = "ST-GCN"
        enc_layers = cfg["model"].get("stgcn_layers", "?")
        phase_task = "WS-C T1.1"
    else:
```

Add a `mamba` branch before the `else`:

```python
    if arch == "bilstm":
        model_name = "BiLSTM"
        enc_layers = cfg["model"].get("enc_layers", "?")
        phase_task = "WS-C T0.1"
    elif arch == "stgcn":
        model_name = "ST-GCN"
        enc_layers = cfg["model"].get("stgcn_layers", "?")
        phase_task = "WS-C T1.1"
    elif arch == "mamba":
        model_name = "SignMamba"
        enc_layers = cfg["model"].get("num_mamba_layers", "?")
        phase_task = "WS-D D2"
    else:
```

- [x] **Step 3: Verify no regression**

Run: `.venv/Scripts/python -m pytest -q`
Expected: same pass count as after Task 4 (this task adds no new tests —
it's config + a labeling branch, validated by Task 6's smoke test).

- [x] **Step 4: Commit**

```bash
git add configs/full_mamba.yaml scripts/train_sample.py
git commit -m "feat(train): full_mamba.yaml config + EXPERIMENTS.md labeling for arch=mamba"
```

---

## Task 6: Smoke-test timing measurement (mandatory decision gate)

**Files:** none (runs code).

**Why this task exists on its own, separate from Task 7's real run:** the
design doc is explicit that the sequential scan's real training-time cost
is genuinely unknown and must be measured, not assumed — anchors from prior
runs on this machine span a 57× range (TConv+CTC: 0.75 min/epoch; ST-GCN:
42.8 min/epoch, unexplained) that doesn't predict a Python-loop-based
architecture's cost either way.

- [ ] **Step 1: Run a short timed smoke test**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_mamba.yaml --epochs 2 --device cuda
```

**Do not add `--out-dir`.** (Corrected in place after a real run hit this:
`train_sample.py`'s `--out-dir` overwrites `cfg["trial"]["out_dir"]`, which
is the *same* directory the packed dataset — `sample_manifest.csv`,
`packed.npy`, `lengths.npy` — is read from, not a separate sandbox for run
output. Pointing it at a fresh directory like `smoke_mamba_timing` breaks
the dataset load with `FileNotFoundError` before training even starts.
`--epochs 2` alone is sufficient to isolate the smoke run: `config_hash(cfg)`
includes `epochs`, so this lands in its own `data_full/runs/<smoke-hash>/`,
distinct from the eventual real 60-epoch run's folder, without needing
`--out-dir` at all.)

Expected: completes without error (confirms the whole encoder→decoder→CTC
path works end-to-end on real data, not just the tiny synthetic tests), and
prints a per-epoch line with elapsed time (`train_sample.py`'s existing
epoch-loop logging). Record the **per-epoch minutes** from this run.

- [ ] **Step 2: Extrapolate and decide**

Multiply the measured per-epoch time by 60 (the real run's epoch count in
`configs/full_mamba.yaml`). Compare against TConv+CTC's 44.8 min total (for
context) and this repo's general willingness to run multi-hour jobs (D1's
plan budgeted a full working day for ~9 runs; ST-GCN's single run took
42.8h and was still run to completion).

- **If the extrapolated full run is roughly a few hours or less:** proceed
  to Task 7 with the full 60 epochs as configured — no change needed.
- **If the extrapolated full run would be on the order of a day or more**
  (ST-GCN-scale): this is a real finding, not something to route around
  silently. Do not silently cut epochs or dataset size. Instead: (a) log the
  measured per-epoch time and extrapolation to `docs/logs/CHANGELOG.md` as
  its own entry (what was measured, why it matters for D2's viability), and
  (b) **still run the full 60 epochs** unless it would take multiple days —
  in that specific case, stop and log to `docs/logs/FAILURES.md` that the
  pure-PyTorch sequential scan is computationally impractical on this
  hardware for a full run, with the measured numbers as evidence, and treat
  D2 as blocked pending either a parallel-scan reimplementation or a cloud
  environment (both explicitly out of scope for this plan — do not attempt
  either without a new spec).

- [ ] **Step 3: Clean up the smoke run**

```bash
git checkout -- docs/logs/EXPERIMENTS.md
```

(The smoke test's `EXPERIMENTS.md` row is throwaway, same convention as
D1's Task 7/8 smoke tests — the real run in Task 7 produces the row that
belongs in history. Since Step 1 no longer uses `--out-dir`, the smoke run's
folder is a normal `data_full/runs/<smoke-hash>/`, already covered by this
repo's existing gitignore for run output — no separate cleanup needed
beyond reverting the `EXPERIMENTS.md` row above.)

- [ ] **Step 4: Commit the timing finding**

```bash
git add docs/logs/CHANGELOG.md
git commit -m "docs(changelog): WS-D D2 Mamba smoke-test timing measurement"
```

(Only if Step 2 produced a CHANGELOG entry — a fast smoke test with a
comfortably-short extrapolation doesn't need its own entry; fold that
number into Task 8/9's CHANGELOG entries instead. Use judgment, but do not
lose the measured number either way — write it down somewhere in this
task's own notes if not in CHANGELOG.md yet.)

---

## Task 7: Real training run on `random-v1`

**Files:** none (runs code).

- [ ] **Step 1: ⏱ Run the real training**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_mamba.yaml --device cuda
```

Let it run to completion per Task 6's decision. Confirm on completion:
`docs/logs/EXPERIMENTS.md` has a new row labeled `SignMamba` (not
`SignTransformer` — this is exactly what Task 5 Step 2 was for; if it's
mislabeled, Task 5 wasn't applied correctly — fix it before continuing).
Note the run's config hash — call it `<mamba-hash>` for the rest of this
plan.

- [ ] **Step 2: Significance vs vanilla and vs TConv+CTC**

```bash
.venv/Scripts/python scripts/results_significance.py --run "vanilla=data_full/runs/c19d0ea08824" --run "mamba=data_full/runs/<mamba-hash>" --baseline vanilla --metrics bleu4 chrf++ --n 1000 --out docs/logs/SIGNIFICANCE_MAMBA_VS_VANILLA.md
.venv/Scripts/python scripts/results_significance.py --run "tconv_ctc=data_full/runs/b0a590ea3317" --run "mamba=data_full/runs/<mamba-hash>" --baseline tconv_ctc --metrics bleu4 chrf++ --n 1000 --out docs/logs/SIGNIFICANCE_MAMBA_VS_TCONV_CTC.md
```

- [ ] **Step 3: Diagnostics**

```bash
.venv/Scripts/python scripts/prediction_diagnostics.py --pred data_full/runs/<mamba-hash>/predictions.csv --out-json data_full/runs/<mamba-hash>/diagnostics.json
```

Read the resulting `diagnostics.json` (unique-hypothesis rate, mean
hyp/ref length, repeated-hypothesis rate — the same fields used for every
other model's Table 5 entry, and the same fields that caught ST-GCN's
runaway-repetition failure mode in FAIL-005; check for that pattern here
too).

- [ ] **Step 4: Commit**

```bash
git add docs/logs/EXPERIMENTS.md
git commit -m "feat(mamba): WS-D D2 real training run on random-v1"
```

---

## Task 8: Length-stratified analysis

**Files:**
- Create: `scripts/length_stratified_report.py`

**Why a new script, not a join against `predictions.csv`:** see the design
doc's "Validation" section — `predictions.csv` has no recording id, and
`LengthBucketBatchSampler` reorders rows by length within 640-row
megabatches even with `shuffle=False`, so a saved run's row order cannot be
reconstructed externally. This script re-decodes the test split itself,
capturing each example's true frame length in the same loop that produces
its hypothesis, so the two are paired by construction.

**No unit test for this task** — read-only analysis orchestration over
already-tested pieces (`build_model`, `resolve_split`, the model's own
`generate`/`beam_search`), same reasoning as D1's `pretrain_mkm.py`
(Task 7 of `docs/plan/2026-07-25-ws-d-d1-ssl-pretraining.md`) — validated by
running it for real in Step 2 below.

- [ ] **Step 1: Implement the script**

```python
# scripts/length_stratified_report.py
"""Length-stratified chrF++ report (WS-D D2).

Re-decodes a run's test split while capturing each example's true frame
length in the same pass, then reports corpus-level metrics separately for
clips >= --threshold frames vs below it. Read-only analysis over an
existing checkpoint -- no training, no change to predictions.csv or any
script that consumes it.

    python scripts/length_stratified_report.py --run data_full/runs/<hash> --threshold 256 --out data_full/runs/<hash>/length_stratified.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # import torch before pandas (FAIL-002)

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bornil.benchmark import metrics as M
from bornil.benchmark.splits import resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.models.build import build_model
from scripts.train_sample import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split-root", default="splits")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    device = resolve_device(args.device or cfg["train"].get("device", "auto"))

    out_dir = cfg["trial"]["out_dir"]
    man = pd.read_csv(os.path.join(out_dir, "sample_manifest.csv"))
    features = np.load(os.path.join(out_dir, "packed.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    tok = Tokenizer(cfg["data"]["tokenizer"])
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, _ = resolve_split(man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
                          fracs=tuple(cfg["split"]["fracs"]), split_root=args.split_root)
    rows = man[man["recording"].isin(set(sp["test"]))]
    pack_idx = rows["pack_idx"].tolist()
    targets = [targets_by_rec[r] for r in rows["recording"]]
    ds = SignDataset(features, lengths, pack_idx, targets,
                     normalize=cfg["data"]["normalize"], upper_body=cfg["data"]["upper_body"])
    collate = make_collate_fn(tok.pad_id)
    bs = LengthBucketBatchSampler(ds.item_lengths(), cfg["train"]["batch_size"],
                                  shuffle=False, seed=cfg["trial"]["seed"])
    loader = DataLoader(ds, batch_sampler=bs, collate_fn=collate)

    model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    beam_size = int(cfg["train"].get("beam_size", 1))
    hyps, refs, ex_lengths = [], [], []
    with torch.no_grad():
        for x, y, x_lengths, _ in loader:
            x = x.to(device)
            src_kpm = model.src_padding_mask(x_lengths.to(device), x.size(1))
            if beam_size > 1:
                gen = model.beam_search(x, bos_id=tok.bos_id, eos_id=tok.eos_id,
                                        beam_size=beam_size,
                                        max_len=cfg["train"]["max_decode_len"],
                                        src_key_padding_mask=src_kpm)
            else:
                gen = model.generate(x, bos_id=tok.bos_id, eos_id=tok.eos_id,
                                     max_len=cfg["train"]["max_decode_len"],
                                     src_key_padding_mask=src_kpm)
            for ids in gen:
                hyps.append(tok.decode(ids))
            for row in y:
                refs.append(tok.decode(row.tolist()))
            ex_lengths.extend(x_lengths.tolist())

    ex_lengths = np.array(ex_lengths)
    long_mask = ex_lengths >= args.threshold
    hyps_arr = np.array(hyps, dtype=object)
    refs_arr = np.array(refs, dtype=object)

    result = {
        "run": args.run,
        "threshold": args.threshold,
        "n_long": int(long_mask.sum()),
        "n_short": int((~long_mask).sum()),
        "long": M.corpus_metrics(list(hyps_arr[long_mask]), list(refs_arr[long_mask])) if long_mask.any() else None,
        "short": M.corpus_metrics(list(hyps_arr[~long_mask]), list(refs_arr[~long_mask])) if (~long_mask).any() else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for both the Mamba run and the TConv+CTC reference**

```bash
.venv/Scripts/python scripts/length_stratified_report.py --run data_full/runs/<mamba-hash> --threshold 256 --device cuda --out data_full/runs/<mamba-hash>/length_stratified.json
.venv/Scripts/python scripts/length_stratified_report.py --run data_full/runs/b0a590ea3317 --threshold 256 --device cuda --out data_full/runs/b0a590ea3317/length_stratified.json
```

Expected: both print a JSON object with `n_long`/`n_short` counts (should
match between the two runs — same test split) and `chrf++` inside `long`/
`short`. `n_long` should be roughly 17% of the 1799-row `random-v1` test set.

- [ ] **Step 3: Commit**

```bash
git add scripts/length_stratified_report.py
git commit -m "feat(analysis): length-stratified chrF++ report for WS-D D2"
```

---

## Task 9: Results write-up + final verification

**Files:**
- Modify: `docs/logs/RESULTS_OVERVIEW.md`
- Modify: `docs/logs/CHANGELOG.md`
- Modify only if needed: `docs/logs/FAILURES.md`

- [ ] **Step 1: Record the verdict — whatever it is**

Add `SignMamba` to `docs/logs/RESULTS_OVERVIEW.md` Table 2 (the random-v1
ablation table), following the exact row format of the `ST-GCN` row added
for FAIL-005 (Model | Run | Test rows | BLEU-4 | chrF++ | ROUGE-L |
BERTScore F1 | Interpretation). State the significance verdict vs vanilla
and vs TConv+CTC plainly. Add a short new subsection (e.g. "Table 2d: WS-D
D2 Length-Stratified Cut") with the Task 8 numbers for both runs side by
side and state whether any SSM (dis)advantage concentrates in the long-clip
stratum, per the design doc's four pre-registered outcome interpretations —
quote which one matched.

If chrF++ does **not** significantly beat TConv+CTC (the design doc's
predicted outcome), this is a valid, complete result — log it to
`docs/logs/FAILURES.md` using the exact template (see FAIL-005/006/007 for
the format), referencing the pre-registered prediction table from the
design doc and stating which row it matched. **Do not tune `d_state`,
`expand`, `conv_kernel`, or `num_mamba_layers` against this result** — that
was fixed once in Task 5 specifically to keep this comparison honest.

Add a `docs/logs/CHANGELOG.md` entry summarizing the whole D2 workstream
(model, run hash, significance results, length-stratified verdict, any
FAILURES.md entry).

- [ ] **Step 2: Full test suite**

```bash
.venv/Scripts/python -m pytest -q
```

Expected: all pass, exit 0.

- [ ] **Step 3: Final report**

Produce a short summary covering: Task 6's measured smoke-test timing and
what it implied for the full run; the real run's config hash and metrics;
significance vs vanilla and vs TConv+CTC with p-values; the length-
stratified verdict (which of the design doc's four predicted outcomes
matched); any `docs/logs/FAILURES.md` entry added; the full commit list for
this plan; full test-suite status; and an explicit statement of whether D2
is a supported win for the WS-D comparison (D1 is already closed negative —
this plan's job is to produce D2's verdict, not to guarantee a positive
one).
