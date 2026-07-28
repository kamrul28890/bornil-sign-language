# WS-D / D2 — Pure-PyTorch Selective-SSM (Mamba) Encoder: Design

> Status: approved in brainstorming session 2026-07-26/28. Follow-on to
> `docs/plan/2026-07-13-bornil-implementation-plan.md` §8.2 (D2). Feeds the
> WS-D novel-method comparison (D1 closed negative — see
> `docs/logs/FAILURES.md` FAIL-006/FAIL-007 — D2 and D3 remain).

## Goal

Implement a Mamba-style selective state-space encoder in pure PyTorch (no
`mamba-ssm` dependency) and test it against the current best encoder
(TConv2k5+Transformer+CTC, `b0a590ea3317`, chrF++ 10.913) on `random-v1`,
both overall and specifically on longer clips.

## Context / scope correction from the master plan

`docs/plan/2026-07-13-bornil-implementation-plan.md` §8.2 motivates D2 with
"linear-time in sequence length → can consume full-length clips (not just
128 frames), directly attacking the undersampling loss quantified in §5.3."
This premise is stale, verified during brainstorming (2026-07-26):

- The pipeline already packs/trains at **512 frames**, not 128. Checking
  `data_full/lengths.npy` directly: only **2.1%** of clips (382/17,988) hit
  the 512-frame cap, and only 3.2% exceed 90% of it. Truncation is a minor
  edge case, not a systemic "undersampling loss."
- **D2 is reframed as an architecture bet**, not a data-recovery bet: does a
  selective-SSM's different inductive bias (gated recurrent state,
  input-dependent recency-weighting) help this benchmark, independent of any
  truncation story.

Separately: the official `mamba-ssm` PyPI package has **no Windows wheels**
and resolves to a from-source build requiring the CUDA Toolkit (`nvcc`) and
an MSVC compiler — both confirmed absent on this machine
(`pip download --no-deps --no-build-isolation mamba-ssm==2.3.2.post1` fails
at the metadata-generation step; `nvcc`/`cl.exe` both not found). **This
plan implements the Mamba block in pure PyTorch instead** — no new
dependency, fully portable, but the sequential selective-scan will not have
the official CUDA kernel's speed (see "Compute budget").

**Explicitly out of scope for this spec:** D3 (KP→Bangla-LLM decoder) — its
own independent spec/plan cycle. Any future revisit of the *official*
`mamba-ssm` package (e.g., if the dev machine or a cloud environment later
gets a working CUDA-toolkit + MSVC setup) is also out of scope here — this
spec is specifically the pure-PyTorch reimplementation.

## Pre-registered prediction

Recorded before any code exists, matching this repo's convention
(FAIL-004/006/007 all pre-registered a prediction before running).

**Prediction.** Given the input ablation's finding
(`docs/logs/INPUT_ABLATION.md`) that this benchmark is dominated by static
pose rather than motion (frame-shuffling costs the current best model only
−0.88 chrF++), and that ST-GCN — a motion-focused architecture — already
failed here significantly below both vanilla and TConv+CTC (FAIL-005), **the
plausible outcome is another weak-or-no win for the SSM encoder.** However,
SSM's inductive bias (long-range sequential recency-weighting via a gated
recurrent state) is mechanistically unrelated to ST-GCN's (spatial
joint-graph convolution) — one architecture failing on this static-pose-
dominated benchmark does not predict the other will, which is the actual
reason this is still worth running rather than skipped by analogy to
FAIL-005.

**How each outcome is read:**

| Result | Interpretation |
|---|---|
| SSM chrF++ significantly below TConv+CTC (predicted) | Confirms the static-pose diagnosis generalizes across architectural inductive biases, not just motion-focused ones. Publishable alongside FAIL-005. |
| No significant difference either way | SSM is a viable but non-differentiating architecture choice here; report plainly. |
| SSM significantly beats TConv+CTC | Surprise — would need to be reconciled with the input ablation (does the selective/gated mechanism exploit something the frame-shuffle test didn't detect?) before being taken as a headline result. |
| SSM wins specifically on the long-clip stratified cut but not overall | The most informative surprise: would suggest the inductive-bias advantage is real but small enough to wash out in an aggregate dominated by short clips (median clip length is 139 frames). |

## Architecture

A full Mamba block (Gu & Dao, 2023), reimplemented in pure PyTorch — no
custom CUDA kernel, no parallel associative scan (see "Sequential scan"
below for why).

**Per block:**
1. RMSNorm on the block input.
2. Linear projection to `2 * d_inner` (`d_inner = expand * d_model`), split
   into `(x, z)`.
3. Depthwise causal `Conv1d` over `x` (kernel `conv_kernel`, causal padding
   — pad left by `conv_kernel - 1`, trim the right so output length equals
   input length and position `t`'s output never sees position `> t`), then
   SiLU.
4. **Selective-scan parameters computed from the conv output** (this
   input-dependence is what makes it "selective" rather than a fixed-
   coefficient linear SSM/S4): `Δ = softplus(Linear_Δ(conv_out))` shape
   `(B,T,d_inner)`; `B_t = Linear_B(conv_out)`, `C_t = Linear_C(conv_out)`
   shape `(B,T,d_state)` each (shared across the `d_inner` channel dimension,
   matching the original Mamba parameterization — not per-channel `B`/`C`).
   A learned per-channel `A` parameter (shape `(d_inner, d_state)`,
   initialized negative — e.g. `-exp(A_log)` — so the discretized decay
   `exp(Δ·A)` stays in `(0,1)`) is fixed (not input-dependent, matching
   Mamba's actual design — only `Δ`, `B`, `C` are selective).
5. **Sequential scan** over `T` (see below): recurrence
   `h_t = exp(Δ_t · A) ⊙ h_{t-1} + Δ_t · B_t ⊗ x_t`,
   `y_t = C_t · h_t + D ⊙ x_t` (`D` a learned per-channel skip, standard in
   Mamba). `h_t` has shape `(B, d_inner, d_state)`.
6. Gate: `y = y * SiLU(z)`.
7. Output Linear back to `d_model`.
8. Residual add (block input + step 7's output).

**Sequential scan, not a parallel associative scan.** Implemented as a
Python `for t in range(T)` loop, vectorized across `(B, d_inner, d_state)`
at each step. This is simpler to implement correctly and unit-test against
a closed-form reference than a log-space cumsum parallel-scan
reformulation, and clips cap at 512 frames — this is a research comparison,
not a speed benchmark. **Known risk, addressed explicitly in "Compute
budget":** a per-timestep Python loop is a GPU-inefficiency pattern
(kernel-launch-bound, not compute-bound) distinct from — and potentially
worse than — whatever made `docs/logs/FAILURES.md` FAIL-005's ST-GCN run
57× slower than TConv+CTC (confirmed during this design's review: ST-GCN's
slowness was **not** a Python-loop pattern — its forward pass is fully
vectorized einsum/Conv2d, so its root cause is unexplained and doesn't
transfer as a direct estimate here). The Mamba block's loop is a genuinely
different and well-precedented slow pattern (this is exactly why un-fused
RNN/LSTM implementations are slower than matmul-heavy architectures at
equivalent parameter counts).

**Stack:** `num_mamba_layers` blocks (default 4, matching the Transformer's
`num_encoder_layers=4` for comparability). **No TConv frontend, no explicit
positional encoding** — both exist elsewhere to compensate for a Transformer's
permutation-invariance; an SSM's recurrence is inherently sequential/order-
aware, so both would be redundant. (Decision made here, not asked of the
user — this is standard practice in the SSM literature, e.g. S4/Mamba papers
omit explicit positional encoding.)

**Padding.** Force `Δ_t = 0` at padded timesteps (`t >= length`) — this
freezes `h_t = h_{t-1}` exactly (identity update) rather than corrupting the
state with zero-padded input. `y_t` at padded positions is computed but
never consumed downstream (decoder `memory_key_padding_mask`, CTC losses
already restrict to `input_lengths`), matching every other encoder's
padding convention in this repo.

**Same downstream contract as every other WS-C encoder** (`SignTransformer`,
`SignBiLSTM`, `SignSTGCN`): `encode(src, src_key_padding_mask) ->
(B,S,d_model)` memory, consumed by the existing shared `nn.Transformer`
decoder + optional CTC head. Only the encoder changes — matching how
BiLSTM and ST-GCN were compared. **`SignMamba`'s decode-side methods
(`_decode_memory`, `forward`, `ctc_logits`, `forward_with_ctc`, `generate`,
`beam_search`) should be copied verbatim from `bornil/models/stgcn.py`'s
`SignSTGCN`** (lines ~184–260 as of this writing) — only `encode()`'s body
differs (Mamba stack instead of graph-conv blocks). Do not reimplement this
plumbing from scratch; it is identical boilerplate across every encoder in
this codebase by design.

## Components

New `bornil/models/mamba.py`:
- `SelectiveScan` — the core recurrence, either a standalone function
  `selective_scan(x, delta, A, B, C, D) -> y` (all pre-computed tensors in)
  or a small stateless module — implementer's choice, but it must be unit-
  testable in isolation from the rest of the block (no `nn.Module` state
  needed for the scan itself — `A`/`D` are passed in as tensors, owned by
  `MambaBlock`).
- `MambaBlock(nn.Module)` — one block as described above:
  `__init__(d_model, d_state, expand, conv_kernel)`;
  `forward(x, delta_mask) -> Tensor (B,T,d_model)` where `delta_mask` is the
  `(B,T)` bool padding mask used to zero `Δ` at padded positions.
- `SignMamba(nn.Module)` — the full model, mirroring `SignSTGCN`'s
  constructor signature shape: `__init__(input_dim, vocab_size, d_model,
  nhead, num_mamba_layers, d_state, expand, conv_kernel,
  num_decoder_layers, dim_feedforward, dropout, pad_id, use_ctc)`.

`bornil/models/build.py`: add `"mamba"` to `SUPPORTED_ARCHS` and a dispatch
branch reading `m["num_mamba_layers"]`, `m.get("d_state", 16)`,
`m.get("expand", 2)`, `m.get("conv_kernel", 4)` (Mamba paper defaults:
`d_state=16`, `expand=2`, `conv_kernel=4` — fixed for this spec, not tuned,
same discipline as D1's mask ratio: pick once, record it, don't grid-search
before the core comparison is confirmed).

`configs/full_mamba.yaml` mirrors `configs/full_ctc.yaml`'s `data`/`split`/
`train` blocks exactly (same 60 epochs, batch 32, lr, `ctc_weight: 0.3`,
etc. — only `model.*` differs: `arch: mamba`, `num_mamba_layers: 4`,
`d_state: 16`, `expand: 2`, `conv_kernel: 4`, `ctc_head: true`, no
`temporal_conv_layers` key since `SignMamba` has no TConv frontend).

## Validation / success criteria

1. **Core claim:** chrF++ on `random-v1` (`configs/full_mamba.yaml`) vs
   vanilla (`c19d0ea08824`) and vs TConv+CTC (`b0a590ea3317`), two-sided
   paired bootstrap n=1000 — same protocol as every Table 2 entry in
   `docs/logs/RESULTS_OVERVIEW.md`.
2. **Length-stratified cut:** using the same run's `predictions.csv` joined
   back to test-clip frame lengths (`data_full/lengths.npy` indexed by the
   test split's `pack_idx` — same join key pattern already used elsewhere in
   this repo), compute chrF++ separately for clips `>= 256` frames (~17% of
   clips, the top stratum) vs `< 256` frames, for both `SignMamba` and
   `b0a590ea3317`'s existing predictions. Report whether any SSM advantage
   (or disadvantage) concentrates in the long-clip stratum.
3. **Retrieval floor + diversity/length diagnostics** — `prediction_diagnostics.py`
   on the new run, same as every other model row (Table 4/5 convention).
4. Every run gets a normal `EXPERIMENTS.md` row.

## Testing (TDD, matching repo convention)

- `tests/models/test_selective_scan.py` — recurrence correctness: for a
  small `T`/`d_state`, compare `selective_scan`'s output against a
  brute-force reference that computes the same recurrence with an explicit
  Python loop over a *second*, independently-written implementation (not
  the same code path) — e.g. using plain Python floats/lists rather than
  batched tensor ops, so a bug shared between "the implementation" and "the
  reference" is actually two different bugs, not one. Also test: `Δ=0` at a
  position exactly freezes state (`h_t == h_{t-1}`); gradient check
  (`loss.backward()` produces finite, non-None grads for `A`, `D`, and the
  `Δ`/`B`/`C` projection weights).
- `tests/models/test_mamba_block.py` — padding invariance: two batches
  identical except for their padded tail must produce identical
  `MambaBlock` output at every non-padded position (this is the property
  the whole padding design depends on — test it directly, don't just trust
  the `Δ=0` unit fact from the scan test to imply it holds through the full
  block). Output shape test.
- `tests/models/test_mamba.py` (mirrors `tests/models/test_stgcn.py`) —
  `SignMamba` forward-shape test, `build_model` dispatch test for
  `arch="mamba"`, and an **overfit-tiny gate** (this repo's standing
  pattern) — teacher-forced loss on a tiny repeated batch must drop to near
  zero within a bounded step count, proving the whole
  encoder→decoder→loss path is wired correctly before any GPU/cloud run.

## Compute budget (for planning, not a hard spec)

**This is the one part of this spec with real uncertainty, and it must be
measured, not assumed** — see the implementation plan's mandatory smoke-test
timing step before committing to the full 60-epoch run. Anchors available
from prior runs on this exact machine (14,390 `random-v1` train clips,
batch 32):
- TConv2k5+Transformer+CTC (`b0a590ea3317`): 44.8 min / 60 epochs ≈ 0.75
  min/epoch. Fully vectorized matmul/conv architecture — the fast end of
  what this GPU does for this dataset.
- ST-GCN+CTC (`2dfdcbdf4e2d`): 2569.5 min / 60 epochs ≈ 42.8 min/epoch (57×
  slower). Also fully vectorized (einsum + Conv2d, no Python-level
  per-frame loop — confirmed by reading `bornil/models/stgcn.py` during
  this design's review) — its root cause is **unexplained** and was not
  worth re-investigating for a closed WS-C T1.1 negative result, but it
  demonstrates this GPU/setup can be surprisingly slow for reasons that
  aren't obvious from the architecture on paper.
- `SignMamba`'s sequential scan is a **different, well-precedented slow
  pattern** (Python-loop/kernel-launch-bound, like an un-fused RNN) that
  neither anchor directly predicts. Expect somewhere between "noticeably
  slower than TConv+CTC" and, in a pessimistic case, "ST-GCN-scale slow" —
  the range is too wide to commit to a number without measuring.
