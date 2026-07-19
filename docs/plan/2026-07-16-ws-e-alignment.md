# WS-E Alignment: CTC Forced Alignment + Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Follow-on to `docs/plan/2026-07-13-bornil-implementation-plan.md` §9 (WS-E) and `docs/plan/2026-07-16-ws-c-suite-and-benchmark-hardening.md` (Phase B built the CTC head this plan consumes).

**Goal:** Induce frame↔subword alignments for Bornil from sentence-only labels using the trained joint-CTC checkpoint, and validate them without any gold alignment by triangulating against an independent signal.

**Architecture:** The Phase-B CTC head already emits per-frame subword posteriors. Viterbi forced alignment over the CTC lattice turns those posteriors plus the known target sentence into a per-frame label assignment — no training, decode-time only, over checkpoints that already exist. Because there is no ground-truth alignment to score against, correctness is established by **triangulation**: CTC boundaries are compared against motion changepoints (keypoint velocity minima, an independent non-model signal) and against decoder cross-attention, and both must agree above a randomized chance baseline.

**Tech Stack:** PyTorch (Viterbi in pure torch/numpy), numpy, pandas. CPU-sufficient — this is decode-time work over saved checkpoints; no GPU training.

## Global Constraints

- **No retraining.** This plan consumes existing checkpoints only: `data_full/runs/b0a590ea3317/best.pt` (random-v1), `data_full/runs/40cffb9e62bb/best.pt` (signerindep-v1), `data_full/runs/a5a9060499a4/best.pt` (unseen-sentence-v1). Each run folder has a `config.json` to rebuild the model from.
- **CTC conventions are fixed** (set in WS-C Phase B): `blank == pad_id == 0`; tokenizer ids are `pad=0, bos=2, eos=3, vocab=4000`; CTC labels are the subword ids strictly between BOS and EOS.
- **The model contract:** `SignTransformer.ctc_logits(src, src_key_padding_mask=None) -> (B, S, vocab_size)`; build with `use_ctc=True`. `feature_dim(upper_body)` gives the input dim — never a literal.
- **Do not modify the locked metric protocol** (`bornil/benchmark/metrics.py`) or any WS-C model/eval behavior. This plan is additive: a new `bornil/align/` package plus scripts.
- **No gold alignment exists.** Never write a step that claims alignment "accuracy". Validation is agreement-vs-chance, monotonicity, and coverage. Any claim of correctness must be relative to a randomized baseline.
- **Windows import order:** every script entry point imports `bornil.runtime` before pandas (FAIL-002). Use `.venv/Scripts/python`. PowerShell has no `&&` — chain with `;`.
- **Logging discipline:** meaningful changes → `docs/logs/CHANGELOG.md`; dead-ends → `docs/logs/FAILURES.md`. This plan produces analysis artifacts, not EXPERIMENTS rows (no training runs).

## Scope

**In scope:** CTC forced alignment (program plan §9.2.1), attention alignment (§9.2.2), unsupervised motion changepoints as an independent validator (§9.2.4), and cross-method agreement.

**Explicitly out of scope — defer to their own plans:** contrastive word↔clip dual encoder (§9.2.3), pseudo-gloss induction and the sign lexicon (§9.2.5), the two-stage pseudo-gloss SLT variant, and anything in WS-F (avatar). Those all depend on this layer working and validated first. **Do not build them here (YAGNI).**

---

## Plan amendments (2026-07-19, after the input ablation)

`docs/logs/INPUT_ABLATION.md` landed after this plan was written and changes four
things. Amendments are recorded here rather than rewritten inline so the original
reasoning stays auditable.

1. **New Task A0 gates Phase B.** The ablation showed the CTC model retains 92% of
   its chrF++ with frame order shuffled — it is close to time-invariant on the
   character metric. If the CTC head's per-frame posteriors are also temporally
   flat, forced alignment returns essentially arbitrary segmentations and Phase B
   would measure nothing: it would report chance agreement for a *degenerate-head*
   reason, not the interesting *ungrounded-temporal-model* reason. A0 distinguishes
   these before any motion-changepoint machinery is built.
2. **A1 must be vectorized, not a Python double loop.** Measured on this machine at
   T=512 / S=51: the plan's `for t: for s:` body with tensor indexing runs 0.63 s
   per clip (~19 min across the three runs at the real median length of 171
   frames). Tolerable once — but Phase B's agreement-vs-chance needs a randomized
   baseline, and at N=100 permutations that becomes ~32 h. A vectorized numpy
   implementation measured 8.3 ms per clip (**76× faster**), making the same
   baseline ~20 min. Build it vectorized from the start.
3. **A1's test suite does not exercise the DP.** Every test uses log-probs so peaked
   that per-frame argmax already gives the answer, so they would pass against a
   greedy implementation that ignores the CTC constraints entirely. Add cases where
   the constrained Viterbi path *must* differ from greedy.
4. **A2 uses `build_model`.** `scripts/eval_checkpoint_decode.py` no longer contains
   inline model construction to copy — it now calls `bornil.models.build.build_model`,
   which reads `model.ctc_head` from the run config. The plan's note that
   "`use_ctc=True` must be set from the run's `config.json`" described a bug that
   was fixed on 2026-07-19; A2 gets this for free and must not hand-roll it.

### Task A0: Is the CTC head temporally informative? (gate for Phase B)

**Files:** create `scripts/ctc_posterior_stats.py`; no new library module.

**Why:** cheap falsification of the premise Phase A/B rest on. Consumes an existing
checkpoint, no retraining.

Report, over the test split of `b0a590ea3317`:
- **Blank rate** — fraction of frames whose CTC argmax is blank. Near 1.0 means the
  head emits almost nothing per-frame and alignments will be driven by tie-breaking.
- **Per-frame posterior entropy** (mean, and vs. the log-vocab ceiling). Near-uniform
  posteriors mean no frame carries label evidence.
- **Temporal variability** — number of distinct non-blank argmax labels per clip, and
  mean run length of a constant argmax. A head that outputs the same label for all
  frames is temporally uninformative even if confident.
- **Contrast against `frame_shuffle`** — recompute all three on frame-shuffled input.
  If the statistics are unchanged, the CTC head is genuinely time-invariant, which
  predicts chance-level Phase B agreement for a mechanistic reason.

**Gate:** if the blank rate exceeds ~0.95 *and* per-clip distinct labels is ~1, stop
and log to `docs/logs/FAILURES.md` — Phase B cannot produce an interpretable result
and the WS-E→WS-F bridge needs rethinking. Otherwise continue to B.

### Task A1: Viterbi forced-alignment algorithm

**Files:**
- Create: `bornil/align/__init__.py` (docstring only)
- Create: `bornil/align/ctc_forced.py`
- Test: `tests/align/test_ctc_forced.py` (create; add `tests/align/__init__.py` if the other test dirs have one — check `tests/models/`)

**Interfaces:**
- Produces:
  - `class AlignmentError(Exception)` — raised when a label sequence cannot fit the frame budget.
  - `forced_align(log_probs, labels, blank=0) -> list[int]` — `log_probs` is a `(T, V)` float tensor of **log-softmax** values, `labels` a list of subword ids (no blank/BOS/EOS). Returns a length-`T` list of extended-state indices (the Viterbi path).
  - `path_to_spans(path, labels) -> list[tuple[int, int, int]]` — converts a Viterbi path to `(label_index, start_frame, end_frame)` triples, one per label occurrence that received at least one frame; `end_frame` is inclusive.
  - `align_clip(log_probs, labels, blank=0) -> list[tuple[int, int, int]]` — convenience: `forced_align` then `path_to_spans`.

**Algorithm note (read before implementing):** standard CTC forced alignment. Interleave blanks into the label sequence to build the extended sequence `ext = [b, y1, b, y2, b, ..., yL, b]` of length `S = 2L+1`. Viterbi over `(T, S)`: from state `s` at `t-1` you may stay at `s`, advance to `s`, or skip from `s-2` **only when** `ext[s]` is not blank and `ext[s] != ext[s-2]` (the skip is what forbids collapsing two identical adjacent labels). Odd extended indices are labels: extended index `s` maps to label index `(s-1)//2`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/align/test_ctc_forced.py
import bornil.runtime  # noqa: F401
import pytest
import torch

from bornil.align.ctc_forced import AlignmentError, align_clip, forced_align, path_to_spans


def _log_probs_favoring(seq, vocab=5, T=None, blank=0):
    """Build (T,V) log-probs that strongly favor emitting seq[t] at frame t."""
    T = T or len(seq)
    p = torch.full((T, vocab), 1e-6)
    for t, v in enumerate(seq):
        p[t, v] = 1.0
    return (p / p.sum(-1, keepdim=True)).log()


def test_recovers_obvious_alignment():
    # frames:      b  1  1  b  2  b     labels [1, 2]
    lp = _log_probs_favoring([0, 1, 1, 0, 2, 0])
    spans = align_clip(lp, [1, 2], blank=0)
    assert spans == [(0, 1, 2), (1, 4, 4)]


def test_path_length_equals_frame_count():
    lp = _log_probs_favoring([0, 1, 1, 0, 2, 0])
    path = forced_align(lp, [1, 2], blank=0)
    assert len(path) == 6


def test_spans_are_monotonic_and_non_overlapping():
    lp = _log_probs_favoring([0, 1, 1, 0, 2, 2, 0, 3, 0])
    spans = align_clip(lp, [1, 2, 3], blank=0)
    assert [s[0] for s in spans] == [0, 1, 2]
    for (_, s0, e0), (_, s1, _e1) in zip(spans, spans[1:]):
        assert s0 <= e0 < s1


def test_repeated_adjacent_labels_need_a_separating_blank():
    # labels [1, 1] require at least 3 frames: 1, b, 1
    lp = _log_probs_favoring([1, 0, 1])
    spans = align_clip(lp, [1, 1], blank=0)
    assert [s[0] for s in spans] == [0, 1]
    assert spans[0][2] < spans[1][1]


def test_infeasible_when_too_few_frames():
    lp = _log_probs_favoring([1, 2])          # 2 frames
    with pytest.raises(AlignmentError):
        forced_align(lp, [1, 2, 3, 4], blank=0)   # needs >= 4


def test_path_to_spans_skips_labels_that_got_no_frames():
    # A path that never visits label index 1 (extended state 3).
    path = [0, 1, 1, 2]      # blank, y1, y1, blank
    spans = path_to_spans(path, [7, 9])
    assert spans == [(0, 1, 2)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/align/test_ctc_forced.py -v`
Expected: FAIL — `No module named 'bornil.align'`.

- [ ] **Step 3: Implement the module**

```python
# bornil/align/ctc_forced.py
"""CTC forced alignment (WS-E §9.2.1): frames -> subword labels.

Given per-frame CTC log-probabilities from a trained joint-CTC encoder and the
known target subword sequence, Viterbi decoding over the CTC lattice assigns
every frame to either a label or blank. This yields a frame-level alignment
"for free" from a model trained with sentence-only supervision — Bornil has no
glosses and no timings, so this is the cheapest route to temporal grounding.

Conventions (fixed by WS-C Phase B): blank == pad_id == 0; `labels` excludes
BOS/EOS/pad. Extended sequence is [b, y1, b, ..., yL, b]; odd extended indices
are labels, and extended index s maps to label index (s-1)//2.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

NEG_INF = float("-inf")


class AlignmentError(Exception):
    """Raised when the label sequence cannot be aligned to the frame budget."""


def _extended(labels: Sequence[int], blank: int) -> List[int]:
    ext = [blank]
    for y in labels:
        ext.append(int(y))
        ext.append(blank)
    return ext


def forced_align(log_probs: torch.Tensor, labels: Sequence[int],
                 blank: int = 0) -> List[int]:
    """Viterbi-align `labels` to frames. Returns the per-frame extended-state path.

    log_probs: (T, V) log-softmax values. Raises AlignmentError when infeasible
    (too few frames for the labels, including blanks required between repeats).
    """
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must be (T, V), got {tuple(log_probs.shape)}")
    labels = [int(y) for y in labels]
    if not labels:
        raise AlignmentError("cannot align an empty label sequence")
    ext = _extended(labels, blank)
    T, S = log_probs.size(0), len(ext)

    alpha = torch.full((T, S), NEG_INF, dtype=torch.float64)
    back = torch.zeros((T, S), dtype=torch.long)
    alpha[0, 0] = float(log_probs[0, ext[0]])
    if S > 1:
        alpha[0, 1] = float(log_probs[0, ext[1]])

    for t in range(1, T):
        for s in range(S):
            best, arg = alpha[t - 1, s], s
            if s > 0 and alpha[t - 1, s - 1] > best:
                best, arg = alpha[t - 1, s - 1], s - 1
            if (s > 1 and ext[s] != blank and ext[s] != ext[s - 2]
                    and alpha[t - 1, s - 2] > best):
                best, arg = alpha[t - 1, s - 2], s - 2
            if best == NEG_INF:
                continue
            alpha[t, s] = best + float(log_probs[t, ext[s]])
            back[t, s] = arg

    last, prev = alpha[T - 1, S - 1], alpha[T - 1, S - 2] if S > 1 else NEG_INF
    if last == NEG_INF and prev == NEG_INF:
        raise AlignmentError(
            f"infeasible alignment: {T} frames cannot cover {len(labels)} labels")
    end = S - 1 if last >= prev else S - 2

    path = [0] * T
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = int(back[t, s])
    return path


def path_to_spans(path: Sequence[int],
                  labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Convert a Viterbi path to (label_index, start_frame, end_frame) triples.

    end_frame is inclusive. Labels that received no frames are omitted (a valid
    CTC outcome when the model routes them through blanks).
    """
    spans: List[Tuple[int, int, int]] = []
    current = None
    for t, s in enumerate(path):
        if s % 2 == 0:            # blank state
            current = None
            continue
        li = (s - 1) // 2
        if current is not None and current[0] == li:
            spans[-1] = (li, spans[-1][1], t)
        else:
            spans.append((li, t, t))
            current = (li,)
    return spans


def align_clip(log_probs: torch.Tensor, labels: Sequence[int],
               blank: int = 0) -> List[Tuple[int, int, int]]:
    """forced_align + path_to_spans."""
    return path_to_spans(forced_align(log_probs, labels, blank=blank), labels)
```

Also create `bornil/align/__init__.py`:

```python
"""WS-E alignment: induce word/frame correspondence from sentence-only labels."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/align/test_ctc_forced.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add bornil/align/__init__.py bornil/align/ctc_forced.py tests/align/
git commit -m "feat(align): CTC forced alignment via Viterbi (WS-E 9.2.1)"
```

### Task A2: Alignment extraction script over a trained checkpoint

**Files:**
- Create: `scripts/align_checkpoint.py`
- Test: `tests/align/test_align_checkpoint.py` (create)

**Read first:** `scripts/eval_checkpoint_decode.py` — it already implements "rebuild the model from a run's `config.json`, load `best.pt`, iterate the test split." Copy that structure rather than reinventing it; this script is the alignment analogue.

**Interfaces:**
- Produces:
  - `align_split(model, loader, tokenizer, device) -> pandas.DataFrame` with columns `row, label_index, token_id, token_str, start_frame, end_frame, n_frames` (one row per aligned label occurrence; `row` indexes the test example, `n_frames` is the clip's true frame count).
  - CLI: `python scripts/align_checkpoint.py --run data_full/runs/b0a590ea3317 --out data_full/runs/b0a590ea3317/alignments.csv`
- Consumes: `align_clip` (A1); `SignTransformer(..., use_ctc=True).ctc_logits`; `Tokenizer`; `SignDataset`/`make_collate_fn`; `resolve_split`; `load_config`/run `config.json`.

- [ ] **Step 1: Write the failing test** (unit-test the aggregation, not the full CLI — no checkpoint needed)

```python
# tests/align/test_align_checkpoint.py
import bornil.runtime  # noqa: F401
import torch

from scripts.align_checkpoint import spans_to_rows


class _FakeTok:
    def decode(self, ids):
        return "tok" + "_".join(str(i) for i in ids)


def test_spans_to_rows_shapes_and_fields():
    spans = [(0, 1, 3), (1, 5, 5)]
    rows = spans_to_rows(row=7, spans=spans, labels=[11, 12],
                         tokenizer=_FakeTok(), n_frames=9)
    assert len(rows) == 2
    assert rows[0] == {"row": 7, "label_index": 0, "token_id": 11,
                       "token_str": "tok11", "start_frame": 1,
                       "end_frame": 3, "n_frames": 9}
    assert rows[1]["start_frame"] == 5 and rows[1]["end_frame"] == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/align/test_align_checkpoint.py -v`
Expected: FAIL — `No module named 'scripts.align_checkpoint'`.

- [ ] **Step 3: Implement the script**

Structure it as: `spans_to_rows(row, spans, labels, tokenizer, n_frames) -> list[dict]` (pure, tested above), then `align_split(...)` looping the loader — for each example, take `ctc_logits` for its true frame count (slice off padding using `x_lengths`), `log_softmax(-1)`, strip BOS/EOS/pad from the target to get `labels`, call `align_clip`, and convert. Wrap each clip in `try/except AlignmentError` and count failures rather than crashing the run. Then `main()` with `--run`, `--split-root`, `--device`, `--out`, mirroring `eval_checkpoint_decode.py`'s argument handling and model reconstruction (`use_ctc=True` must be set from the run's `config.json` `model.ctc_head`).

Print a summary line: `aligned N/M clips | K labels | infeasible: I`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/align/test_align_checkpoint.py -v`
Expected: 1 passed.

- [ ] **Step 5: Produce real alignments for all three CTC runs**

```bash
.venv/Scripts/python scripts/align_checkpoint.py --run data_full/runs/b0a590ea3317 --device cuda --out data_full/runs/b0a590ea3317/alignments.csv
.venv/Scripts/python scripts/align_checkpoint.py --run data_full/runs/40cffb9e62bb --device cuda --out data_full/runs/40cffb9e62bb/alignments.csv
.venv/Scripts/python scripts/align_checkpoint.py --run data_full/runs/a5a9060499a4 --device cuda --out data_full/runs/a5a9060499a4/alignments.csv
```
Expected: each prints an `aligned N/M clips` summary and writes a CSV. **If the infeasible count is high (>5%), stop and investigate** — it likely means the 512-frame ceiling is truncating long targets; log it to `docs/logs/FAILURES.md` before continuing.

- [ ] **Step 6: Commit**

```bash
git add scripts/align_checkpoint.py tests/align/test_align_checkpoint.py
git commit -m "feat(align): extract CTC forced alignments from a run checkpoint"
```

---

## Phase B — Validation without gold alignment (triangulation)

**Why this phase exists:** there is no ground-truth alignment for Bornil, so "the alignments look plausible" is not evidence. The claim we can defend is: *CTC boundaries coincide with an independent motion signal more often than chance.* That is what this phase measures.

### Task B1: Motion changepoints from keypoint velocity

**Files:**
- Create: `bornil/align/motion.py`
- Test: `tests/align/test_motion.py` (create)

**Interfaces:**
- Produces:
  - `frame_velocity(feats) -> np.ndarray` — `feats` is `(T, D)` prepared keypoints; returns length-`T` mean L2 magnitude of the per-frame difference (index 0 set to the value at index 1 so the array stays length `T`).
  - `changepoints(velocity, min_distance=3) -> list[int]` — indices of local minima of velocity, enforcing a minimum separation. Sign boundaries tend to fall at motion minima (the hands slow between signs).

- [ ] **Step 1: Write the failing tests**

```python
# tests/align/test_motion.py
import numpy as np

from bornil.align.motion import changepoints, frame_velocity


def test_velocity_length_matches_frames():
    feats = np.random.RandomState(0).randn(20, 6)
    v = frame_velocity(feats)
    assert v.shape == (20,)


def test_velocity_zero_for_static_clip():
    feats = np.ones((10, 4))
    assert np.allclose(frame_velocity(feats), 0.0)


def test_changepoints_find_motion_minima():
    # velocity dips at index 5 and 15
    v = np.ones(21) * 5.0
    v[5] = 0.1
    v[15] = 0.1
    cps = changepoints(v, min_distance=3)
    assert 5 in cps and 15 in cps


def test_changepoints_respect_min_distance():
    v = np.ones(21) * 5.0
    v[5] = 0.1
    v[6] = 0.1
    cps = changepoints(v, min_distance=3)
    assert len([c for c in cps if 4 <= c <= 7]) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/align/test_motion.py -v`
Expected: FAIL — `No module named 'bornil.align.motion'`.

- [ ] **Step 3: Implement**

```python
# bornil/align/motion.py
"""Unsupervised temporal segmentation signal (WS-E §9.2.4).

Keypoint velocity minima approximate sign boundaries: the hands decelerate
between signs. This is model-independent, which is exactly why it is useful —
it gives CTC forced alignment something external to be checked against, in a
dataset with no gold alignment.
"""
from __future__ import annotations

from typing import List

import numpy as np


def frame_velocity(feats: np.ndarray) -> np.ndarray:
    """Mean L2 magnitude of the per-frame keypoint difference. Length T."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"feats must be (T, D), got {feats.shape}")
    if feats.shape[0] < 2:
        return np.zeros(feats.shape[0])
    diff = np.linalg.norm(np.diff(feats, axis=0), axis=1) / np.sqrt(feats.shape[1])
    return np.concatenate([diff[:1], diff])


def changepoints(velocity: np.ndarray, min_distance: int = 3) -> List[int]:
    """Indices of local velocity minima, greedily thinned to `min_distance`."""
    v = np.asarray(velocity, dtype=np.float64)
    if v.size < 3:
        return []
    cand = [i for i in range(1, v.size - 1) if v[i] <= v[i - 1] and v[i] <= v[i + 1]]
    cand.sort(key=lambda i: v[i])
    chosen: List[int] = []
    for i in cand:
        if all(abs(i - c) >= min_distance for c in chosen):
            chosen.append(i)
    return sorted(chosen)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/align/test_motion.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add bornil/align/motion.py tests/align/test_motion.py
git commit -m "feat(align): keypoint-velocity changepoints as an independent boundary signal"
```

### Task B2: Boundary agreement metric + chance baseline

**Files:**
- Create: `bornil/align/agreement.py`
- Test: `tests/align/test_agreement.py` (create)

**Interfaces:**
- Produces:
  - `boundary_f1(pred, ref, tolerance=2) -> dict` — returns `{"precision", "recall", "f1", "n_pred", "n_ref"}`; a predicted boundary matches a reference boundary within `±tolerance` frames, each reference matched at most once (greedy nearest-first).
  - `chance_boundary_f1(n_pred, n_ref, n_frames, tolerance=2, trials=200, seed=0) -> float` — mean F1 of uniformly random boundary sets of the same sizes. **This is the number the real F1 must beat** for any alignment claim.

- [ ] **Step 1: Write the failing tests**

```python
# tests/align/test_agreement.py
from bornil.align.agreement import boundary_f1, chance_boundary_f1


def test_perfect_match():
    r = boundary_f1([3, 10, 20], [3, 10, 20], tolerance=0)
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0


def test_tolerance_window_counts_near_misses():
    r = boundary_f1([4, 11], [3, 10], tolerance=2)
    assert r["f1"] == 1.0


def test_no_match():
    r = boundary_f1([0, 1], [50, 60], tolerance=2)
    assert r["f1"] == 0.0


def test_each_reference_matched_at_most_once():
    # two predictions crowding one reference -> recall 1/1, precision 1/2
    r = boundary_f1([10, 11], [10], tolerance=2)
    assert r["recall"] == 1.0
    assert r["precision"] == 0.5


def test_chance_baseline_is_between_zero_and_one():
    c = chance_boundary_f1(n_pred=5, n_ref=5, n_frames=100, tolerance=2, trials=50)
    assert 0.0 <= c <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/align/test_agreement.py -v`
Expected: FAIL — `No module named 'bornil.align.agreement'`.

- [ ] **Step 3: Implement** `boundary_f1` (greedy nearest-first matching, each reference consumed once) and `chance_boundary_f1` (uniform random boundary sets via `np.random.default_rng(seed)`, averaged over `trials`). Return floats, not numpy scalars.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/align/test_agreement.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add bornil/align/agreement.py tests/align/test_agreement.py
git commit -m "feat(align): boundary agreement F1 + randomized chance baseline"
```

### Task B3: Alignment validation report

**Files:**
- Create: `scripts/alignment_report.py`
- Create (output): `docs/logs/ALIGNMENT_VALIDATION.md`

**Interfaces:**
- Consumes: `alignments.csv` (A2), the packed features + lengths (for velocity), `frame_velocity`/`changepoints` (B1), `boundary_f1`/`chance_boundary_f1` (B2).
- Produces: a Markdown report per run with, over the test split:
  - **Agreement vs chance** — mean boundary F1 of CTC span boundaries against motion changepoints, and the chance baseline, with the ratio. *This is the headline validation number.*
  - **Sanity diagnostics** — % of clips aligned (vs infeasible); mean blank fraction per clip; mean span duration in frames; % of labels that received zero frames; monotonicity violations (must be 0 by construction — report it as a guard).
  - **Length sanity** — correlation between clip frame count and number of aligned labels (should be positive; a flat correlation would suggest the alignment is not tracking content).

- [ ] **Step 1: Generate the report for the random-v1 CTC run**

```bash
.venv/Scripts/python scripts/alignment_report.py --run data_full/runs/b0a590ea3317 --out docs/logs/ALIGNMENT_VALIDATION.md
```

- [ ] **Step 2: Interpret and record the verdict**

Write the verdict explicitly in `docs/logs/ALIGNMENT_VALIDATION.md` and add a `docs/logs/CHANGELOG.md` entry:
- **If agreement F1 clearly exceeds the chance baseline** → the alignments carry real temporal signal; WS-E is viable and this independently corroborates the "CTC grounds the encoder" claim from `RESULTS_OVERVIEW.md` Table 5.
- **If agreement is at or near chance** → this is an important negative result, not a failure to hide. Log it to `docs/logs/FAILURES.md`, and note that the chrF++/length gains may come from a better language-side prior rather than temporal grounding. **Do not proceed to Phase C or to pseudo-glosses on a chance-level alignment.**

- [ ] **Step 3: Commit**

```bash
git add scripts/alignment_report.py docs/logs/ALIGNMENT_VALIDATION.md docs/logs/CHANGELOG.md
git commit -m "feat(align): alignment validation report vs motion-changepoint chance baseline"
```

---

## Phase C — Attention alignment + cross-method agreement

**Gate:** only start this if Phase B's agreement clearly beat chance. If it did not, stop and reassess — a second alignment method will not rescue a chance-level first one.

### Task C1: Capture decoder cross-attention

**Files:**
- Create: `bornil/align/attention.py`
- Test: `tests/align/test_attention.py` (create)

**Implementation note (this is the tricky part — read it):** `nn.TransformerDecoderLayer` calls its `multihead_attn` with `need_weights=False`, so cross-attention weights are not returned by a plain forward pass. Do **not** try to read them off the module afterwards. Instead implement a context manager that temporarily wraps each decoder layer's `multihead_attn.forward` to force `need_weights=True, average_attn_weights=True` and record the returned weights:

```python
@contextmanager
def capture_cross_attention(decoder):
    """Temporarily force nn.TransformerDecoderLayer cross-attn to return weights.

    Yields a list that fills with one (B, L, S) tensor per decoder layer per
    forward pass. Restores the original forwards on exit.
    """
    store, originals = [], []
    for layer in decoder.layers:
        mha = layer.multihead_attn
        original = mha.forward
        originals.append((mha, original))

        def patched(q, k, v, _orig=original, **kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            out, w = _orig(q, k, v, **kw)
            store.append(w.detach())
            return out, w

        mha.forward = patched
    try:
        yield store
    finally:
        for mha, original in originals:
            mha.forward = original
```

- **Interfaces produced:** `capture_cross_attention(decoder)` (context manager, yields a list of `(B, L, S)` tensors); `attention_spans(weights, n_labels) -> list[tuple[int,int,int]]` — argmax-frame per target position, aggregated to per-label `(label_index, start_frame, end_frame)`.
- **Test it** with a tiny `SignTransformer`: assert the store fills with one tensor per decoder layer, each shaped `(B, L, S)`, and that the model's output logits are **unchanged** vs an uncaptured forward (the patch must be side-effect-free).

### Task C2: CTC↔attention agreement

**Files:**
- Modify: `scripts/alignment_report.py` (add an attention-vs-CTC agreement section)

Report boundary F1 between CTC spans and attention spans, against the same chance baseline. Per program plan §9's validation rule, **cross-method agreement above chance is the confidence signal** in the absence of gold alignment. Add the result to `docs/logs/ALIGNMENT_VALIDATION.md` and a CHANGELOG entry. If attention is diffuse/uninterpretable (a known failure mode, §9 "Failure modes → fixes"), record that and rely on CTC as primary — that is an expected, documentable outcome, not a blocker.

---

## Self-review notes (coverage check)

- **§9.2.1 CTC forced alignment** → Phase A (A1 algorithm, A2 extraction).
- **§9.2.2 Attention alignment** → Phase C, explicitly gated on Phase B.
- **§9.2.4 Unsupervised segmentation** → Phase B1, used as the independent validator rather than as a standalone deliverable.
- **§9 validation "triangulate, no gold alignment"** → Phase B2/B3 (agreement vs randomized chance) and C2 (cross-method agreement).
- **§9.2.3 contrastive word↔clip, §9.2.5 pseudo-glosses, sign lexicon** → deliberately out of scope; own plan once this layer is validated.
- **§9.3 two granularities** → this plan delivers the **frame/clip level** only. Word/unit level needs §9.2.3+9.2.5 and is deferred.

## Pre-registered prediction (recorded 2026-07-19, before Phase A0 was run)

Written down *before* any alignment code existed, so the outcome cannot be
rationalized after the fact. Evidence: `docs/logs/INPUT_ABLATION.md`.

**Prediction.** CTC forced alignments will show **weak but above-chance** boundary
agreement with motion changepoints — clearly better than the randomized baseline,
but far from the agreement a genuinely temporally-grounded model would produce.

**Basis.** Frame-order shuffling costs the CTC model only −0.88 chrF++ (92%
retained) but 89% of its BLEU-4. Read together: whatever temporal structure the
encoder has is real but weak, and it shows up in word-order-sensitive metrics
rather than character-overlap ones. So there should be *some* temporal signal for
forced alignment to latch onto, but not much.

**How each outcome is read:**

| Phase B result | Interpretation |
|---|---|
| Agreement at chance, **and** A0 shows a degenerate head (blank rate >0.95, ~1 distinct label) | Uninformative — the head never emits per-frame evidence. Not a finding about grounding. Log to FAILURES.md, rethink the bridge. |
| Agreement at chance, **and** A0 shows a healthy head | **Strong negative result.** The CTC gain in Table 5 is not temporal grounding. Consistent with the ablation and publishable with it. |
| Weak but above chance (**predicted**) | Confirms the ablation. The joint claim — grounded but not temporally grounded — is stronger than either result alone. |
| Strongly above chance | **Surprise; contradicts the ablation.** Most likely reconciliation: chrF++ is too blunt to register temporal structure that alignment does capture (consistent with BLEU-4 retaining only 11%). Would require re-examining `frame_shuffle` before any claim is made. |

## Note on the honest outcome

The most valuable property of this plan is that **it can fail informatively**. If CTC alignments sit at chance against motion changepoints, that is a real finding: it would mean the joint-CTC chrF++/length-calibration gains in `RESULTS_OVERVIEW.md` Table 5 come from something other than temporal grounding, and the WS-E→WS-F bridge needs a different approach. Log that outcome to `docs/logs/FAILURES.md` and report it — negative results here are paper material (program plan §8 "negative results are publishable"), and they would change the interpretation of the WS-C results too.
