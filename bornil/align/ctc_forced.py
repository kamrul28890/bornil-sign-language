"""CTC forced alignment (WS-E §9.2.1): frames -> subword labels.

Given per-frame CTC log-probabilities from a trained joint-CTC encoder and the
known target subword sequence, Viterbi decoding over the CTC lattice assigns
every frame to either a label or blank. This yields a frame-level alignment
"for free" from a model trained with sentence-only supervision — Bornil has no
glosses and no timings, so this is the cheapest route to temporal grounding.

Conventions (fixed by WS-C Phase B): blank == pad_id == 0; `labels` excludes
BOS/EOS/pad. Extended sequence is [b, y1, b, ..., yL, b]; odd extended indices
are labels, and extended index s maps to label index (s-1)//2.

The recursion is vectorized over extended states: Phase B's agreement-vs-chance
needs many randomized re-alignments, and a per-(t,s) Python loop measured 0.63 s
per clip at T=512 vs 8.3 ms here (~76x). See the plan amendments in
docs/plan/2026-07-16-ws-e-alignment.md.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

NEG_INF = -np.inf


class AlignmentError(Exception):
    """Raised when the label sequence cannot be aligned to the frame budget."""


def _extended(labels: Sequence[int], blank: int) -> np.ndarray:
    ext = np.empty(2 * len(labels) + 1, dtype=np.int64)
    ext[0::2] = blank
    ext[1::2] = labels
    return ext


def forced_align(log_probs, labels: Sequence[int], blank: int = 0) -> List[int]:
    """Viterbi-align `labels` to frames. Returns the per-frame extended-state path.

    log_probs: (T, V) log-softmax values. Raises AlignmentError when infeasible
    (too few frames for the labels, including blanks required between repeats).
    """
    if isinstance(log_probs, torch.Tensor):
        if log_probs.dim() != 2:
            raise ValueError(f"log_probs must be (T, V), got {tuple(log_probs.shape)}")
        lp = log_probs.detach().float().cpu().numpy()
    else:
        lp = np.asarray(log_probs, dtype=np.float64)
        if lp.ndim != 2:
            raise ValueError(f"log_probs must be (T, V), got {lp.shape}")

    labels = [int(y) for y in labels]
    if not labels:
        raise AlignmentError("cannot align an empty label sequence")

    ext = _extended(labels, blank)
    T, S = lp.shape[0], len(ext)
    if T == 0:
        raise AlignmentError("cannot align to zero frames")
    # Each label needs a frame, plus a separating blank between equal neighbours.
    min_frames = len(labels) + sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    if T < min_frames:
        raise AlignmentError(
            f"infeasible alignment: {T} frames cannot cover {len(labels)} labels "
            f"(needs >= {min_frames})")

    emit = lp[:, ext]                       # (T, S) gather the emission once
    # A skip from s-2 is legal only into a non-blank state whose label differs
    # from the one two states back -- this is what forbids collapsing repeats.
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    alpha = np.full((T, S), NEG_INF, dtype=np.float64)
    back = np.zeros((T, S), dtype=np.int64)
    alpha[0, :min(2, S)] = emit[0, :min(2, S)]

    states = np.arange(S)
    for t in range(1, T):
        prev_alpha = alpha[t - 1]
        stay = prev_alpha
        prev = np.concatenate(([NEG_INF], prev_alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([NEG_INF, NEG_INF], prev_alpha[:-2])),
                        NEG_INF)
        cand = np.stack([stay, prev, skip])          # (3, S)
        arg = cand.argmax(axis=0)
        best = cand[arg, states]
        alpha[t] = np.where(np.isneginf(best), NEG_INF, best + emit[t])
        back[t] = states + np.array([0, -1, -2])[arg]

    last, prev_end = alpha[T - 1, S - 1], alpha[T - 1, S - 2] if S > 1 else NEG_INF
    if np.isneginf(last) and np.isneginf(prev_end):
        raise AlignmentError(
            f"infeasible alignment: {T} frames cannot cover {len(labels)} labels")

    s = S - 1 if last >= prev_end else S - 2
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = int(s)
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
        if current == li:
            spans[-1] = (li, spans[-1][1], t)
        else:
            spans.append((li, t, t))
            current = li
    return spans


def align_clip(log_probs, labels: Sequence[int],
               blank: int = 0) -> List[Tuple[int, int, int]]:
    """forced_align + path_to_spans."""
    return path_to_spans(forced_align(log_probs, labels, blank=blank), labels)
