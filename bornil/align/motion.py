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


def smooth_velocity(velocity: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average; length preserved. window<=1 is a no-op.

    Raw per-frame velocity is jittery, so its local minima are dominated by
    single-frame noise rather than sign rests. Smoothing first makes the minima
    correspond to actual decelerations.
    """
    v = np.asarray(velocity, dtype=np.float64)
    if window <= 1 or v.size == 0:
        return v
    window = min(window, v.size)
    kernel = np.ones(window) / window
    # 'same' keeps length T; edge division corrects the shorter edge windows.
    smoothed = np.convolve(v, kernel, mode="same")
    norm = np.convolve(np.ones_like(v), kernel, mode="same")
    return smoothed / norm


def changepoints(velocity: np.ndarray, min_distance: int = 3,
                 quantile: float = None) -> List[int]:
    """Indices of local velocity minima, greedily thinned to `min_distance`.

    Candidates are interior local minima; ties and plateaus keep the earliest
    index. Chosen deepest-first so the strongest minima survive the thinning.

    `quantile` (0..1), when given, keeps only minima whose velocity is at or
    below that quantile of the whole clip — i.e. real rests, not shallow dips.
    This is what stops the changepoint set from tiling the timeline and
    saturating the agreement metric.
    """
    v = np.asarray(velocity, dtype=np.float64)
    if v.size < 3:
        return []
    cand = [i for i in range(1, v.size - 1) if v[i] <= v[i - 1] and v[i] <= v[i + 1]]
    if quantile is not None and cand:
        thresh = float(np.quantile(v, quantile))
        cand = [i for i in cand if v[i] <= thresh]
    cand.sort(key=lambda i: v[i])
    chosen: List[int] = []
    for i in cand:
        if all(abs(i - c) >= min_distance for c in chosen):
            chosen.append(i)
    return sorted(chosen)
