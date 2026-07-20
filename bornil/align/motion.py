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
    """Indices of local velocity minima, greedily thinned to `min_distance`.

    Candidates are interior local minima; ties and plateaus keep the earliest
    index. Chosen deepest-first so the strongest minima survive the thinning.
    """
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
