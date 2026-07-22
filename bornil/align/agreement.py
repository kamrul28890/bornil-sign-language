"""Boundary agreement between two point sets, with randomized null baselines
(WS-E §9 validation).

There is no gold alignment for Bornil, so a raw agreement number means nothing on
its own — a few points on a short timeline agree by luck. Every agreement F1 must
be read against a null of the same sizes on the same timeline. Two nulls:

- `chance_boundary_f1`  both point sets drawn uniformly at random. The weak null.
- `shift_null_f1`       the real reference held fixed; the prediction circularly
                        shifted by a random offset, preserving its own spacing /
                        clustering. The stronger null — it removes only the
                        *phase* relationship, not the density or bunching that can
                        inflate the uniform null when both signals cluster.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def boundary_f1(pred: Sequence[int], ref: Sequence[int],
                tolerance: int = 2) -> Dict[str, float]:
    """F1 of predicted boundaries against reference boundaries.

    A prediction matches a reference within +/- tolerance frames; each reference
    is consumed at most once, matched globally nearest-first.
    """
    pred = list(pred)
    ref = list(ref)
    n_pred, n_ref = len(pred), len(ref)
    if n_pred == 0 or n_ref == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": n_pred, "n_ref": n_ref}

    pairs = []
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            d = abs(p - r)
            if d <= tolerance:
                pairs.append((d, i, j))
    pairs.sort()

    used_pred, used_ref = set(), set()
    matched = 0
    for _d, i, j in pairs:
        if i in used_pred or j in used_ref:
            continue
        used_pred.add(i)
        used_ref.add(j)
        matched += 1

    precision = matched / n_pred
    recall = matched / n_ref
    f1 = (2 * precision * recall / (precision + recall)) if matched else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "n_pred": n_pred, "n_ref": n_ref}


def _random_points(n: int, n_frames: int, rng) -> List[int]:
    if n <= 0:
        return []
    n = min(n, n_frames)
    return sorted(int(x) for x in rng.choice(n_frames, size=n, replace=False))


def chance_boundary_f1(n_pred: int, n_ref: int, n_frames: int, tolerance: int = 2,
                       trials: int = 200, seed: int = 0) -> float:
    """Mean F1 of two uniformly-random boundary sets of the given sizes."""
    if n_pred == 0 or n_ref == 0 or n_frames <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    scores = [
        boundary_f1(_random_points(n_pred, n_frames, rng),
                    _random_points(n_ref, n_frames, rng), tolerance)["f1"]
        for _ in range(trials)
    ]
    return float(np.mean(scores))


def shift_null_f1(pred: Sequence[int], ref: Sequence[int], n_frames: int,
                  tolerance: int = 2, trials: int = 200, seed: int = 0) -> float:
    """Mean F1 of `ref` against `pred` circularly shifted by a random offset.

    Preserves the prediction's internal spacing, so it controls for clustering
    that the uniform null does not.
    """
    pred = list(pred)
    ref = list(ref)
    if not pred or not ref or n_frames <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        off = int(rng.integers(n_frames))
        shifted = sorted((p + off) % n_frames for p in pred)
        scores.append(boundary_f1(shifted, ref, tolerance)["f1"])
    return float(np.mean(scores))
