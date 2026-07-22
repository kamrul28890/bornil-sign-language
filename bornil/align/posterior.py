"""CTC posterior sanity statistics (WS-E Task A0).

A0 gates Phase B. Forced alignment (A1) only produces an interpretable
segmentation if the CTC head emits genuine per-frame label evidence. If the head
is degenerate — blank almost everywhere, or a single constant label across a clip
— then Viterbi assigns frames by tie-breaking rather than by signal, and any
downstream agreement-vs-chance in Phase B measures nothing. These statistics
falsify that failure mode cheaply, from an existing checkpoint, no retraining.

Per clip:
- `blank_rate`      fraction of frames whose CTC argmax is blank
- `distinct_nonblank`  count of distinct non-blank argmax labels
- `mean_run_length`    mean length of a constant-argmax run (temporal stickiness)
- `mean_entropy`    mean per-frame posterior entropy (nats)
- `entropy_ratio`   mean_entropy / ln(V); 1.0 means uniform (no information)
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch


def _to_numpy(log_probs) -> np.ndarray:
    if isinstance(log_probs, torch.Tensor):
        lp = log_probs.detach().cpu().numpy()
    else:
        lp = np.asarray(log_probs)
    lp = lp.astype(np.float64, copy=False)
    if lp.ndim != 2:
        raise ValueError(f"log_probs must be (T, V), got {lp.shape}")
    return lp


def clip_posterior_stats(log_probs, blank: int = 0) -> Dict[str, float]:
    """Per-frame posterior statistics for one clip's (T, V) CTC log-probs."""
    lp = _to_numpy(log_probs)
    T, V = lp.shape
    if T == 0:
        raise ValueError("cannot compute stats for zero frames")

    argmax = lp.argmax(axis=1)
    is_blank = argmax == blank
    nonblank = argmax[~is_blank]

    # Run-length: number of maximal constant-argmax segments.
    n_runs = 1 + int(np.count_nonzero(argmax[1:] != argmax[:-1]))

    probs = np.exp(lp)
    # -sum p log p per frame; where p==0 the term is 0 (nan_to_num handles it).
    entropy = -np.nan_to_num(probs * lp, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1)

    return {
        "blank_rate": float(is_blank.mean()),
        "distinct_nonblank": int(np.unique(nonblank).size),
        "mean_run_length": float(T / n_runs),
        "mean_entropy": float(entropy.mean()),
        "entropy_ratio": float(entropy.mean() / math.log(V)),
    }


_MEAN_KEYS = ("blank_rate", "distinct_nonblank", "mean_run_length",
              "mean_entropy", "entropy_ratio")


def aggregate_stats(per_clip: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each statistic across clips, plus the clip count."""
    per_clip = list(per_clip)
    if not per_clip:
        raise ValueError("no clips to aggregate")
    out: Dict[str, float] = {"n_clips": len(per_clip)}
    for k in _MEAN_KEYS:
        out[k] = float(np.mean([c[k] for c in per_clip]))
    return out
