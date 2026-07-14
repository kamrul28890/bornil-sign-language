"""Trivial baselines (plan §6.4) — the floors a real model must clear.

  * most_frequent : always predict the most common training sentence.
  * random        : predict a random training sentence (seeded).
  * retrieval     : nearest-neighbour training sentence by pooled video feature.

If a model scores near these, it is exploiting dataset priors rather than
translating. Report them alongside every model number.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np


def most_frequent_hypotheses(train_texts: Sequence[str], n: int) -> List[str]:
    mode = Counter(train_texts).most_common(1)[0][0]
    return [mode] * n


def random_hypotheses(train_texts: Sequence[str], n: int, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(train_texts), size=n)
    return [train_texts[i] for i in idx]


def retrieval_hypotheses(train_feats: np.ndarray, train_texts: Sequence[str],
                         test_feats: np.ndarray) -> List[str]:
    """1-NN by Euclidean distance in the pooled-feature space.

    train_feats: (Ntrain, D), test_feats: (Ntest, D) — e.g. mean-pooled keypoints.
    Returns the training sentence of the closest train clip for each test clip.
    """
    train_feats = np.asarray(train_feats, dtype=np.float32)
    test_feats = np.asarray(test_feats, dtype=np.float32)
    # (Ntest, Ntrain) squared distances via the expansion |a-b|^2 = |a|^2+|b|^2-2ab
    tr_sq = (train_feats ** 2).sum(1)[None, :]
    te_sq = (test_feats ** 2).sum(1)[:, None]
    d2 = te_sq + tr_sq - 2.0 * test_feats @ train_feats.T
    nn = d2.argmin(axis=1)
    return [train_texts[i] for i in nn]
