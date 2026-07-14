"""Tests for trivial baselines (plan §6.4).

These are the floors a real model must clear: always predicting the most common
training sentence, predicting a random training sentence, and nearest-neighbour
retrieval by pooled video feature. If a model barely beats these, it hasn't
learned to translate — it's exploiting dataset priors.
"""
import numpy as np

from bornil.benchmark import trivial


def test_most_frequent_returns_mode():
    train = ["a", "b", "a", "c", "a"]
    hyps = trivial.most_frequent_hypotheses(train, n=3)
    assert hyps == ["a", "a", "a"]


def test_random_is_from_train_and_deterministic():
    train = ["a", "b", "c", "d"]
    h1 = trivial.random_hypotheses(train, n=5, seed=0)
    h2 = trivial.random_hypotheses(train, n=5, seed=0)
    assert h1 == h2
    assert len(h1) == 5
    assert set(h1) <= set(train)


def test_retrieval_picks_nearest_neighbour():
    train_feats = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 5.0]], dtype=np.float32)
    train_texts = ["origin", "far", "up"]
    test_feats = np.array([[0.1, 0.1], [9.0, 11.0]], dtype=np.float32)
    hyps = trivial.retrieval_hypotheses(train_feats, train_texts, test_feats)
    assert hyps == ["origin", "far"]
