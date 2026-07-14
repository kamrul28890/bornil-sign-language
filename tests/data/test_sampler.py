"""Tests for length-bucketed batching (WS-C training efficiency).

Grouping clips of similar length into a batch means we pad a 160-frame clip next
to other ~160-frame clips instead of next to a 512-frame one — far less wasted
compute on variable-length keypoint sequences. The sampler must still cover every
clip exactly once and stay stochastic across epochs.
"""
import numpy as np

from bornil.data.sampler import LengthBucketBatchSampler


def _lengths(n=53, seed=0):
    return list(np.random.default_rng(seed).integers(20, 500, size=n))


def test_covers_every_index_exactly_once():
    lens = _lengths(53)
    s = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=1)
    seen = [i for batch in s for i in batch]
    assert sorted(seen) == list(range(len(lens)))


def test_batch_count_matches_len():
    lens = _lengths(53)
    s = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=1)
    batches = list(s)
    assert len(batches) == len(s) == (53 + 7) // 8


def test_no_batch_exceeds_batch_size():
    lens = _lengths(53)
    s = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=1)
    assert all(len(b) <= 8 for b in s)


def test_bucketing_groups_similar_lengths():
    # shuffle=False + a bucket covering everything => batches are globally
    # length-sorted, so the per-batch lengths come out non-decreasing.
    lens = _lengths(40)
    s = LengthBucketBatchSampler(lens, batch_size=5, shuffle=False,
                                 bucket_multiplier=100)
    flat = [lens[i] for batch in s for i in batch]
    assert flat == sorted(flat)


def test_deterministic_first_epoch():
    lens = _lengths(53)
    a = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=7)
    b = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=7)
    assert list(a) == list(b)


def test_epochs_differ_when_shuffled():
    lens = _lengths(53)
    s = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=7)
    assert list(s) != list(s)   # second epoch reshuffles


def test_drop_last():
    lens = _lengths(53)
    s = LengthBucketBatchSampler(lens, batch_size=8, shuffle=True, seed=1, drop_last=True)
    batches = list(s)
    assert all(len(b) == 8 for b in batches)
    assert len(batches) == 53 // 8
