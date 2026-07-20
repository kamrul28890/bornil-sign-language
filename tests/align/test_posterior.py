import bornil.runtime  # noqa: F401

import math

import numpy as np
import torch

from bornil.align.posterior import clip_posterior_stats, aggregate_stats


def _peaked_logprobs(argmax_seq, vocab=6):
    """(T,V) log-probs whose per-frame argmax is argmax_seq[t]."""
    T = len(argmax_seq)
    p = torch.full((T, vocab), 1e-6)
    for t, v in enumerate(argmax_seq):
        p[t, v] = 1.0
    return (p / p.sum(-1, keepdim=True)).log()


def test_blank_rate_counts_blank_argmax_frames():
    lp = _peaked_logprobs([0, 0, 3, 0, 5])   # 3/5 frames are blank
    s = clip_posterior_stats(lp, blank=0)
    assert s["blank_rate"] == 3 / 5


def test_distinct_nonblank_labels():
    lp = _peaked_logprobs([0, 3, 3, 5, 5, 5])
    s = clip_posterior_stats(lp, blank=0)
    assert s["distinct_nonblank"] == 2       # {3, 5}


def test_all_blank_clip_has_zero_distinct_labels():
    lp = _peaked_logprobs([0, 0, 0])
    s = clip_posterior_stats(lp, blank=0)
    assert s["blank_rate"] == 1.0
    assert s["distinct_nonblank"] == 0


def test_mean_argmax_run_length():
    # argmax runs: 3,3,3 | 5 | 3,3  -> lengths 3,1,2 -> mean 2.0
    lp = _peaked_logprobs([3, 3, 3, 5, 3, 3])
    s = clip_posterior_stats(lp, blank=0)
    assert s["mean_run_length"] == 2.0


def test_entropy_is_low_for_peaked_and_high_for_uniform():
    peaked = clip_posterior_stats(_peaked_logprobs([3, 5, 3]), blank=0)
    uniform = clip_posterior_stats(torch.full((3, 6), 1.0 / 6).log(), blank=0)
    assert peaked["mean_entropy"] < 0.1
    # uniform over V=6 has entropy ln(6)
    assert abs(uniform["mean_entropy"] - math.log(6)) < 1e-5
    assert abs(uniform["entropy_ratio"] - 1.0) < 1e-5


def test_entropy_ratio_is_entropy_over_log_vocab():
    s = clip_posterior_stats(_peaked_logprobs([3, 5], vocab=8), blank=0)
    assert abs(s["entropy_ratio"] - s["mean_entropy"] / math.log(8)) < 1e-9


def test_accepts_numpy_and_torch_equally():
    seq = [0, 3, 3, 5]
    a = clip_posterior_stats(_peaked_logprobs(seq), blank=0)
    b = clip_posterior_stats(_peaked_logprobs(seq).numpy(), blank=0)
    assert a == b


def test_non_zero_blank():
    lp = _peaked_logprobs([9, 9, 3], vocab=10)
    s = clip_posterior_stats(lp, blank=9)
    assert s["blank_rate"] == 2 / 3
    assert s["distinct_nonblank"] == 1


def test_aggregate_averages_per_clip_stats():
    a = clip_posterior_stats(_peaked_logprobs([0, 3]), blank=0)
    b = clip_posterior_stats(_peaked_logprobs([5, 5]), blank=0)
    agg = aggregate_stats([a, b])
    assert agg["n_clips"] == 2
    assert abs(agg["blank_rate"] - (a["blank_rate"] + b["blank_rate"]) / 2) < 1e-9
    assert abs(agg["distinct_nonblank"] - (a["distinct_nonblank"] + b["distinct_nonblank"]) / 2) < 1e-9


def test_aggregate_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        aggregate_stats([])
