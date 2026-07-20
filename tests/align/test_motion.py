import numpy as np

from bornil.align.motion import changepoints, frame_velocity, smooth_velocity


def test_velocity_length_matches_frames():
    feats = np.random.RandomState(0).randn(20, 6)
    v = frame_velocity(feats)
    assert v.shape == (20,)


def test_velocity_zero_for_static_clip():
    feats = np.ones((10, 4))
    assert np.allclose(frame_velocity(feats), 0.0)


def test_changepoints_find_motion_minima():
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


def test_velocity_reflects_motion_magnitude():
    # A point moving 1 unit/frame in x has per-frame diff norm 1 / sqrt(D).
    feats = np.zeros((5, 4))
    feats[:, 0] = np.arange(5)
    v = frame_velocity(feats)
    assert np.allclose(v[1:], 1.0 / np.sqrt(4))


def test_changepoints_short_clip_returns_empty():
    assert changepoints(np.array([1.0, 0.0]), min_distance=3) == []


def test_changepoints_sorted_and_unique():
    rng = np.random.RandomState(1)
    v = rng.rand(60)
    cps = changepoints(v, min_distance=3)
    assert cps == sorted(cps)
    assert len(cps) == len(set(cps))
    # min-distance is respected globally
    assert all(b - a >= 3 for a, b in zip(cps, cps[1:]))


def test_velocity_rejects_non_2d():
    import pytest
    with pytest.raises(ValueError):
        frame_velocity(np.zeros((3, 4, 5)))


def test_smooth_preserves_length_and_noop_for_window_one():
    v = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    assert smooth_velocity(v, window=1).shape == v.shape
    assert np.allclose(smooth_velocity(v, window=1), v)
    assert smooth_velocity(v, window=3).shape == v.shape


def test_smooth_reduces_single_frame_spikes():
    v = np.ones(21)
    v[10] = 100.0
    sm = smooth_velocity(v, window=5)
    assert sm[10] < v[10]           # spike attenuated
    assert abs(sm.mean() - v.mean()) < 1e-9  # average preserved (edge-normalized)


def test_smooth_constant_is_unchanged():
    v = np.full(15, 3.0)
    assert np.allclose(smooth_velocity(v, window=5), 3.0)


def test_quantile_filters_shallow_minima():
    # Spread signal so the median sits between the two minima. Deep rest at
    # idx 15 (v=0.1), shallow dip at idx 5 (v=6) with a median of 2.0.
    v = np.array([8, 8, 8, 8, 8, 6, 8, 8, 8, 8,
                  2, 2, 2, 2, 2, 0.1, 2, 2, 2, 2, 2], dtype=float)
    assert np.median(v) == 2.0
    deep_only = changepoints(v, min_distance=3, quantile=0.5)
    assert 15 in deep_only and 5 not in deep_only
    # Without the filter, both are local minima.
    both = changepoints(v, min_distance=3)
    assert 5 in both and 15 in both


def test_quantile_yields_fewer_changepoints_on_noisy_signal():
    rng = np.random.RandomState(3)
    v = rng.rand(200)
    many = changepoints(v, min_distance=3)
    few = changepoints(v, min_distance=3, quantile=0.4)
    assert len(few) < len(many)
