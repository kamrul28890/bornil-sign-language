import numpy as np

from bornil.align.motion import changepoints, frame_velocity


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
