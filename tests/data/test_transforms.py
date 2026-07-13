"""Characterization tests for the keypoint transforms (WS-A).

These lock the *current* behavior of the teammate's normalization + landmark
selection so we can refactor, add representations, and run ablations without
silently changing the model input contract. Numpy-only; no dataset needed.

Layout reminder (per frame, 258 raw values):
  pose 33 x (x,y,z,vis) = 132  | left hand 21 x (x,y,z) = 63 | right hand = 63
  shoulders: pose landmark 11 (L), 12 (R).
"""
import numpy as np
import pytest

from bornil.data import transforms as tf


def _frame_with_shoulders():
    """One (1,258) frame: shoulders at (0.4,0.5)/(0.6,0.5), nose at (0.5,0.3),
    left-hand root at (0.5,0.5,0.2). Center=(0.5,0.5), shoulder width=0.2."""
    x = np.zeros((1, 258), dtype=np.float32)
    # pose landmark 11 (L shoulder): x,y,z,vis
    x[0, 11 * 4:11 * 4 + 4] = [0.4, 0.5, 0.0, 1.0]
    # pose landmark 12 (R shoulder)
    x[0, 12 * 4:12 * 4 + 4] = [0.6, 0.5, 0.0, 1.0]
    # pose landmark 0 (nose)
    x[0, 0:4] = [0.5, 0.3, 0.1, 0.9]
    # left hand landmark 0 (starts at 132)
    x[0, 132:135] = [0.5, 0.5, 0.2]
    return x


# --- dimensions -------------------------------------------------------------

def test_feature_dims():
    assert tf.feature_dim(upper_body=False) == 258
    assert tf.feature_dim(upper_body=True) == 218


def test_select_reduces_258_to_218():
    seq = np.zeros((5, 258), dtype=np.float32)
    assert tf.select_keypoints(seq).shape == (5, 218)


def test_select_is_idempotent_on_218():
    seq = np.zeros((5, 218), dtype=np.float32)
    assert tf.select_keypoints(seq).shape == (5, 218)


def test_select_rejects_bad_dim():
    with pytest.raises(ValueError):
        tf.select_keypoints(np.zeros((5, 100), dtype=np.float32))


# --- normalization ----------------------------------------------------------

def test_normalize_preserves_shape():
    assert tf.normalize_keypoints(_frame_with_shoulders()).shape == (1, 258)


def test_normalize_rejects_bad_shape():
    with pytest.raises(ValueError):
        tf.normalize_keypoints(np.zeros((5, 218), dtype=np.float32))


def test_normalize_anchors_to_shoulder_midpoint():
    out = tf.normalize_keypoints(_frame_with_shoulders())
    # nose: x=(0.5-0.5)*5=0, y=(0.3-0.5)*5=-1.0, z=0.1*5=0.5
    assert out[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert out[0, 1] == pytest.approx(-1.0, abs=1e-5)
    assert out[0, 2] == pytest.approx(0.5, abs=1e-5)


def test_normalize_preserves_visibility_channel():
    out = tf.normalize_keypoints(_frame_with_shoulders())
    assert out[0, 3] == pytest.approx(0.9, abs=1e-6)          # nose visibility
    assert out[0, 11 * 4 + 3] == pytest.approx(1.0, abs=1e-6)  # L-shoulder visibility


def test_normalize_shares_reference_across_hands():
    out = tf.normalize_keypoints(_frame_with_shoulders())
    # left-hand root at (0.5,0.5,0.2): x->0, y->0, z->1.0 (same center/scale as pose)
    assert out[0, 132] == pytest.approx(0.0, abs=1e-5)
    assert out[0, 133] == pytest.approx(0.0, abs=1e-5)
    assert out[0, 134] == pytest.approx(1.0, abs=1e-5)


def test_normalize_leaves_undetected_landmarks_zero():
    out = tf.normalize_keypoints(_frame_with_shoulders())
    # pose landmark 5 was all zeros (undetected) -> stays zero.
    assert np.all(out[0, 5 * 4:5 * 4 + 4] == 0.0)


def test_normalize_passthrough_when_no_pose():
    # No shoulders -> no valid reference -> frame returned unchanged.
    seq = np.zeros((1, 258), dtype=np.float32)
    seq[0, 132:135] = [0.5, 0.5, 0.2]
    out = tf.normalize_keypoints(seq)
    assert np.allclose(out, seq)


def test_prepare_features_normalizes_then_selects():
    out = tf.prepare_features(_frame_with_shoulders())
    assert out.shape == (1, 218)


# --- backward-compat shim ---------------------------------------------------

def test_bsl_utils_shim_reexports():
    import bsl_utils
    assert bsl_utils.feature_dim(upper_body=True) == 218
    assert bsl_utils.SELECTED_FEATURE_DIM == 218
