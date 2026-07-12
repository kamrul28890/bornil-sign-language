"""Shared utilities for the BSL preprocessing / modeling pipeline.

Keep the keypoint layout and normalization logic here (one source of truth) so
the preprocessing notebook, the training dataset, and any inference code all
agree on exactly how features are interpreted and normalized.
"""

import numpy as np

# Feature layout per frame (258 values), as produced by the preprocessing
# notebook's extract_keypoints_from_video:
#   pose       : 33 landmarks x 4 (x, y, z, visibility) = 132   -> [0:132]
#   left  hand : 21 landmarks x 3 (x, y, z)             = 63    -> [132:195]
#   right hand : 21 landmarks x 3 (x, y, z)             = 63    -> [195:258]
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_LEN = POSE_LANDMARKS * 4          # 132
HAND_LEN = HAND_LANDMARKS * 3          # 63
FEATURE_DIM = POSE_LEN + 2 * HAND_LEN  # 258

# MediaPipe Pose landmark indices used as the body anchor.
L_SHOULDER = 11
R_SHOULDER = 12

_EPS = 1e-6

# --- Landmark selection for modeling ---------------------------------------
# Drop pose landmarks 23-32 (hips, knees, ankles, heels, feet). In this
# upper-body dataset MediaPipe reports mean visibility <= 0.08 for them and
# places them below the frame -- i.e. extrapolated noise. The 23 kept landmarks
# (head, shoulders, arms, hands; indices 0..22) all have visibility >= 0.5.
#
# IMPORTANT: the stored .npy files keep ALL 33 pose landmarks (raw). This
# selection is applied at load time, so POSE_KEEP can change here WITHOUT
# re-running MediaPipe, and every consumer derives its input width from
# SELECTED_FEATURE_DIM below (never a hard-coded number) so nothing drifts.
POSE_KEEP = list(range(23))                               # indices 0..22
SELECTED_POSE_LEN = len(POSE_KEEP) * 4                    # 92
SELECTED_FEATURE_DIM = SELECTED_POSE_LEN + 2 * HAND_LEN   # 218


def _normalize_block(block, center, inv_scale, valid):
    """Normalize one landmark block in place-safe fashion.

    block     : (T, N, C) with C == 4 (pose: x, y, z, visibility) or C == 3 (hand: x, y, z)
    center    : (T, 2) per-frame shoulder midpoint (x, y)
    inv_scale : (T,)   per-frame 1 / shoulder_width
    valid     : (T,)   frames that have a usable pose reference

    Returns a new array. x, y are translated + scaled; z is scaled only (pose
    and hand z use different origins, so a shared translation is meaningless).
    A pose block's visibility channel is left untouched. Landmarks that are
    zero-filled (undetected) and frames without a valid reference are unchanged.
    """
    present = np.abs(block).sum(axis=2) > 0            # (T, N) real vs zero-filled landmark
    keep = (present & valid[:, None])[..., None]       # (T, N, 1)

    out = block.copy()                                 # preserves visibility (channel 3)
    out[..., 0] = (block[..., 0] - center[:, None, 0]) * inv_scale[:, None]
    out[..., 1] = (block[..., 1] - center[:, None, 1]) * inv_scale[:, None]
    out[..., 2] = block[..., 2] * inv_scale[:, None]

    return np.where(keep, out, block)


def normalize_keypoints(seq):
    """Body-anchor normalize a (T, FEATURE_DIM) keypoint sequence.

    For each frame we translate by the shoulder midpoint and scale by the
    shoulder width, applying the SAME per-frame reference to the pose and both
    hands. This makes the representation invariant to where the signer stands in
    the frame and how large they appear, while preserving hand/body geometry.

    The (T, 258) shape is unchanged, so this is a drop-in transform at load time.
    Kept deliberately raw-in / normalized-out (no in-place mutation of the input)
    so it is safe to call inside a Dataset.__getitem__.

    Notes:
      * pose visibility (every 4th pose value) is left untouched
      * undetected (zero-filled) hands/pose stay zero
      * frames with no pose or a near-zero shoulder width are returned unchanged
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (T, {FEATURE_DIM}), got {seq.shape}")
    seq = seq.copy()

    pose = seq[:, :POSE_LEN].reshape(-1, POSE_LANDMARKS, 4)
    lh = seq[:, POSE_LEN:POSE_LEN + HAND_LEN].reshape(-1, HAND_LANDMARKS, 3)
    rh = seq[:, POSE_LEN + HAND_LEN:].reshape(-1, HAND_LANDMARKS, 3)

    # Per-frame reference from the shoulders (only where the pose was detected).
    pose_present = np.abs(pose).sum(axis=(1, 2)) > 0
    center = (pose[:, L_SHOULDER, :2] + pose[:, R_SHOULDER, :2]) / 2.0      # (T, 2)
    scale = np.linalg.norm(pose[:, L_SHOULDER, :2] - pose[:, R_SHOULDER, :2], axis=1)  # (T,)
    valid = pose_present & (scale > _EPS)
    inv_scale = np.where(valid, 1.0 / np.where(scale > _EPS, scale, 1.0), 0.0)

    pose = _normalize_block(pose, center, inv_scale, valid)
    lh = _normalize_block(lh, center, inv_scale, valid)
    rh = _normalize_block(rh, center, inv_scale, valid)

    seq[:, :POSE_LEN] = pose.reshape(-1, POSE_LEN)
    seq[:, POSE_LEN:POSE_LEN + HAND_LEN] = lh.reshape(-1, HAND_LEN)
    seq[:, POSE_LEN + HAND_LEN:] = rh.reshape(-1, HAND_LEN)
    return seq


def select_keypoints(seq):
    """Reduce a (T, FEATURE_DIM) sequence to the upper-body modeling landmarks.

    Drops the low-visibility lower-body pose landmarks (keeps POSE_KEEP) and
    leaves both hands untouched, returning (T, SELECTED_FEATURE_DIM).

    Apply AFTER normalize_keypoints (which needs the full pose for the shoulder
    anchor). Idempotent: an already-selected sequence is returned unchanged, so
    it is safe to call more than once.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[1] == SELECTED_FEATURE_DIM:
        return seq
    if seq.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"expected (T, {FEATURE_DIM}) or (T, {SELECTED_FEATURE_DIM}), got {seq.shape}"
        )
    pose = seq[:, :POSE_LEN].reshape(-1, POSE_LANDMARKS, 4)[:, POSE_KEEP, :]
    pose = pose.reshape(seq.shape[0], SELECTED_POSE_LEN)
    hands = seq[:, POSE_LEN:]
    return np.concatenate([pose, hands], axis=1)


def prepare_features(seq, normalize=True, upper_body=True):
    """Single load-time transform: normalize then select.

    Use this everywhere features are loaded (training, validation, inference) so
    the model input contract lives in one place. Pair with feature_dim() to size
    the model's input layer instead of hard-coding a width.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if normalize:
        seq = normalize_keypoints(seq)
    if upper_body:
        seq = select_keypoints(seq)
    return seq


def feature_dim(upper_body=True):
    """Per-frame input width the model should expect for the given options."""
    return SELECTED_FEATURE_DIM if upper_body else FEATURE_DIM
