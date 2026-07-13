"""Backward-compatibility shim.

The canonical keypoint layout / normalization / selection logic now lives in
`bornil.data.transforms`. This module re-exports it so existing notebooks and
scripts that do `from bsl_utils import prepare_features, feature_dim` keep
working unchanged. Prefer importing from `bornil.data.transforms` in new code.
"""

from bornil.data.transforms import (  # noqa: F401
    FEATURE_DIM,
    HAND_LANDMARKS,
    HAND_LEN,
    L_SHOULDER,
    POSE_KEEP,
    POSE_LANDMARKS,
    POSE_LEN,
    R_SHOULDER,
    SELECTED_FEATURE_DIM,
    SELECTED_POSE_LEN,
    feature_dim,
    normalize_keypoints,
    prepare_features,
    select_keypoints,
)
