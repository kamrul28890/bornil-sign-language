"""Generate the figures used by docs/PIPELINE_WALKTHROUGH.md.

Runs the real preprocessing on one sample clip and renders a figure for each
stage: a frame strip, the uniform-sampling timeline, a MediaPipe keypoint
overlay, the 258-d feature layout, the shoulder-anchor normalization (before/
after scatter), and the upper-body landmark selection. Figures are written to
docs/assets/pipeline/ as PNGs. Figure text is ASCII only (Bengali is rendered in
the Markdown, not baked into images, to avoid font issues).

    python scripts/make_pipeline_figures.py
    python scripts/make_pipeline_figures.py --clip chunks/chunk_1/<uuid>.webm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.quiet     # noqa: F401
import bornil.mp_guard  # noqa: F401

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import mediapipe as mp

from bornil.data.transforms import (
    normalize_keypoints, POSE_LEN, HAND_LEN, POSE_LANDMARKS, POSE_KEEP,
    L_SHOULDER, R_SHOULDER,
)
from bsl_extract import sample_frame_indices, keypoints_from_results

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

OUT = os.path.join("docs", "assets", "pipeline")
DEFAULT_CLIP = "chunks/chunk_1/f07dbede-8ed9-4754-aa68-683824f10e16.webm"
MAX_FRAMES = 128


def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def fig_frame_strip(frames, path, k=10):
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    fig, axes = plt.subplots(1, k, figsize=(2 * k, 2.6))
    for ax, i in zip(axes, idx):
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {i}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One clip = {len(frames)} frames of continuous signing "
                 f"(10 shown across the whole clip)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def fig_sampling(total, path, max_frames=MAX_FRAMES):
    kept = sorted(set(sample_frame_indices(total, max_frames)))
    fig, ax = plt.subplots(figsize=(12, 1.8))
    ax.hlines(0, 0, total - 1, color="#cccccc", lw=2, zorder=1)
    ax.vlines(kept, -0.35, 0.35, color="#2563eb", lw=0.8, zorder=2)
    ax.set_xlim(-5, total + 5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("frame index")
    ax.set_title(f"Uniform temporal sampling: {len(kept)} frames kept out of "
                 f"{total} (every ~{total/max_frames:.1f}th frame)")
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _holistic_on(frame_bgr):
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1,
                              refine_face_landmarks=False) as h:
        return h.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def fig_keypoint_overlay(frame_bgr, results, path):
    overlay = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        overlay, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        mp_drawing.draw_landmarks(
            overlay, hand, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style())
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("raw frame")
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("MediaPipe Holistic keypoints\n(33 pose + 21 + 21 hand)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Keypoint extraction: one frame -> 258 numbers", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def fig_feature_layout(path):
    segs = [("pose 33x(x,y,z,visibility) = 132", 0, 132, "#3b82f6"),
            ("left hand 21x(x,y,z) = 63", 132, 195, "#10b981"),
            ("right hand 21x(x,y,z) = 63", 195, 258, "#f59e0b")]
    fig, ax = plt.subplots(figsize=(12, 1.8))
    for label, a, b, c in segs:
        ax.barh(0, b - a, left=a, color=c, edgecolor="white")
        ax.text((a + b) / 2, 0, label, ha="center", va="center", fontsize=10, color="white")
    ax.set_xlim(0, 258)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("index in the 258-dim per-frame feature vector")
    ax.set_title("What one frame becomes: a 258-number vector")
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _pose_xy(vec258):
    pose = vec258[:POSE_LEN].reshape(POSE_LANDMARKS, 4)
    lh = vec258[POSE_LEN:POSE_LEN + HAND_LEN].reshape(21, 3)
    rh = vec258[POSE_LEN + HAND_LEN:].reshape(21, 3)
    return pose, lh, rh


def fig_normalization(vec258, path):
    raw = vec258.astype(np.float32)
    norm = normalize_keypoints(raw[None, :])[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, v, title in ((axes[0], raw, "raw (image coords, 0..1)"),
                         (axes[1], norm, "normalized (shoulder-anchored)")):
        pose, lh, rh = _pose_xy(v)
        present = np.abs(pose).sum(1) > 0
        ax.scatter(pose[present, 0], pose[present, 1], s=18, c="#3b82f6", label="pose")
        for hnd, c, lab in ((lh, "#10b981", "left hand"), (rh, "#f59e0b", "right hand")):
            m = np.abs(hnd).sum(1) > 0
            if m.any():
                ax.scatter(hnd[m, 0], hnd[m, 1], s=10, c=c, label=lab)
        cx = (pose[L_SHOULDER, 0] + pose[R_SHOULDER, 0]) / 2
        cy = (pose[L_SHOULDER, 1] + pose[R_SHOULDER, 1]) / 2
        ax.scatter([cx], [cy], marker="x", s=120, c="red", label="shoulder midpoint")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Normalization: re-center on the shoulder midpoint, scale by "
                 "shoulder width\n(makes it invariant to where the signer stands "
                 "and how big they appear)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def fig_landmark_selection(vec258, path):
    pose, _, _ = _pose_xy(vec258.astype(np.float32))
    fig, ax = plt.subplots(figsize=(5.5, 6))
    keep = set(POSE_KEEP)
    for i in range(POSE_LANDMARKS):
        if np.abs(pose[i]).sum() == 0:
            continue
        kept = i in keep
        ax.scatter(pose[i, 0], pose[i, 1], s=60,
                   c="#10b981" if kept else "#ef4444",
                   marker="o" if kept else "x")
    ax.scatter([], [], c="#10b981", label="kept (0-22: head, arms, hands)")
    ax.scatter([], [], c="#ef4444", marker="x", label="dropped (23-32: hips/legs)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Upper-body landmark selection: 258 -> 218\n"
                 "lower-body pose points have ~0 visibility here")
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    bornil.quiet.install_runtime()

    print("reading frames...", flush=True)
    frames = read_all_frames(args.clip)
    total = len(frames)
    print(f"  {total} frames", flush=True)

    # pick a mid-clip frame likely to have hands raised
    kept = sorted(set(sample_frame_indices(total, MAX_FRAMES)))
    mid = frames[int(0.45 * total)]
    print("running MediaPipe on the sample frame...", flush=True)
    with bornil.quiet.redirect_c_stderr(os.devnull):
        results = _holistic_on(mid)
        vec = np.array(keypoints_from_results(results), dtype=np.float32)

    print("rendering figures...", flush=True)
    fig_frame_strip(frames, os.path.join(OUT, "01_frame_strip.png"))
    fig_sampling(total, os.path.join(OUT, "02_frame_sampling.png"))
    fig_keypoint_overlay(mid, results, os.path.join(OUT, "03_keypoint_overlay.png"))
    fig_feature_layout(os.path.join(OUT, "04_feature_layout.png"))
    fig_normalization(vec, os.path.join(OUT, "05_normalization.png"))
    fig_landmark_selection(vec, os.path.join(OUT, "06_landmark_selection.png"))
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
