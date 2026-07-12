"""MediaPipe keypoint extraction — single source of truth.

Imported by bsl_preprocessing_clean.ipynb and by run_extraction.py (parallel
full-dataset extraction). Output per frame is 258 RAW values:
pose 33x4 + left hand 21x3 + right hand 21x3. Frames are uniformly sampled
to at most max_frames across the whole clip.
"""
import cv2
import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic


def sample_frame_indices(total_frames, max_frames):
    """Which frame indices to keep from a clip.

    If the clip has <= max_frames frames, keep them all. Otherwise pick
    max_frames indices spread uniformly across the whole clip (first..last),
    so the entire sign is covered instead of only the first few seconds.
    """
    if max_frames is None or total_frames <= max_frames:
        return list(range(total_frames))
    idx = np.linspace(0, total_frames - 1, max_frames)
    return sorted(set(int(round(i)) for i in idx))


def keypoints_from_results(results):
    """Flatten one frame's holistic result into the 258-value vector."""
    keypoints = []

    # Pose: 33 landmarks x 4 (x, y, z, visibility) = 132
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0.0] * 33 * 4)

    # Left hand: 21 landmarks x 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 21 * 3)

    # Right hand: 21 landmarks x 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 21 * 3)

    return keypoints


def count_frames(video_path):
    # CAP_PROP_FRAME_COUNT and the CSV frame_count are unreliable for .webm
    # (often off by a few), so count the actually-decodable frames by reading
    # through once. This keeps the uniform sampling landing on real frames and
    # covering the true end of the clip.
    cap = cv2.VideoCapture(video_path)
    n = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n += 1
    cap.release()
    return n


def extract_keypoints_from_video(video_path, max_frames=None):
    total_frames = count_frames(video_path)
    keep = set(sample_frame_indices(total_frames, max_frames))

    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only run the (expensive) pose/hand model on sampled frames.
            if frame_idx in keep:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                all_keypoints.append(keypoints_from_results(results))

            frame_idx += 1

    cap.release()
    return np.array(all_keypoints, dtype=np.float32)
