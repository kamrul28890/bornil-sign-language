"""Extract MediaPipe keypoints for the trial sample (WS-A, plan §5).

Runs the SAME extraction as the full pipeline (bsl_extract) over the sampled
clips, reading videos from the local `chunks/`. Resumable: clips whose .npy
already exists are skipped. Sequential + single-process (simple and robust on
Windows); the full-dataset run uses run_extraction.py's parallel path.

    python scripts/extract_sample.py                 # all clips in sample_manifest
    python scripts/extract_sample.py --limit 8       # first N (smoke)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.mp_guard  # noqa: F401  # block broken-DLL tensorflow before mediapipe

import numpy as np
import pandas as pd

from bsl_extract import extract_keypoints_from_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="trial_5pct")
    ap.add_argument("--video-root", default=".", help="root that holds chunks/")
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.out_dir, "sample_manifest.csv"))
    if args.limit:
        man = man.head(args.limit)
    feat_dir = os.path.join(args.out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    n = len(man)
    ok = skip = miss = fail = 0
    t0 = time.time()
    for i, r in enumerate(man.itertuples(index=False), 1):
        dst = os.path.join(feat_dir, r.recording.replace(".webm", ".npy"))
        if os.path.exists(dst):
            skip += 1
        else:
            vp = os.path.join(args.video_root, r.chunk_path, r.recording)
            if not os.path.exists(vp):
                miss += 1
                print(f"  MISSING {vp}", flush=True)
            else:
                try:
                    feats = extract_keypoints_from_video(vp, max_frames=args.max_frames)
                    np.save(dst, feats)
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"  FAIL {r.recording}: {e!r}", flush=True)
        if i % 5 == 0 or i == n:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (n - i) / rate / 60 if rate else 0
            pct = 100.0 * i / n
            print(f"  [{pct:5.1f}%] {i}/{n} | {rate:.2f} clip/s | ETA {eta:.1f} min | "
                  f"ok={ok} skip={skip} miss={miss} fail={fail}", flush=True)

    print(f"DONE in {(time.time()-t0)/60:.1f} min | "
          f"ok={ok} skip={skip} miss={miss} fail={fail} -> {feat_dir}", flush=True)


if __name__ == "__main__":
    main()
