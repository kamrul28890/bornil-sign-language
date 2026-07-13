"""Parallel, resumable MediaPipe feature extraction for the full BSL dataset.

Runs the SAME extraction as the notebook (via bsl_extract.py) across multiple
processes. Safe to interrupt and re-run: videos whose .npy already exists are
skipped, so a re-run continues where it left off.

    python3 run_extraction.py                 # full dataset, default workers
    python3 run_extraction.py --limit 12      # first 12 not-yet-done (smoke test)
    python3 run_extraction.py --workers 6

Uses the 'spawn' start method (required for MediaPipe on macOS), so the worker
imports its extraction code from bsl_extract (must be importable, not inline).
"""
import os

# Keep each worker single-threaded for BLAS/OpenMP so N workers don't oversubscribe
# the cores (MediaPipe parallelism is handled per-process). Set before numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

from bsl_extract import extract_keypoints_from_video

ROOT = "/Volumes/Easha/SSD Backup/Bornil_Dataset/archive"
CSV_PATH = os.path.join(ROOT, "video_data.csv")
FEATURE_DIR = "/Volumes/Easha/SSD Backup/Bornil_Dataset/features_mediapipe_v2"
MANIFEST_PATH = os.path.join(ROOT, "processed_manifest.csv")
MAX_FRAMES = 128


def _feature_path(recording):
    return os.path.join(FEATURE_DIR, recording.replace(".webm", ".npy"))


def worker(task):
    chunk_path, recording = task
    vp = os.path.join(ROOT, chunk_path, recording)
    sp = _feature_path(recording)
    try:
        if os.path.exists(sp):
            return ("skip", recording)
        if not os.path.exists(vp):
            return ("missing", recording)
        feats = extract_keypoints_from_video(vp, max_frames=MAX_FRAMES)
        np.save(sp, feats)
        return ("ok", recording)
    except Exception as e:  # one bad video must not kill the run
        return ("fail", f"{recording}: {e!r}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N not-yet-done videos")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    os.makedirs(FEATURE_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    tasks, already = [], 0
    for _, r in df.iterrows():
        if os.path.exists(_feature_path(r["recording"])):
            already += 1
        else:
            tasks.append((r["chunk_path"], r["recording"]))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    log(f"total={len(df)} already_done={already} todo={len(tasks)} workers={args.workers}")

    counts = {"ok": 0, "skip": 0, "missing": 0, "fail": 0}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
            counts[status] += 1
            if status == "fail":
                log(f"FAIL {info}")
            if i % 100 == 0 or i == len(tasks):
                el = time.time() - t0
                rate = i / el if el else 0
                eta_h = (len(tasks) - i) / rate / 3600 if rate else 0
                log(f"{i}/{len(tasks)} | {rate:.2f} vid/s | ETA {eta_h:.1f} h | "
                    f"ok={counts['ok']} fail={counts['fail']} missing={counts['missing']}")

    # Rebuild the manifest from everything on disk (only on full runs).
    if args.limit is None:
        allrows = pd.read_csv(CSV_PATH)
        allrows["clean_text"] = allrows["text"].apply(lambda t: " ".join(str(t).strip().split()))
        allrows["feature_path"] = allrows["recording"].apply(_feature_path)
        dfp = allrows[allrows["feature_path"].apply(os.path.exists)].copy()
        cols = ["text", "clean_text", "recording", "chunk_path", "feature_path"]
        dfp[cols].to_csv(MANIFEST_PATH, index=False)
        log(f"manifest rows={len(dfp)} -> {MANIFEST_PATH}")

    log(f"EXTRACTION DONE in {(time.time() - t0) / 3600:.2f} h | "
        f"ok={counts['ok']} skip={counts['skip']} missing={counts['missing']} fail={counts['fail']}")


if __name__ == "__main__":
    main()
