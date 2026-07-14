"""Extract MediaPipe keypoints for the trial sample (WS-A, plan §5).

Runs the SAME extraction as the full pipeline (bsl_extract) over the sampled
clips, reading videos from the local `chunks/`. Resumable: clips whose .npy
already exists are skipped.

MediaPipe Holistic is CPU-only on Windows (the GPU can't help here), so speed
comes from running many clips in parallel across CPU cores. Use --workers to
control it (default: cpu_count - 2). --workers 1 forces the simple sequential
path and mirrors native warnings to <out_dir>/extract_stderr.log.

    python scripts/extract_sample.py --out-dir trial_2pct            # parallel
    python scripts/extract_sample.py --out-dir trial_2pct --workers 8
    python scripts/extract_sample.py --out-dir trial_2pct --limit 8  # smoke
"""
import argparse
import os
import sys
import time

# Keep each worker single-threaded so N parallel workers don't oversubscribe the
# cores (MediaPipe/BLAS otherwise each spawn many threads -> 50+ threads on 16
# cores -> thrashing at 100% CPU with low throughput). Set BEFORE numpy/mediapipe.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.quiet     # noqa: F401  # silence benign cv2/mediapipe/tf/protobuf noise
import bornil.mp_guard  # noqa: F401  # block broken-DLL tensorflow before mediapipe

import multiprocessing as mp

import numpy as np
import pandas as pd

from bsl_extract import extract_keypoints_from_video

MAX_FRAMES_DEFAULT = 128


def _worker(task):
    """Extract one clip. Native stderr noise is sent to the null device so the
    parallel console stays clean; failures are returned as messages instead."""
    dst, video_path, max_frames = task
    try:
        with bornil.quiet.redirect_c_stderr(os.devnull):
            feats = extract_keypoints_from_video(video_path, max_frames=max_frames)
        np.save(dst, feats)
        return ("ok", dst)
    except Exception as e:
        return ("fail", f"{os.path.basename(video_path)}: {e!r}")


def _build_tasks(man, feat_dir, video_root, max_frames):
    """Return (tasks_to_do, skip_count, missing_list)."""
    tasks, skip, missing = [], 0, []
    for r in man.itertuples(index=False):
        dst = os.path.join(feat_dir, r.recording.replace(".webm", ".npy"))
        if os.path.exists(dst):
            skip += 1
            continue
        vp = os.path.join(video_root, r.chunk_path, r.recording)
        if not os.path.exists(vp):
            missing.append(vp)
            continue
        tasks.append((dst, vp, max_frames))
    return tasks, skip, missing


def _progress(done, total, t0, ok, fail, skip, miss):
    el = time.time() - t0
    rate = done / el if el else 0
    eta = (total - done) / rate / 60 if rate else 0
    pct = 100.0 * done / total if total else 100.0
    print(f"  [{pct:5.1f}%] {done}/{total} | {rate:.2f} clip/s | ETA {eta:.1f} min | "
          f"ok={ok} skip={skip} miss={miss} fail={fail}", flush=True)


def main():
    bornil.quiet.install_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="trial_5pct")
    ap.add_argument("--video-root", default=".", help="root that holds chunks/")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel CPU workers (default cpu_count-2; 1 = sequential)")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.out_dir, "sample_manifest.csv"))
    if args.limit:
        man = man.head(args.limit)
    feat_dir = os.path.join(args.out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    tasks, skip, missing = _build_tasks(man, feat_dir, args.video_root, args.max_frames)
    for vp in missing:
        print(f"  MISSING {vp}", flush=True)
    todo = len(tasks)
    ok = fail = 0
    miss = len(missing)
    workers = min(args.workers, todo) if todo else 1
    print(f"clips: total={len(man)} already={skip} todo={todo} missing={miss} "
          f"| workers={workers}", flush=True)

    t0 = time.time()
    if todo == 0:
        pass
    elif workers <= 1:
        # Sequential: keep native noise in a single log file.
        with bornil.quiet.redirect_c_stderr(os.path.join(args.out_dir, "extract_stderr.log")):
            for i, task in enumerate(tasks, 1):
                status, info = _worker(task)
                if status == "ok":
                    ok += 1
                else:
                    fail += 1
                    print(f"  FAIL {info}", flush=True)
                if i % 5 == 0 or i == todo:
                    _progress(i, todo, t0, ok, fail, skip, miss)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, (status, info) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
                if status == "ok":
                    ok += 1
                else:
                    fail += 1
                    print(f"  FAIL {info}", flush=True)
                if i % 5 == 0 or i == todo:
                    _progress(i, todo, t0, ok, fail, skip, miss)

    print(f"DONE in {(time.time()-t0)/60:.1f} min | "
          f"ok={ok} skip={skip} miss={miss} fail={fail} -> {feat_dir}", flush=True)


if __name__ == "__main__":
    main()
