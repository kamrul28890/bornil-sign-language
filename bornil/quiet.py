"""Silence the benign but very noisy logging emitted during keypoint extraction.

OpenCV/FFmpeg (`[matroska,webm] Unknown-sized element ...`), MediaPipe/glog
(`W0000 ... inference_feedback_manager ...`, XNNPACK INFO), TensorFlow-Lite, and
protobuf all print warnings/info to stderr that are NOT errors — they appear on
every run for everyone. They drown out the actual progress line.

Import this module BEFORE cv2 / mediapipe / bsl_extract so the environment
variables take effect at library load time, then call `install_runtime()` after
the imports to quiet the Python-level loggers too.
"""
import os
import warnings

# Must be set before the native libraries are imported.
os.environ.setdefault("GLOG_minloglevel", "2")        # glog: hide INFO + WARNING
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # TF/TFLite: errors only
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # ffmpeg: quiet
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("GLOG_logtostderr", "0")

warnings.filterwarnings("ignore")


def install_runtime() -> None:
    """Quiet the Python-visible loggers after cv2/absl are importable."""
    try:
        import cv2
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


import contextlib
import sys


@contextlib.contextmanager
def redirect_c_stderr(to_path):
    """Redirect the OS-level stderr (fd 2) to a file for the duration.

    MediaPipe/absl emit C++ warnings straight to fd 2, ignoring Python logging
    config and even GLOG_minloglevel on the Windows pip wheels. Duplicating fd 2
    to a log file sends that native noise to the file while our progress (printed
    to stdout, fd 1) stays clean on the console. Real error detail is preserved
    in the file rather than lost. Falls back to a no-op if fd redirection isn't
    available.
    """
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except Exception:
        yield
        return
    f = open(to_path, "w", encoding="utf-8", errors="replace")
    try:
        os.dup2(f.fileno(), 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved, 2)
        os.close(saved)
        f.close()
