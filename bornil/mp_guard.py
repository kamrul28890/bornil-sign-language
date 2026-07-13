"""Make MediaPipe import cleanly without a working TensorFlow.

MediaPipe's `__init__` eagerly imports `mediapipe.tasks`, which imports
tensorflow purely for docstring decorators (see its `optional_dependencies.py`).
That module falls back gracefully **only** on `ModuleNotFoundError` — but a
tensorflow whose native DLL fails to load raises a plain `ImportError`, which
leaks through and breaks the entire MediaPipe import. The legacy `solutions`
API (Holistic) does not use tensorflow at all.

This guard installs a meta-path finder that makes any `tensorflow` import raise
`ModuleNotFoundError`, so MediaPipe takes its intended fallback. It only affects
the current process — the system tensorflow install is untouched. Import this
module BEFORE importing mediapipe / bsl_extract.
"""
import sys


class _TensorFlowBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ModuleNotFoundError(f"tensorflow import blocked by bornil.mp_guard: {name}")
        return None


def install() -> None:
    if "tensorflow" in sys.modules:
        return  # already imported successfully somewhere; leave it alone
    if not any(isinstance(f, _TensorFlowBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _TensorFlowBlocker())


install()
