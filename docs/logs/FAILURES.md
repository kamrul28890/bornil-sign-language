# Bornil — Failure & Fix Log

> Failures are data. Every crash, divergence, disappointing result, or dead-end
> gets an entry: what we expected, what happened, why (root cause if known), and
> the fix or next hypothesis. This is what makes the "reasons + possible fixes"
> requirement real, and it becomes the paper's Limitations / negative-results notes.
> Newest at top.

## Entry template

```
### FAIL-<id>  (YYYY-MM-DD)  [phase/task]
- **Expected:** what we thought would happen
- **Observed:** what actually happened (numbers, error, curve shape)
- **Root cause:** confirmed cause, or "hypothesis:" if not yet proven
- **Fix / next step:** what we changed, or the next experiment to isolate it
- **Status:** open | fixed | wontfix | superseded-by <FAIL-id>
- **Links:** EXP-id(s), commit, config
```

---

<!-- Log entries below, newest first -->

### FAIL-003  (2026-07-13)  [trial / extraction, Windows]
- **Expected:** `import mediapipe` to work so keypoint extraction runs.
- **Observed:** intermittent `ImportError: DLL load failed while importing _pywrap_tensorflow_internal` → "Failed to load the native TensorFlow runtime", crashing the whole mediapipe import (0 features produced).
- **Root cause:** confirmed — mediapipe's `__init__` eagerly imports `mediapipe.tasks`, which imports tensorflow only for docstring decorators. Its fallback (`optional_dependencies.py`) catches `ModuleNotFoundError` only; a tensorflow with a broken native DLL raises plain `ImportError`, so it leaks through. Holistic (legacy `solutions` API) never uses tensorflow.
- **Fix / next step:** `bornil/mp_guard.py` installs a meta-path finder that makes any tensorflow import raise `ModuleNotFoundError`, so mediapipe takes its intended fallback. Process-local — the system tensorflow install is untouched (other projects unaffected). Import `bornil.mp_guard` before mediapipe/bsl_extract. Smoke extraction then produced 8/8 features, (128,258), ~6.7 s/clip.
- **Status:** fixed
- **Links:** bornil/mp_guard.py, scripts/extract_sample.py

### FAIL-002  (2026-07-13)  [P0 / env, Windows]
- **Expected:** full `pytest tests/` to pass with exit 0 once torch/sentencepiece tests were added.
- **Observed:** segfault (exit 139) during collection when the full suite loads pandas + torch + sacrebleu + sentencepiece together; each subdir passed alone.
- **Root cause:** confirmed — `import pandas` **before** `import torch` fails torch's native DLL init (`WinError 1114` loading `c10.dll`); the later combined use then crashes. `import torch` first is clean.
- **Fix / next step:** `conftest.py` imports torch at pytest startup (before any test module imports pandas); `bornil/runtime.py` provides the same guard for runtime scripts (`import bornil.runtime` first). Rule: **torch before pandas** everywhere both are used. Full suite now 55 passed, exit 0.
- **Status:** fixed
- **Links:** conftest.py, bornil/runtime.py

### FAIL-001  (2026-07-13)  [P0 / infra]
- **Expected:** file writes to the repo on D: to succeed.
- **Observed:** `ENOSPC: no space left on device`; D: at 100% (0 bytes free).
- **Root cause:** confirmed — D: held `chunks/` (172 GB videos, untracked/gitignored) plus **~86 GB of dangling git objects** (`git count-objects`: 7,476 objects / 86.27 GiB loose, in-pack 0; only **72 reachable** from all refs). The dangling blobs (~52 MB each, thousands of them) came from large files `git add`ed but never committed, then unstaged.
- **Fix / next step:** user freed ~99 GB manually (D: now 79% used) — unblocked. Remaining cleanup to reclaim ~86 GB: `git reflog expire --expire=now --all && git prune --expire=now && git gc` (prune unlinks directly → low temp-space need, safe when near-full). Prevention: ensure `chunks/` and feature dirs stay in `.gitignore` (they are, via merged teammate `.gitignore`); never `git add .` at repo root with videos present.
- **Status:** fixed — ran `git reflog expire --expire=now --all && git prune --expire=now && git gc`. `.git` 87 GB → 2.3 MB; D: 99 GB → 185 GB free (61% used). `fsck` clean, history intact, 42 tests pass.
- **Links:** CHANGELOG 2026-07-13 config/transforms/disk entry
