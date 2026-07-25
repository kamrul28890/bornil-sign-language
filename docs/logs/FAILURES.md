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

### FAIL-005  (2026-07-19, logged 2026-07-25)  [WS-C T1.1 / ST-GCN encoder]
- **Expected:** a spatio-temporal graph encoder (explicit joint-graph structure + temporal convolution over it) would match or beat TConv+CTC (chrF++ 10.913), since it models motion more directly than a flat keypoint sequence.
- **Observed:** ST-GCN+CTC scored chrF++ 8.265 on random-v1 — significantly below both vanilla (p=0.000) and TConv+CTC (p=0.000, `docs/logs/SIGNIFICANCE_STGCN_VS_VANILLA.md` / `SIGNIFICANCE_STGCN_VS_TCONV_CTC.md`), below both trivial floors on chrF++, after 42.8h training (~57x TConv+CTC's 44.8min for a worse result). BLEU-4 not significant vs either. Diversity is not collapsed (93.3% unique hyps, in line with vanilla) but 11.3% of rows are runaway repeated-token hypotheses — a failure mode not seen in the other full runs.
- **Root cause:** hypothesis. `docs/logs/INPUT_ABLATION.md` shows the exploitable signal on this benchmark is largely static pose (frame-order shuffling costs the CTC model only −0.88 chrF++). A spatio-temporal graph encoder's entire value proposition is modeling motion structure over the joint graph, so it has little to win on here relative to a simpler encoder that happens to already capture per-frame pose well.
- **Fix / next step:** do not tune ST-GCN further; closes WS-C T1.1 as a negative result. If motion-sensitive architectures are revisited, they should be motivated by a benchmark/training change that first makes motion informative (e.g. sign-level segmentation targets), not applied to the current static-pose-dominated setup.
- **Status:** wontfix (negative result; publishable as a Limitations / negative-results note alongside FAIL-004)
- **Links:** EXP-data_full_stgcn-2dfdcbdf4e2d; `docs/logs/SIGNIFICANCE_STGCN_VS_VANILLA.md`, `docs/logs/SIGNIFICANCE_STGCN_VS_TCONV_CTC.md`, `docs/logs/INPUT_ABLATION.md`

### FAIL-004  (2026-07-20)  [WS-E Phase B / alignment validation]
- **Expected:** CTC forced-alignment placements would agree with keypoint-motion changepoints above chance (pre-registered prediction: "weak but above chance"), validating the alignments as temporal grounding.
- **Observed:** at chance on all three CTC runs. `ratio_shift` (mean F1 / circular-shift null) = 0.94/0.89/0.96 for confident anchors vs velocity minima and 0.86–0.94 for all labels; identical picture against velocity maxima except that anchors *avoid* peaks strongly and reproducibly (0.28/0.30/0.30×). A first run looked even lower but was confounded by a saturated reference (36.9 changepoints/clip with ±2 tolerance ≈ tiling); fixing the changepoint detector (smoothing + keep only deepest-quantile minima → 11.3/clip) did not lift agreement above chance.
- **Root cause:** hypothesis. The forced alignment does not localize CTC subword spikes to sign-boundary motion rests. Not attributable to a dead metric — the same metric detects a strong negative association with velocity peaks, so it has power. Leading explanations, unseparated by this test: (1) subword≠sign granularity; (2) anchors sit mid-articulation (they avoid both extrema of motion), so velocity minima are the wrong target; (3) the head's temporal responsiveness (A0) is about *which* labels fire, not *when*.
- **Fix / next step:** do not proceed to Phase C or pseudo-glosses on this signal (plan gate). Grounding itself is NOT refuted — the input ablation (mismatch −4.71 chrF++) and A0 (56% distinct-label drop under frame shuffle) stand. Any further temporal-grounding work should target sign-level segmentation with an onset/edge reference rather than subword-vs-rest. Full analysis: `docs/logs/ALIGNMENT_VALIDATION.md`.
- **Status:** wontfix (negative result; publishable as a Limitations / negative-results note)
- **Links:** EXP b0a590ea3317 / 40cffb9e62bb / a5a9060499a4; `docs/logs/ALIGNMENT_VALIDATION.md`, `docs/logs/WSE_A0_POSTERIOR.md`, `docs/logs/INPUT_ABLATION.md`

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
