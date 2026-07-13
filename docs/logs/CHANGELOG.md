# Bornil — Implementation Changelog

> Append-only, newest at top. One entry per meaningful change (code, data, config, decision).
> Keep it terse; link to the experiment log or a commit for detail.

## Format

```
## YYYY-MM-DD — <short title>
- **What:** what changed (files/modules/decision)
- **Why:** motivation / which phase-task this serves
- **Impact:** what now works / what downstream can rely on
- **Commit:** <hash or PR link>
```

---

## 2026-07-13 — Phase 0 (WS-B spine): official splits + metric protocol
- **What:** Created `bornil/` package; `bornil/benchmark/splits.py` (random-v1, signerindep-v1 [chunk-as-session], unseen-sentence-v1 + leakage checks + file I/O) and `bornil/benchmark/metrics.py` (SacreBLEU BLEU-1..4 @ `intl` tokenizer, chrF/chrF++, ROUGE-L, bootstrap CI, NFC-normalized; optional BanglaBERT BERTScore). `scripts/make_splits.py`, 22 passing tests, `pyproject.toml`.
- **Why:** WS-B is the spine everything else scores against (plan §6). Split correctness = no leakage.
- **Impact:** Frozen split files under `splits/` (leakage-checked on all 17,988 clips: random 14390/1799/1799; signerindep 29/4/3 chunks = 14500/1988/1500; unseen-sentence with zero text overlap). Locked, reproducible metric protocol callable via `corpus_metrics()`.
- **Findings:** 36 chunks total; 17,928 unique normalized sentences (near-total uniqueness confirms open-vocabulary framing).
- **Env note:** dev machine has only numpy/pandas/pytest/pyyaml/sacrebleu — no torch/sentencepiece yet; MediaPipe features live off-repo. Model/data-dependent phases run in the cloud/teammate env.
- **Commit:** (this commit)

## 2026-07-13 — Repo synced + research program plan created
- **What:** Merged teammate's keypoint pipeline (`easha003/BdSL`) into `main`; wrote `docs/plan/2026-07-13-bornil-implementation-plan.md` and logging scaffold.
- **Why:** Establish the shared plan of record before Phase 0.
- **Impact:** Team has a single high-level→detailed plan; logging conventions defined.
- **Commit:** d9b2d65
