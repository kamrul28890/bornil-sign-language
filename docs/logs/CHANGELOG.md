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

## 2026-07-13 — Model layer: tokenizer + SignDataset + Transformer baseline (WS-A/WS-C)
- **What:** Installed CPU torch 2.11 + sentencepiece. `bornil/data/tokenizer.py` (SentencePiece wrapper, BOS/EOS + special-strip; tested on real `bsl_sp.model`). `bornil/data/dataset.py` (`SignDataset` over packed features + `make_collate_fn`; tested on synthetic memmap). `bornil/models/transformer.py` (`SignTransformer` encoder-decoder SLT baseline, sinusoidal PE, causal + padding masks). `conftest.py` + `bornil/runtime.py` enforce torch-before-pandas import order.
- **Why:** First runnable SLT baseline (plan WS-C T0.2) and the data contract feeding it.
- **Impact:** **Overfit-tiny gate automated and passing** — the Transformer memorizes a synthetic clip→subword batch (loss < 0.05, exact teacher-forced reproduction), proving the arch/masking/loss are correct before any GPU/cloud run. Full suite **55 passing, exit 0**.
- **Fix:** resolved a Windows segfault (pandas-before-torch DLL conflict) — see FAIL-002.
- **Commit:** (this commit)

## 2026-07-13 — Phase 0: config system + keypoint-transform tests + disk triage
- **What:** `bornil/config.py` (YAML load/save, deep-merge, stable order-independent config hash) with 7 tests. Moved keypoint transforms to `bornil/data/transforms.py` (canonical) with 13 characterization tests locking normalization/selection behavior; `bsl_utils.py` is now a re-export shim so teammate notebooks keep working. Full suite: **42 passing**.
- **Why:** Config hash tags every future run (plan §4.3); characterization tests let us refactor WS-A + run normalization ablations without silently changing the model input contract.
- **Impact:** `bornil.config` + `bornil.data.transforms` ready for the training loop; input-contract behavior is now regression-guarded.
- **Disk triage:** D: hit 100% full. Root cause: `chunks/` = **172 GB** videos (on-disk here, gitignored) + **~86 GB dangling git objects** (only 72 of 7,476 objects reachable — leftover from large files staged-but-never-committed). User freed ~99 GB manually (now 79% used). **Still recommended:** `git reflog expire --expire=now --all && git prune --expire=now && git gc` to reclaim the ~86 GB (pending user go-ahead). Logged in FAILURES as FAIL-001.
- **Commit:** (this commit)

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
