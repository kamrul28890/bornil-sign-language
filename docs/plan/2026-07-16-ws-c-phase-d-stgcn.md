# WS-C Phase D - ST-GCN Skeleton Encoder (T1.1)

> Detailed task plan expanded from `docs/plan/2026-07-16-ws-c-suite-and-benchmark-hardening.md` Part II / Phase D. Execute task-by-task in order, TDD-first, without starting Phase E/F/G.

**Goal:** Add a pure-PyTorch ST-GCN skeleton encoder as another drop-in SLT model, train it as **ST-GCN + joint CTC/attention** on official `random-v1`, and compare it against both the vanilla Transformer baseline and the current TConv+CTC champion using chrF++ as the primary metric.

**Current reference points:**
- Vanilla random-v1: `data_full/runs/c19d0ea08824`, chrF++ 9.646.
- TConv+CTC champion random-v1: `data_full/runs/b0a590ea3317`, chrF++ 10.913.
- Retrieval floor random-v1: chrF++ 9.20.

## Design Decisions

- **Uniform graph tensor from kp-218:** convert `(B,T,218)` to `(B,T,V=65,C=3)` by taking xyz for every joint and dropping pose visibility. Pose visibility is available only for the 23 pose joints; hands have no matching visibility channel. Dropping visibility keeps the graph channels homogeneous and avoids injecting a pose-only feature that would bias graph convolution toward body landmarks.
- **Joint order:** pose landmarks 0-22, left-hand landmarks 0-20, right-hand landmarks 0-20.
- **Adjacency:** fixed graph with self-loops, MediaPipe pose edges restricted to pose landmarks 0-22, standard MediaPipe hand skeleton edges for each hand, and wrist-to-hand-root edges `(pose 15 -> left hand 0)` and `(pose 16 -> right hand 0)`.
- **Normalization:** symmetric degree normalization, `D^-1/2 A D^-1/2`, registered as a non-trainable buffer.
- **Architecture:** ST-GCN blocks alternate spatial graph convolution and temporal convolution over frames. Joint features are pooled across joints per frame to produce `(B,T,d_model)` encoder memory for the existing Transformer decoder.
- **Decode contract:** `SignSTGCN` exposes `forward`, `encode`, `src_padding_mask`, `generate`, `beam_search`, `ctc_logits`, and `forward_with_ctc`; generation delegates to `bornil.models.decode`.
- **Experiment:** run ST-GCN **with CTC on** (`model.ctc_head: true`, `train.ctc_weight: 0.3`) to isolate encoder effects against TConv+CTC. Greedy decode remains official (`beam_size: 1`).

## Task D1 - ST-GCN Tests First

**Files:**
- Create: `tests/models/test_stgcn.py`

**Checks:**
- [x] `kp218_to_xyz65` maps kp-218 to `(B,T,65,3)`, drops pose visibility, and preserves hand xyz.
- [x] `build_stgcn_adjacency` returns a `(65,65)` normalized tensor with self-loops, symmetric edges, pose wrist to hand-root edges, and no trainable parameters.
- [x] `SignSTGCN` returns decoder logits shaped `(B,L,vocab)`, CTC logits shaped `(B,S,vocab)`, and exposes the same decoding methods as the other encoders.
- [x] Tiny synthetic training with CTC decreases loss.

**Run and expected failure:**
- [x] `.venv/Scripts/python -m pytest tests/models/test_stgcn.py -v`
- Expected: fails with `ModuleNotFoundError: No module named 'bornil.models.stgcn'`.

## Task D2 - Implement ST-GCN Encoder

**Files:**
- Create: `bornil/models/stgcn.py`

**Implementation:**
- [x] Add `kp218_to_xyz65(src)` with strict `src.size(-1) == 218` validation.
- [x] Add `build_stgcn_adjacency(normalize=True)` with fixed MediaPipe pose/hand/wrist edges.
- [x] Add pure-torch `GraphConv` and `STGCNBlock` modules.
- [x] Add `SignSTGCN`, matching the existing encoder contract and reusing `PositionalEncoding`, `causal_mask`, and `bornil.models.decode`.
- [x] Add optional CTC head with `use_ctc`, `ctc_logits`, and `forward_with_ctc`.

**Verify:**
- [x] `.venv/Scripts/python -m pytest tests/models/test_stgcn.py -v`
- [x] `.venv/Scripts/python -m pytest tests/models -q`

## Task D3 - Wire `model.arch: stgcn` and Config

**Files:**
- Modify: `scripts/train_sample.py`
- Create: `configs/full_stgcn.yaml`

**Implementation:**
- [x] Add `arch == "stgcn"` to the model factory.
- [x] Pass config-backed keys into `SignSTGCN`: `stgcn_layers`, `stgcn_channels`, `temporal_kernel`, `ctc_head`, plus the shared decoder keys.
- [x] Update experiment-row naming so ST-GCN rows say `ST-GCN` and phase `WS-C T1.1`.
- [x] Create `configs/full_stgcn.yaml` with `model.arch: stgcn`, CTC enabled, `train.ctc_weight: 0.3`, `beam_size: 1`, and official full-data random-v1 settings.

**Verify:**
- [x] `.venv/Scripts/python -m pytest tests/models/test_stgcn.py tests/train/test_ctc_loss.py -q`
- [x] `.venv/Scripts/python -m pytest -q`

**Commit:**
- [x] `git add bornil/models/stgcn.py tests/models/test_stgcn.py scripts/train_sample.py configs/full_stgcn.yaml docs/plan/2026-07-16-ws-c-phase-d-stgcn.md`
- [x] `git commit -m "feat(model): ST-GCN skeleton encoder (WS-C T1.1)"`

## Task D4 - Overfit-Tiny Gate on Real Data

**Command:**
- [x] `.venv/Scripts/python scripts/train_sample.py --config configs/full_stgcn.yaml --out-dir trial_1pct --epochs 3 --device cuda`

**Gate:**
- [x] Confirm epoch loss decreases and is finite before the full run.
- [ ] If loss is NaN or non-decreasing, debug before full training; first fallback is `train.ctc_weight: 0.1` in a smoke-only config override path documented in `FAILURES.md`.

## Task D5 - Full ST-GCN+CTC Random-v1 Run

**Command:**
- [ ] `.venv/Scripts/python scripts/train_sample.py --config configs/full_stgcn.yaml --device cuda`

**Expected artifacts:**
- [ ] New run folder under `data_full/runs/<hash>/`.
- [ ] New `EXP-data_full_stgcn-<hash>` row in `docs/logs/EXPERIMENTS.md`.
- [ ] `metrics.json`, `predictions.csv`, `config.json`, `best.pt`.

**Read correctly:**
- [ ] Judge model on chrF++ first.
- [ ] Compare against retrieval floor chrF++ 9.20 and TConv+CTC chrF++ 10.913.
- [ ] Compute output diversity, top-1 share, mean hypothesis length, and mean reference length.

## Task D6 - Significance and Diagnostics

**Commands:**
- [ ] `.venv/Scripts/python scripts/results_significance.py --run "vanilla=data_full/runs/c19d0ea08824" --run "stgcn=data_full/runs/<hash>" --baseline vanilla --metrics bleu4 chrf++ --n 1000 --out docs/logs/SIGNIFICANCE_STGCN_VS_VANILLA.md`
- [ ] `.venv/Scripts/python scripts/results_significance.py --run "tconv_ctc=data_full/runs/b0a590ea3317" --run "stgcn=data_full/runs/<hash>" --baseline tconv_ctc --metrics bleu4 chrf++ --n 1000 --out docs/logs/SIGNIFICANCE_STGCN_VS_TCONV_CTC.md`

**Diagnostics:**
- [ ] Generate diversity/length stats from `predictions.csv`.
- [ ] Record whether ST-GCN clears the retrieval floor and whether it beats TConv+CTC on chrF++ at p<0.05.

## Task D7 - Logs, Overview, and Commit

**Files:**
- Modify: `docs/logs/RESULTS_OVERVIEW.md`
- Modify: `docs/logs/CHANGELOG.md`
- Modify only if needed: `docs/logs/FAILURES.md`

**Updates:**
- [ ] Add ST-GCN+CTC to the random-v1 ablation table.
- [ ] Add or extend the diversity/length table with ST-GCN diagnostics.
- [ ] Add pointers to ST-GCN significance reports.
- [ ] Add newest CHANGELOG entry with run hash, chrF++, p-values, and diversity/length summary.
- [ ] Log any collapse, NaN, under-generation surprise, or failure in `FAILURES.md`.

**Verify and commit:**
- [ ] `.venv/Scripts/python -m pytest -q`
- [ ] `git add docs/logs/EXPERIMENTS.md docs/logs/RESULTS_OVERVIEW.md docs/logs/CHANGELOG.md docs/logs/SIGNIFICANCE_STGCN_VS_VANILLA.md docs/logs/SIGNIFICANCE_STGCN_VS_TCONV_CTC.md`
- [ ] `git commit -m "docs(results): add ST-GCN random-v1 benchmark"`

## Final Report

Report:
- [ ] ST-GCN EXPERIMENTS row and run hash.
- [ ] chrF++ vs vanilla and vs TConv+CTC with p-values at n=1000.
- [ ] Retrieval-floor result.
- [ ] Diversity/length numbers: unique hypotheses, unique %, top-1 share, mean hypothesis length, mean reference length.
- [ ] Any `FAILURES.md` entries.
- [ ] Commit list.
- [ ] Full suite status.
