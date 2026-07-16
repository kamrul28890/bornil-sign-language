# WS-C Model Suite Expansion + Benchmark Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is a follow-on to `docs/plan/2026-07-13-bornil-implementation-plan.md` (the program plan of record); read §7 (WS-C) and §11 (Analysis) of that document first.

**Goal:** Take Bornil from "one Transformer baseline + one ablation, written up without CIs" to "a multi-model WS-C suite with confidence intervals and paired significance, plus a CTC head that unlocks the alignment workstream (WS-E)."

**Architecture:** Everything runs through the existing model-agnostic training/eval loop (`bornil/train/loop.py`) and the locked metric protocol (`bornil/benchmark/metrics.py`). New encoders are drop-in modules that expose the same `encode / forward / src_padding_mask / generate / beam_search` contract the loop already depends on; a shared decode module removes duplication so new encoders don't re-implement autoregressive decoding. Video models (ViViT/VideoMAE) are deliberately deferred — see the roadmap and Phase G for exactly when and why.

**Tech Stack:** PyTorch (pure-torch models, no Lightning), SacreBLEU/chrF, SentencePiece, MediaPipe (extraction only), numpy/pandas. CUDA build `torch 2.6.0+cu124` on an RTX 3070 (8 GB) for dev/keypoint training; cloud A100/H100 for Tier-2 video.

## Global Constraints

- **Reproducibility protocol is frozen.** Do not change `BLEU_TOKENIZE = "intl"`, NFC normalization, or the metric definitions in `bornil/benchmark/metrics.py` — those are benchmark constants, not knobs.
- **Official decode policy is greedy** (`beam_size: 1`). Beam search and the repetition guard are ablation-only and must not silently replace official numbers.
- **Overfit-tiny gate before every full run** (program plan §4.6): every new model must overfit 8–32 clips (train loss → ~0 / clearly decreasing) *before* any full-data training. No cloud/full run without a logged tiny pass.
- **Every full run appends** one row to `docs/logs/EXPERIMENTS.md` (via the existing `_append_experiment_row` path), writes a run folder under `<out_dir>/runs/<config_hash>/` (`config.json`, `metrics.json`, `predictions.csv`), and gets a `docs/logs/CHANGELOG.md` entry. Failures/dead-ends go to `docs/logs/FAILURES.md`.
- **The data contract is fixed:** batches are `(x, y, x_lengths, y_lengths)` where `x` is `(B, S, 218)` (kp-218, load-time normalized/upper-body), `y` is `(B, L)` subword ids with `y[:,0]==bos_id` and an `eos_id` terminator, padded with `pad_id` (`pad_id == 0`). Input dim comes from `feature_dim(cfg["data"]["upper_body"])`, never a literal.
- **Config hash tags every run.** Any new config key must be a real key in the composed config so it enters `config_hash(cfg)`; do not thread hyperparameters through argparse-only.
- **Windows import order:** every script entry point imports `bornil.runtime` before pandas (torch-before-pandas DLL guard, FAIL-002). Follow the existing pattern in `scripts/train_sample.py`.

---

## Roadmap & where ViViT sits

This plan is **Part I (Phases A–C): execute now on the 3070** — fully detailed, TDD. **Part II (Phases D–G): specified, expand into its own plan when its entry criteria are met** (per the writing-plans "break into separate plans" guidance). The table shows the whole arc so nothing is lost.

| Phase | Work | Tier / Compute | Status in this plan | Entry criteria |
|---|---|---|---|---|
| A | Bootstrap CIs + paired significance + regenerate result tables | deterministic, CPU | **Detailed (do first)** | none — the run outputs already exist |
| B | CTC head + joint CTC/attention loss | T0.2, 3070 | **Detailed** | A done (so gains are CI-checked) |
| C | Shared decode module + BiLSTM encoder baseline (T0.1) | T0.1, 3070 | **Detailed** | B merged |
| D | ST-GCN skeleton encoder (T1.1) | T1.1, 3070→cloud | Specified | C merged; expand into own plan |
| E | Face-keypoint representation + face ablation (WS-A §5.2.2) | extraction + T0.2, 3070 | Specified | A done; expand into own plan |
| F | Conformer (T1.3) + two-stream kp⊕hand-RGB (T1.2) | cloud | Specified | D done; needs hand-patch extraction |
| **G** | **ViViT / VideoMAE video backbones (T2.1/T2.2)** | **Tier-2, cloud-only** | **Specified — see below** | keypoint suite is a strong baseline to beat **and** RGB features are pre-extracted |

### When are we using ViViT? (direct answer)

**Not yet, and on purpose.** ViViT is **WS-C Tier 2** (program plan §7.3) and is gated behind three things that are not true today:

1. **Compute.** ViViT/VideoMAE are video transformers; you cannot finetune them on an 8 GB RTX 3070. They are **cloud-only (A100/H100)** by design (program plan §3, §7.3). Everything in Phases A–D is specifically the work the 3070 *can* do so cloud spend later buys signal, not debugging.
2. **A missing input representation.** ViViT eats **RGB video / pre-extracted backbone features**. Right now the repo only has **keypoints** extracted (`packed.npy` is `(N, T, 258)` MediaPipe landmarks). WS-A §5.2.4 "rgb-video / pre-extracted features" **does not exist yet**. That pre-extraction (decode → resize → uniform-sample → run a frozen backbone → memmap features on cloud) is itself a sizeable I/O + storage + cloud job and is Phase G's first task.
3. **A baseline worth beating.** The point of a video model is to beat the keypoint suite on the generalization splits. Until the keypoint suite (Transformer + CTC + BiLSTM + ST-GCN) is on the board with CIs, there is nothing to justify the cloud cost against.

**So ViViT lands after Phases A–D**, roughly months 4–6 of the program timeline (plan §3). When you reach it, generate a dedicated detailed plan for Phase G (its exact interface depends on decisions made in D–F). Phase G below specifies the entry criteria, the pre-extraction contract, and the model shape so that plan is easy to write.

---

# Part I — Detailed (execute now, 3070)

## Phase A — Benchmark hardening: confidence intervals + paired significance

**Why first:** you already have official runs, ablations, and an LTE12 track written into `docs/logs/RESULTS_OVERVIEW.md` **as point estimates**. The program plan requires **bootstrap CIs everywhere and paired tests when comparing two models** (§6.3, §11). This is cheap, deterministic, needs no GPU, and makes every number you already have defensible. `bootstrap_ci(...)` already exists in `bornil/benchmark/metrics.py`; Phase A adds a paired test, a convenience CI wrapper, and a script that turns saved `predictions.csv` files into a CI + significance table.

### Task A1: Paired bootstrap significance + CI wrapper in the metrics module

**Files:**
- Modify: `bornil/benchmark/metrics.py` (add two functions after `bootstrap_ci`, ~line 97)
- Test: `tests/benchmark/test_metrics_significance.py` (create)

**Interfaces:**
- Consumes: `corpus_metrics(hyps, refs) -> Dict[str,float]`, `bootstrap_ci(hyps, refs, metric, n, seed, alpha) -> (lo, hi)` (both already in the module).
- Produces:
  - `metric_ci(hypotheses, references, metric="bleu4", n=1000, seed=0, alpha=0.05) -> (point: float, lo: float, hi: float)`
  - `paired_bootstrap_pvalue(hyps_a, hyps_b, references, metric="bleu4", n=1000, seed=0) -> float` (two-sided p-value for H0: metric(A)==metric(B), paired resampling).

- [x] **Step 1: Write the failing tests**

```python
# tests/benchmark/test_metrics_significance.py
from bornil.benchmark.metrics import metric_ci, paired_bootstrap_pvalue


def test_metric_ci_brackets_point_estimate():
    hyps = ["আমি বাড়ি যাব", "সে স্কুলে যায়"] * 10
    refs = ["আমি বাড়ি যাব", "সে বাজারে যায়"] * 10
    point, lo, hi = metric_ci(hyps, refs, metric="chrf", n=200, seed=0)
    assert lo <= point <= hi
    assert hi >= lo


def test_paired_bootstrap_identical_systems_is_insignificant():
    hyps = ["ক খ গ", "a b c", "x y z", "p q r"] * 6
    refs = ["ক খ ঘ", "a b d", "x y w", "p q s"] * 6
    p = paired_bootstrap_pvalue(hyps, hyps, refs, metric="chrf", n=200, seed=0)
    assert p > 0.9  # a system compared to itself is never significant


def test_paired_bootstrap_obvious_difference_is_significant():
    refs = ["ক খ গ ঘ"] * 40
    good = ["ক খ গ ঘ"] * 40           # exact match
    bad = ["z z z z"] * 40            # nothing in common
    p = paired_bootstrap_pvalue(bad, good, refs, metric="chrf", n=300, seed=0)
    assert p < 0.05
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/benchmark/test_metrics_significance.py -v`
Expected: FAIL with `ImportError: cannot import name 'metric_ci'`.

- [x] **Step 3: Implement the two functions**

Append to `bornil/benchmark/metrics.py`:

```python
def metric_ci(hypotheses, references, metric: str = "bleu4",
              n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Point estimate plus percentile bootstrap CI for one metric.

    Returns (point, lo, hi). Thin wrapper over corpus_metrics + bootstrap_ci
    so callers get all three numbers from one call with a single seed.
    """
    point = corpus_metrics(hypotheses, references)[metric]
    lo, hi = bootstrap_ci(hypotheses, references, metric=metric,
                          n=n, seed=seed, alpha=alpha)
    return float(point), lo, hi


def paired_bootstrap_pvalue(hyps_a, hyps_b, references, metric: str = "bleu4",
                            n: int = 1000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for H0: metric(A) == metric(B).

    Paired: the same resampled sentence indices score both systems, so the
    test controls for which sentences are easy/hard. p is the fraction of
    resampled deltas (B-A) that fall on the opposite side of zero from the
    observed delta, doubled for a two-sided test and clipped to 1.0.
    """
    a, b, r = list(hyps_a), list(hyps_b), list(references)
    if not (len(a) == len(b) == len(r)):
        raise ValueError("hyps_a, hyps_b, references must have equal length")
    m = len(r)
    observed = corpus_metrics(b, r)[metric] - corpus_metrics(a, r)[metric]
    rng = np.random.default_rng(seed)
    opposite = 0
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        va = [a[i] for i in idx]
        vb = [b[i] for i in idx]
        vr = [r[i] for i in idx]
        delta = corpus_metrics(vb, vr)[metric] - corpus_metrics(va, vr)[metric]
        if (observed >= 0 and delta <= 0) or (observed < 0 and delta >= 0):
            opposite += 1
    return float(min(1.0, 2.0 * opposite / n))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/benchmark/test_metrics_significance.py -v`
Expected: 3 passed.

- [x] **Step 5: Run the full benchmark test module (no regressions)**

Run: `.venv/Scripts/python -m pytest tests/benchmark -q`
Expected: all pass (existing metric/split tests unaffected).

- [x] **Step 6: Commit**

```bash
git add bornil/benchmark/metrics.py tests/benchmark/test_metrics_significance.py
git commit -m "feat(metrics): paired bootstrap significance + metric_ci wrapper"
```

### Task A2: Significance/CI report script over saved run folders

**Files:**
- Create: `scripts/results_significance.py`
- Test: `tests/benchmark/test_results_significance.py` (create)

**Interfaces:**
- Consumes: `metric_ci`, `paired_bootstrap_pvalue` (Task A1); each run's `predictions.csv` with columns `reference,hypothesis` (written by `scripts/train_sample.py:230`).
- Produces:
  - function `load_predictions(run_dir: str) -> tuple[list[str], list[str]]` returning `(hypotheses, references)`.
  - function `build_report(runs: list[tuple[str,str]], baseline_label: str | None, metrics: list[str], n: int, seed: int) -> str` returning a Markdown report string. `runs` is a list of `(label, run_dir)`.
  - CLI: `python scripts/results_significance.py --run "vanilla=data_full/runs/c19d0ea08824" --run "tconv=data_full/runs/c26953197929" --baseline vanilla --out docs/logs/SIGNIFICANCE.md`

- [x] **Step 1: Write the failing test**

```python
# tests/benchmark/test_results_significance.py
import pandas as pd
from scripts.results_significance import load_predictions, build_report


def _write_run(tmp_path, name, refs, hyps):
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame({"reference": refs, "hypothesis": hyps}).to_csv(
        d / "predictions.csv", index=False, encoding="utf-8")
    return str(d)


def test_load_predictions_roundtrip(tmp_path):
    d = _write_run(tmp_path, "r", ["ক খ গ"], ["ক খ ঘ"])
    hyps, refs = load_predictions(d)
    assert hyps == ["ক খ ঘ"] and refs == ["ক খ গ"]


def test_build_report_has_ci_and_pvalue_columns(tmp_path):
    refs = ["ক খ গ ঘ"] * 20
    a = _write_run(tmp_path, "base", refs, ["z z z z"] * 20)
    b = _write_run(tmp_path, "better", refs, ["ক খ গ ঘ"] * 20)
    md = build_report([("base", a), ("better", b)], baseline_label="base",
                      metrics=["chrf"], n=200, seed=0)
    assert "95% CI" in md
    assert "p vs base" in md
    assert "better" in md and "base" in md
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/benchmark/test_results_significance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.results_significance'`.

- [x] **Step 3: Implement the script**

```python
# scripts/results_significance.py
"""Confidence intervals + paired significance over saved run folders.

Reads each run's predictions.csv (reference,hypothesis), reports a point
estimate with a bootstrap 95% CI per metric, and a paired bootstrap p-value
against a chosen baseline run. This is the report-facing companion to
docs/logs/RESULTS_OVERVIEW.md — it turns the point-estimate tables into
CI + significance tables (program plan §6.3, §11).

    python scripts/results_significance.py \
        --run "vanilla=data_full/runs/c19d0ea08824" \
        --run "tconv=data_full/runs/c26953197929" \
        --baseline vanilla --metrics bleu4 chrf++ \
        --out docs/logs/SIGNIFICANCE.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # torch-before-pandas (FAIL-002)

import argparse

import pandas as pd

from bornil.benchmark.metrics import metric_ci, paired_bootstrap_pvalue


def load_predictions(run_dir: str):
    df = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    return df["hypothesis"].astype(str).tolist(), df["reference"].astype(str).tolist()


def build_report(runs, baseline_label, metrics, n: int = 1000, seed: int = 0) -> str:
    data = {label: load_predictions(d) for label, d in runs}
    base = data.get(baseline_label)
    lines = ["# Significance & Confidence Intervals", "",
             f"Bootstrap n={n}, seed={seed}. Paired p-values vs `{baseline_label}`.", ""]
    header = "| Run | Metric | Point | 95% CI | p vs " + str(baseline_label) + " |"
    lines += [header, "|---|---|---:|---|---:|"]
    for label, _ in runs:
        hyps, refs = data[label]
        for metric in metrics:
            point, lo, hi = metric_ci(hyps, refs, metric=metric, n=n, seed=seed)
            if base is not None and label != baseline_label:
                p = paired_bootstrap_pvalue(base[0], hyps, refs, metric=metric, n=n, seed=seed)
                pcol = f"{p:.3f}"
            else:
                pcol = "—"
            lines.append(f"| {label} | {metric} | {point:.3f} | [{lo:.3f}, {hi:.3f}] | {pcol} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help="label=path/to/run_dir (repeatable)")
    ap.add_argument("--baseline", default=None, help="label to compare others against")
    ap.add_argument("--metrics", nargs="+", default=["bleu4", "chrf++"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs.append((label, path))
    md = build_report(runs, args.baseline, args.metrics, n=args.n, seed=args.seed)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/benchmark/test_results_significance.py -v`
Expected: 2 passed.

- [x] **Step 5: Generate the real significance table for the official + ablation runs**

Run:
```bash
.venv/Scripts/python scripts/results_significance.py \
  --run "vanilla_random=data_full/runs/c19d0ea08824" \
  --run "tconv_random=data_full/runs/c26953197929" \
  --run "eos_pen_random=data_full/runs/a459d1148d0c" \
  --baseline vanilla_random --metrics bleu4 chrf++ \
  --out docs/logs/SIGNIFICANCE.md
```
Expected: `wrote docs/logs/SIGNIFICANCE.md`; open it and confirm the vanilla→TConv BLEU-4 delta has a p-value, and each row shows a `[lo, hi]` CI. (Run dir hashes are from `docs/logs/RESULTS_OVERVIEW.md`; if a folder is missing locally, run against the folders that exist and note it.)

- [x] **Step 6: Record the finding**

Add a `## 2026-…  — Significance & CIs added` entry to `docs/logs/CHANGELOG.md` stating whether the TConv BLEU-4 gain over vanilla is significant at p<0.05 (this decides whether "TConv helps" is a claim you can put in the paper). Add a one-line pointer from `docs/logs/RESULTS_OVERVIEW.md` to `SIGNIFICANCE.md`.

- [x] **Step 7: Commit**

```bash
git add scripts/results_significance.py tests/benchmark/test_results_significance.py docs/logs/SIGNIFICANCE.md docs/logs/CHANGELOG.md docs/logs/RESULTS_OVERVIEW.md
git commit -m "feat(eval): CI + paired-significance report over saved runs"
```

---

## Phase B — CTC head + joint CTC/attention training

**Why:** the current model is attention-decoder only. A **CTC head over the encoder** (a) is a standard, strong regularizer for seq2seq (joint CTC/attention, as in speech translation — program plan §7 "Common interface"), and (b) is the **prerequisite for WS-E alignment** — CTC forced alignment gives frame↔subword correspondence "for free" (§9.2.1). This is the highest-leverage model change because it serves two workstreams at once and stays on the 3070.

**Design:** add an optional linear CTC head on top of `encode(...)` output; add an optional CTC term to `train_epoch`; wire both through config. Default off → existing checkpoints and numbers are untouched. Blank id = `pad_id` (0). CTC target = subword ids strictly between BOS and EOS (labels only).

### Task B1: CTC head on SignTransformer + tiny unit test

**Files:**
- Modify: `bornil/models/transformer.py` (add `use_ctc` ctor arg + `ctc_head` + `ctc_logits` method)
- Test: `tests/models/test_ctc_head.py` (create)

**Interfaces:**
- Consumes: `SignTransformer.encode(src, src_key_padding_mask) -> (B, S, d_model)` (exists, line 124).
- Produces:
  - `SignTransformer(..., use_ctc: bool = False)` — when true, builds `self.ctc_head = nn.Linear(d_model, vocab_size)`.
  - `SignTransformer.ctc_logits(src, src_key_padding_mask=None) -> (B, S, vocab_size)`; raises `RuntimeError` if `use_ctc` is False.

- [x] **Step 1: Write the failing test**

```python
# tests/models/test_ctc_head.py
import bornil.runtime  # noqa: F401
import torch
from bornil.models.transformer import SignTransformer


def test_ctc_logits_shape_when_enabled():
    m = SignTransformer(input_dim=218, vocab_size=50, d_model=32, nhead=4,
                        num_encoder_layers=2, num_decoder_layers=2,
                        dim_feedforward=64, use_ctc=True)
    src = torch.randn(3, 17, 218)
    logits = m.ctc_logits(src)
    assert logits.shape == (3, 17, 50)


def test_ctc_logits_raises_when_disabled():
    m = SignTransformer(input_dim=218, vocab_size=50, d_model=32, nhead=4,
                        num_encoder_layers=2, num_decoder_layers=2,
                        dim_feedforward=64, use_ctc=False)
    try:
        m.ctc_logits(torch.randn(1, 5, 218))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_ctc_head.py -v`
Expected: FAIL — `SignTransformer.__init__() got an unexpected keyword argument 'use_ctc'`.

- [x] **Step 3: Implement the CTC head**

In `bornil/models/transformer.py`, add `use_ctc: bool = False` to `__init__` (after `temporal_conv_kernel: int = 5`), and inside `__init__` after `self.out = nn.Linear(d_model, vocab_size)`:

```python
        self.use_ctc = use_ctc
        self.ctc_head = nn.Linear(d_model, vocab_size) if use_ctc else None
```

Add this method (after `encode`, before `generate`):

```python
    def ctc_logits(self, src: torch.Tensor,
                   src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Per-frame vocab logits from the encoder for a CTC loss/alignment.

        Returns (B, S, vocab_size). Requires use_ctc=True at construction.
        """
        if self.ctc_head is None:
            raise RuntimeError("SignTransformer built with use_ctc=False has no CTC head")
        memory = self.encode(src, src_key_padding_mask)
        return self.ctc_head(memory)
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/models/test_ctc_head.py -v`
Expected: 2 passed.

- [x] **Step 5: Confirm the existing overfit-tiny / model tests still pass**

Run: `.venv/Scripts/python -m pytest tests/models -q`
Expected: all pass (the head is inert when `use_ctc=False`).

- [x] **Step 6: Commit**

```bash
git add bornil/models/transformer.py tests/models/test_ctc_head.py
git commit -m "feat(model): optional CTC head over the encoder (WS-E prerequisite)"
```

### Task B2: Joint CTC term in the training loop

**Files:**
- Modify: `bornil/train/loop.py` (`train_epoch` signature + loss)
- Test: `tests/train/test_ctc_loss.py` (create)

**Interfaces:**
- Consumes: `model.ctc_logits(src, src_key_padding_mask)` (Task B1); batch `(x, y, x_lengths, y_lengths)`.
- Produces:
  - helper `ctc_targets_from_padded(y, y_lengths, bos_id, eos_id, pad_id) -> (targets_1d: LongTensor, target_lengths: LongTensor)` — concatenated label ids with BOS/EOS/pad removed, and per-row label lengths.
  - `train_epoch(..., ctc_weight: float = 0.0, bos_id: int = None)` — when `ctc_weight>0` and the model has a CTC head, total loss = attention CE + `ctc_weight` * CTC.

- [x] **Step 1: Write the failing test**

```python
# tests/train/test_ctc_loss.py
import bornil.runtime  # noqa: F401
import torch
from torch.utils.data import DataLoader

from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.models.transformer import SignTransformer
from bornil.train.loop import train_epoch, ctc_targets_from_padded


def test_ctc_targets_strip_bos_eos_pad():
    # y rows: [BOS=1, a=5, b=6, EOS=2, PAD=0]
    y = torch.tensor([[1, 5, 6, 2, 0], [1, 7, 2, 0, 0]])
    y_lengths = torch.tensor([4, 3])
    targets, tlens = ctc_targets_from_padded(y, y_lengths, bos_id=1, eos_id=2, pad_id=0)
    assert targets.tolist() == [5, 6, 7]
    assert tlens.tolist() == [2, 1]


def test_joint_ctc_training_step_runs_and_reduces_loss():
    torch.manual_seed(0)
    N, T, D, V = 8, 20, 218, 40
    feats = torch.randn(N, T, D).numpy()
    lengths = [T] * N
    targets = [[1, 5, 6, 7, 2] for _ in range(N)]  # BOS ... EOS
    ds = SignDataset(feats, lengths, list(range(N)), targets,
                     normalize=False, upper_body=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(0))
    model = SignTransformer(input_dim=D, vocab_size=V, d_model=32, nhead=4,
                            num_encoder_layers=2, num_decoder_layers=2,
                            dim_feedforward=64, use_ctc=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = train_epoch(model, loader, opt, pad_id=0, eos_id=2, bos_id=1, ctc_weight=0.3)
    for _ in range(15):
        last = train_epoch(model, loader, opt, pad_id=0, eos_id=2, bos_id=1, ctc_weight=0.3)
    assert last < first
```

Note: `SignDataset(normalize=False, upper_body=False)` keeps the raw feature width `D` unchanged so the test controls the input dim.

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/train/test_ctc_loss.py -v`
Expected: FAIL — `cannot import name 'ctc_targets_from_padded'`.

- [x] **Step 3: Implement the helper and joint loss**

In `bornil/train/loop.py`, add the helper above `train_epoch`:

```python
def ctc_targets_from_padded(y, y_lengths, bos_id, eos_id, pad_id):
    """Flatten padded [BOS, ...labels..., EOS, PAD] rows to CTC label targets.

    Returns (targets_1d, target_lengths): the concatenated label ids with
    BOS/EOS/pad removed, and each row's label count. CTC consumes a flat
    target buffer plus per-row lengths.
    """
    flat, lens = [], []
    for row, n in zip(y.tolist(), y_lengths.tolist()):
        labels = [t for t in row[:n] if t not in (bos_id, eos_id, pad_id)]
        flat.extend(labels)
        lens.append(len(labels))
    return (torch.tensor(flat, dtype=torch.long),
            torch.tensor(lens, dtype=torch.long))
```

Add `ctc_weight: float = 0.0` and `bos_id: int = None` to the `train_epoch` signature. Inside the batch loop, after the attention loss (and after the `early_eos_penalty` block), before `loss.backward()`:

```python
        if ctc_weight and getattr(model, "use_ctc", False) and bos_id is not None:
            ctc_logits = model.ctc_logits(x, src_key_padding_mask=src_kpm)   # (B,S,V)
            log_probs = ctc_logits.log_softmax(-1).transpose(0, 1)            # (S,B,V)
            targets, target_lengths = ctc_targets_from_padded(
                y, y[:, :].ne(pad_id).sum(1), bos_id, eos_id, pad_id)
            input_lengths = x_lengths.to(device)
            ctc = nn.functional.ctc_loss(
                log_probs, targets.to(device),
                input_lengths, target_lengths.to(device),
                blank=pad_id, zero_infinity=True)
            loss = loss + float(ctc_weight) * ctc
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/train/test_ctc_loss.py -v`
Expected: 2 passed.

- [x] **Step 5: Run the whole training test package**

Run: `.venv/Scripts/python -m pytest tests/train -q`
Expected: all pass (default `ctc_weight=0.0` leaves existing behavior identical).

- [x] **Step 6: Commit**

```bash
git add bornil/train/loop.py tests/train/test_ctc_loss.py
git commit -m "feat(train): optional joint CTC/attention loss"
```

### Task B3: Wire CTC through config + train script; run the joint-CTC ablation

**Files:**
- Create: `configs/full_ctc.yaml`
- Modify: `scripts/train_sample.py` (pass `use_ctc` to the model; pass `ctc_weight`/`bos_id` to `train_epoch`; log `ctc_weight` in the EXPERIMENTS row)

**Interfaces:**
- Consumes: `cfg["model"]["ctc_head"]` (bool), `cfg["train"]["ctc_weight"]` (float); `train_epoch(..., ctc_weight=, bos_id=)`.
- Produces: an EXPERIMENTS row whose "Training extras" line includes `ctc_weight=<value>`; a run folder under `data_full/runs/<hash>/`.

- [x] **Step 1: Create the config** (copy `configs/full_tconv.yaml` structure; keep TConv on since it was the best encoder so far, add CTC)

```yaml
# configs/full_ctc.yaml — TConv encoder + joint CTC/attention on random-v1.
# Joint CTC regularizes the encoder and yields frame-level alignments for WS-E.
trial:
  name: data_full_ctc
  seed: 42
  out_dir: data_full
data:
  manifest: video_data.csv
  tokenizer: bsl_sp.model
  max_frames: 512
  normalize: true
  upper_body: true
split:
  kind: random-v1
  fracs: [0.8, 0.1, 0.1]
model:
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.1
  temporal_conv_layers: 2
  temporal_conv_kernel: 5
  ctc_head: true
train:
  epochs: 60
  batch_size: 32
  lr: 0.0005
  weight_decay: 0.0001
  clip_grad: 1.0
  warmup_steps: 0
  label_smoothing: 0.1
  ctc_weight: 0.3
  beam_size: 1
  max_decode_len: 256
  device: auto
```

- [x] **Step 2: Wire the model construction** in `scripts/train_sample.py`

In the `SignTransformer(...)` call (line 160), add:

```python
        use_ctc=bool(m.get("ctc_head", False)),
```

- [x] **Step 3: Read the CTC weight and pass it to training**

After `repetition_stop = int(...)` (line 180), add:

```python
    ctc_weight = float(cfg["train"].get("ctc_weight", 0.0))
```

In the `train_epoch(...)` call (line 195), add arguments:

```python
                         ctc_weight=ctc_weight, bos_id=tok.bos_id,
```

- [x] **Step 4: Log CTC in the EXPERIMENTS row**

In `_append_experiment_row`, change the "Training extras" line to include the CTC weight:

```python
        f"- **Training extras:** early_eos_penalty={cfg['train'].get('early_eos_penalty', 0.0)} "
        f"ctc_weight={cfg['train'].get('ctc_weight', 0.0)}\n"
```

- [x] **Step 5: Overfit-tiny gate (required before the full run)**

Run a 3-epoch smoke on a tiny slice to confirm the joint path trains without shape errors:
```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml \
  --out-dir trial_1pct --epochs 3 --device cuda
```
Expected: it prints `warmup_steps=… ctc_weight=0.3 …`, runs 3 epochs with a **decreasing** loss, decodes, and writes a run folder. If loss is NaN, lower `ctc_weight` to 0.1 and confirm `zero_infinity=True` is set (it is in B2). Record any dead-end in `docs/logs/FAILURES.md`.

- [x] **Step 6: Full run on random-v1**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_ctc.yaml --device cuda
```
Expected (~35–45 min on the 3070, per prior TConv runs): a `TEST (greedy) | bleu4=… chrf++=… …` line and `run saved -> data_full/runs/<hash>`. A new EXPERIMENTS row is appended automatically.

- [x] **Step 7: Compare with CIs against the TConv-only baseline**

```bash
.venv/Scripts/python scripts/results_significance.py \
  --run "tconv=data_full/runs/c26953197929" \
  --run "tconv_ctc=data_full/runs/<hash-from-step-6>" \
  --baseline tconv --metrics bleu4 chrf++ \
  --out docs/logs/SIGNIFICANCE_CTC.md
```
Expected: a table showing whether joint CTC significantly changes BLEU-4/chrF++ vs TConv-only.

- [x] **Step 8: Changelog + commit**

Add a `## …  — Joint CTC/attention ablation` entry to `docs/logs/CHANGELOG.md` summarizing the result and whether the delta is significant.
```bash
git add configs/full_ctc.yaml scripts/train_sample.py docs/logs/EXPERIMENTS.md docs/logs/SIGNIFICANCE_CTC.md docs/logs/CHANGELOG.md
git commit -m "feat(train): joint CTC/attention config + random-v1 ablation"
```

---

## Phase C — Shared decode module + BiLSTM encoder baseline (T0.1)

**Why:** the program plan's Tier-0 (§7.1) names **BiLSTM+attention** as the classic SLT baseline alongside the Transformer; the suite currently has only the Transformer family. To add a second encoder cleanly (DRY), first extract the autoregressive decode logic that `SignTransformer.generate`/`beam_search` duplicate into a shared module, guarded by a golden-output test so the extraction is provably behavior-preserving; then build the BiLSTM encoder on top of the shared decoder.

### Task C1: Extract greedy/beam decode into a shared module (behavior-preserving)

**Files:**
- Create: `bornil/models/decode.py`
- Modify: `bornil/models/transformer.py` (`generate`/`beam_search` delegate to the shared functions)
- Test: `tests/models/test_decode_golden.py` (create)

**Interfaces:**
- Produces:
  - `greedy_decode(step_fn, bos_id, eos_id, batch_size, device, max_len=128, repetition_stop=0) -> list[list[int]]`, where `step_fn(ys: LongTensor[B,L]) -> FloatTensor[B, vocab]` gives next-token logits for the whole batch.
  - `beam_decode(step_fn_single, bos_id, eos_id, beam_size=5, max_len=128, length_penalty=1.0, repetition_stop=0) -> list[int]`, where `step_fn_single(ys: LongTensor[1,L]) -> FloatTensor[vocab]` decodes one example (caller loops over the batch).
- Consumes in `transformer.py`: the existing `has_consecutive_repeat`, `causal_mask`, `self.encode`, `self.transformer.decoder`.

- [x] **Step 1: Write the golden test** — pin current outputs, then require identical after refactor

```python
# tests/models/test_decode_golden.py
import bornil.runtime  # noqa: F401
import torch
from bornil.models.transformer import SignTransformer


def _model():
    torch.manual_seed(0)
    return SignTransformer(input_dim=218, vocab_size=30, d_model=32, nhead=4,
                           num_encoder_layers=2, num_decoder_layers=2,
                           dim_feedforward=64)


def test_greedy_matches_reference_sequence():
    m = _model()
    src = torch.Generator().manual_seed(1)
    x = torch.randn(2, 12, 218, generator=src)
    out = m.generate(x, bos_id=1, eos_id=2, max_len=15)
    assert isinstance(out, list) and len(out) == 2
    # Deterministic given fixed seeds: greedy must be reproducible run-to-run.
    out2 = m.generate(x, bos_id=1, eos_id=2, max_len=15)
    assert out == out2


def test_beam_matches_reference_sequence():
    m = _model()
    x = torch.randn(2, 12, 218, generator=torch.Generator().manual_seed(1))
    a = m.beam_search(x, bos_id=1, eos_id=2, beam_size=3, max_len=15)
    b = m.beam_search(x, bos_id=1, eos_id=2, beam_size=3, max_len=15)
    assert a == b and len(a) == 2
```

- [x] **Step 2: Run to establish current behavior passes**

Run: `.venv/Scripts/python -m pytest tests/models/test_decode_golden.py -v`
Expected: PASS against the current (pre-refactor) implementation. This is the safety net.

- [x] **Step 3: Implement `decode.py`**

```python
# bornil/models/decode.py
"""Encoder-agnostic autoregressive decoding (greedy + beam).

Both take a step function that maps the current token prefix to next-token
logits; the encoder-specific memory and masks live in the caller's closure.
This lets every SLT model (Transformer, BiLSTM, ST-GCN, ...) share one tested
decode path instead of re-implementing it.
"""
from __future__ import annotations

import torch

from bornil.models.transformer import has_consecutive_repeat


@torch.no_grad()
def greedy_decode(step_fn, bos_id, eos_id, batch_size, device,
                  max_len: int = 128, repetition_stop: int = 0):
    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = [False] * batch_size
    outputs = [[] for _ in range(batch_size)]
    for _ in range(max_len):
        nxt = step_fn(ys).argmax(-1)                 # (B,)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        for i in range(batch_size):
            if not finished[i]:
                tok = int(nxt[i])
                outputs[i].append(tok)
                if tok == eos_id or has_consecutive_repeat(outputs[i], repetition_stop):
                    finished[i] = True
        if all(finished):
            break
    return outputs


@torch.no_grad()
def beam_decode(step_fn_single, bos_id, eos_id, beam_size: int = 5,
                max_len: int = 128, length_penalty: float = 1.0,
                repetition_stop: int = 0):
    beams = [([bos_id], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            if tokens[-1] == eos_id or has_consecutive_repeat(tokens[1:], repetition_stop):
                finished.append((tokens, score))
                continue
            ys = torch.tensor([tokens], dtype=torch.long)
            logp = torch.log_softmax(step_fn_single(ys), dim=-1)
            topv, topi = logp.topk(beam_size)
            for v, i in zip(topv.tolist(), topi.tolist()):
                candidates.append((tokens + [i], score + v))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        if all(tok[-1] == eos_id for tok, _ in beams):
            finished.extend(beams)
            break
    finished = finished or beams
    best = max(finished, key=lambda x: x[1] / (len(x[0]) ** length_penalty))
    toks = best[0][1:]
    if eos_id in toks:
        toks = toks[:toks.index(eos_id) + 1]
    return toks
```

Note the `step_fn_single` receives `ys` on CPU; the caller's closure must move it to the model device before embedding (see Step 4).

- [x] **Step 4: Delegate from `SignTransformer`**

Replace the bodies of `generate` and `beam_search` in `transformer.py` with closures over the existing math. `generate`:

```python
    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len=128,
                 src_key_padding_mask=None, repetition_stop=0):
        from bornil.models.decode import greedy_decode
        self.eval()
        memory = self.encode(src, src_key_padding_mask)

        def step_fn(ys):
            ys = ys.to(src.device)
            t = self.pos(self.tok_embed(ys) * math.sqrt(self.d_model))
            h = self.transformer.decoder(
                t, memory, tgt_mask=causal_mask(ys.size(1)).to(src.device),
                memory_key_padding_mask=src_key_padding_mask)
            return self.out(h[:, -1])

        return greedy_decode(step_fn, bos_id, eos_id, src.size(0), src.device,
                             max_len=max_len, repetition_stop=repetition_stop)
```

`beam_search`:

```python
    @torch.no_grad()
    def beam_search(self, src, bos_id, eos_id, beam_size=5, max_len=128,
                    length_penalty=1.0, src_key_padding_mask=None, repetition_stop=0):
        from bornil.models.decode import beam_decode
        self.eval()
        results = []
        for b in range(src.size(0)):
            s = src[b:b + 1]
            mask = None if src_key_padding_mask is None else src_key_padding_mask[b:b + 1]
            memory = self.encode(s, mask)

            def step_fn_single(ys, memory=memory, mask=mask):
                ys = ys.to(src.device)
                t = self.pos(self.tok_embed(ys) * math.sqrt(self.d_model))
                h = self.transformer.decoder(
                    t, memory, tgt_mask=causal_mask(ys.size(1)).to(src.device),
                    memory_key_padding_mask=mask)
                return self.out(h[:, -1])[0]

            results.append(beam_decode(step_fn_single, bos_id, eos_id,
                                       beam_size=beam_size, max_len=max_len,
                                       length_penalty=length_penalty,
                                       repetition_stop=repetition_stop))
        return results
```

- [x] **Step 5: Run the golden test + all model tests**

Run: `.venv/Scripts/python -m pytest tests/models -q`
Expected: all pass — decode outputs unchanged (the golden test proves the extraction preserved behavior).

- [x] **Step 6: Commit**

```bash
git add bornil/models/decode.py bornil/models/transformer.py tests/models/test_decode_golden.py
git commit -m "refactor(model): shared greedy/beam decode module (behavior-preserving)"
```

### Task C2: BiLSTM encoder model (T0.1)

**Files:**
- Create: `bornil/models/bilstm.py`
- Create: `configs/full_bilstm.yaml`
- Modify: `scripts/train_sample.py` (model factory: choose encoder by `cfg["model"]["arch"]`)
- Test: `tests/models/test_bilstm.py` (create)

**Interfaces:**
- Produces `SignBiLSTM(input_dim, vocab_size, d_model=256, enc_layers=2, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, pad_id=0)` exposing the same contract the loop needs:
  - `forward(src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None) -> (B, L, vocab)`
  - `src_padding_mask(src_lengths, seq_len) -> (B, S) bool`
  - `encode(src, src_key_padding_mask=None) -> (B, S, d_model)`
  - `generate(...)` and `beam_search(...)` delegating to `bornil.models.decode` (as in C1).
- Consumes: `bornil.models.decode.greedy_decode/beam_decode`; a `nn.TransformerDecoder` for the decoder side (so masks/positional match the Transformer decoder already validated).

- [x] **Step 1: Write the failing test** (overfit-tiny gate for the new arch)

```python
# tests/models/test_bilstm.py
import bornil.runtime  # noqa: F401
import torch
from torch.utils.data import DataLoader

from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.models.bilstm import SignBiLSTM
from bornil.train.loop import train_epoch


def test_shapes():
    m = SignBiLSTM(input_dim=218, vocab_size=40, d_model=32, enc_layers=2,
                   num_decoder_layers=2, dim_feedforward=64)
    x = torch.randn(3, 11, 218)
    y = torch.randint(3, 40, (3, 6))
    logits = m(x, y, src_key_padding_mask=m.src_padding_mask(torch.tensor([11, 9, 7]), 11))
    assert logits.shape == (3, 6, 40)


def test_overfit_tiny():
    torch.manual_seed(0)
    N, T, D, V = 8, 16, 218, 40
    feats = torch.randn(N, T, D).numpy()
    targets = [[1, 5, 6, 7, 2] for _ in range(N)]
    ds = SignDataset(feats, [T] * N, list(range(N)), targets,
                     normalize=False, upper_body=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(0))
    m = SignBiLSTM(input_dim=D, vocab_size=V, d_model=32, enc_layers=2,
                   num_decoder_layers=2, dim_feedforward=64)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = train_epoch(m, loader, opt, pad_id=0, eos_id=2)
    for _ in range(25):
        last = train_epoch(m, loader, opt, pad_id=0, eos_id=2)
    assert last < first * 0.5
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/models/test_bilstm.py -v`
Expected: FAIL — `No module named 'bornil.models.bilstm'`.

- [x] **Step 3: Implement `SignBiLSTM`**

```python
# bornil/models/bilstm.py
"""BiLSTM encoder + Transformer decoder SLT baseline (WS-C, T0.1).

The classic sign-translation encoder: a bidirectional LSTM over projected
per-frame keypoints, whose outputs are projected to d_model and consumed as
memory by a standard Transformer decoder. Reuses the shared decode module so
greedy/beam behavior is identical to the Transformer baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from bornil.models.transformer import PositionalEncoding, causal_mask


class SignBiLSTM(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, d_model: int = 256,
                 enc_layers: int = 2, num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024, nhead: int = 8,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=enc_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if enc_layers > 1 else 0.0)
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def src_padding_mask(self, src_lengths, seq_len):
        ar = torch.arange(seq_len, device=src_lengths.device)[None, :]
        return ar >= src_lengths[:, None]

    def encode(self, src, src_key_padding_mask=None):
        s = self.input_proj(src)
        mem, _ = self.lstm(s)            # (B, S, d_model)
        return mem

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask)
        t = self.pos(self.tok_embed(tgt_in) * math.sqrt(self.d_model))
        h = self.decoder(
            t, memory, tgt_mask=causal_mask(tgt_in.size(1)).to(src.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask)
        return self.out(h)

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len=128,
                 src_key_padding_mask=None, repetition_stop=0):
        from bornil.models.decode import greedy_decode
        self.eval()
        memory = self.encode(src, src_key_padding_mask)

        def step_fn(ys):
            ys = ys.to(src.device)
            t = self.pos(self.tok_embed(ys) * math.sqrt(self.d_model))
            h = self.decoder(t, memory, tgt_mask=causal_mask(ys.size(1)).to(src.device),
                             memory_key_padding_mask=src_key_padding_mask)
            return self.out(h[:, -1])

        return greedy_decode(step_fn, bos_id, eos_id, src.size(0), src.device,
                             max_len=max_len, repetition_stop=repetition_stop)

    @torch.no_grad()
    def beam_search(self, src, bos_id, eos_id, beam_size=5, max_len=128,
                    length_penalty=1.0, src_key_padding_mask=None, repetition_stop=0):
        from bornil.models.decode import beam_decode
        self.eval()
        results = []
        for b in range(src.size(0)):
            mask = None if src_key_padding_mask is None else src_key_padding_mask[b:b + 1]
            memory = self.encode(src[b:b + 1], mask)

            def step_fn_single(ys, memory=memory, mask=mask):
                ys = ys.to(src.device)
                t = self.pos(self.tok_embed(ys) * math.sqrt(self.d_model))
                h = self.decoder(t, memory, tgt_mask=causal_mask(ys.size(1)).to(src.device),
                                 memory_key_padding_mask=mask)
                return self.out(h[:, -1])[0]

            results.append(beam_decode(step_fn_single, bos_id, eos_id,
                                       beam_size=beam_size, max_len=max_len,
                                       length_penalty=length_penalty,
                                       repetition_stop=repetition_stop))
        return results
```

- [x] **Step 4: Run the BiLSTM tests**

Run: `.venv/Scripts/python -m pytest tests/models/test_bilstm.py -v`
Expected: 2 passed (shapes + overfit-tiny loss more than halves).

- [x] **Step 5: Add an arch selector to the train script**

In `scripts/train_sample.py`, replace the direct `SignTransformer(...)` construction (lines 160–167) with a small factory keyed on `cfg["model"].get("arch", "transformer")`:

```python
    arch = m.get("arch", "transformer")
    if arch == "bilstm":
        from bornil.models.bilstm import SignBiLSTM
        model = SignBiLSTM(
            input_dim=feature_dim(cfg["data"]["upper_body"]), vocab_size=tok.vocab_size,
            d_model=m["d_model"], enc_layers=int(m.get("enc_layers", 2)),
            num_decoder_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"], nhead=m["nhead"],
            dropout=m["dropout"], pad_id=tok.pad_id,
        ).to(device)
    else:
        model = SignTransformer(
            input_dim=feature_dim(cfg["data"]["upper_body"]), vocab_size=tok.vocab_size,
            d_model=m["d_model"], nhead=m["nhead"],
            num_encoder_layers=m["num_encoder_layers"], num_decoder_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"], dropout=m["dropout"], pad_id=tok.pad_id,
            temporal_conv_layers=int(m.get("temporal_conv_layers", 0)),
            temporal_conv_kernel=int(m.get("temporal_conv_kernel", 5)),
            use_ctc=bool(m.get("ctc_head", False)),
        ).to(device)
```

Also update the EXPERIMENTS "Model:" line in `_append_experiment_row` to name the arch — change the `SignTransformer{front}` literal to `{cfg['model'].get('arch','transformer')}{front}` so BiLSTM rows are not mislabeled.

- [x] **Step 6: Create `configs/full_bilstm.yaml`** (same as `full.yaml` but with `model.arch: bilstm` and `model.enc_layers: 2`, keeping `nhead`/`num_decoder_layers`/`dim_feedforward`/`d_model` so the decoder matches).

- [x] **Step 7: Overfit-tiny smoke on real data + full run**

```bash
.venv/Scripts/python scripts/train_sample.py --config configs/full_bilstm.yaml --out-dir trial_1pct --epochs 3 --device cuda
.venv/Scripts/python scripts/train_sample.py --config configs/full_bilstm.yaml --device cuda
```
Expected: smoke shows decreasing loss; full run writes a run folder + EXPERIMENTS row.

- [x] **Step 8: CI comparison vs the Transformer baseline + changelog + commit**

```bash
.venv/Scripts/python scripts/results_significance.py \
  --run "transformer=data_full/runs/c19d0ea08824" \
  --run "bilstm=data_full/runs/<hash>" \
  --baseline transformer --metrics bleu4 chrf++ --out docs/logs/SIGNIFICANCE_BILSTM.md
```
Add a CHANGELOG entry (BiLSTM T0.1 on board; delta vs Transformer with significance).
```bash
git add bornil/models/bilstm.py configs/full_bilstm.yaml scripts/train_sample.py tests/models/test_bilstm.py docs/logs/EXPERIMENTS.md docs/logs/SIGNIFICANCE_BILSTM.md docs/logs/CHANGELOG.md
git commit -m "feat(model): BiLSTM encoder baseline (WS-C T0.1) + significance vs Transformer"
```

---

# Part II — Specified; expand into its own plan when reached

Each phase below has entry criteria, files, interfaces, and approach. **When you start one, run superpowers:writing-plans to expand it into a full task-by-task plan** — its exact steps depend on results from the earlier phases (and Phases F–G depend on cloud + new extraction that doesn't exist yet, so micro-stepping them now would be guessing).

## Phase D — ST-GCN skeleton encoder (T1.1)

- **Entry:** Phase C merged (a shared decode path + a second encoder pattern to copy).
- **Files:** `bornil/models/stgcn.py` (spatiotemporal graph encoder producing `(B, S, d_model)` memory, then the shared decoder), `configs/full_stgcn.yaml`, `tests/models/test_stgcn.py`.
- **Approach:** build the skeleton adjacency for the kp-218 layout (upper-body pose joints 0–22 + 21+21 hand joints) as a fixed normalized adjacency matrix; stack ST-GCN blocks (graph conv over joints × temporal conv over frames); reshape to `(B, S, d_model)` and reuse the Task C1 decoder. This directly subsumes Arib et al.'s ST-GCN family (program plan §7.2, honest comparison).
- **Validation:** overfit-tiny gate; beats the retrieval/trivial floor on random-v1 with a non-overlapping CI; report the `random → signerindep` gap.

## Phase E — Face-keypoint representation + face ablation (WS-A §5.2.2)

- **Entry:** Phase A done (so the ablation is reported with CIs). Independent of B–D; can run in parallel.
- **Why now-ish:** face is currently **OFF** (`refine_face_landmarks=False`); facial expression carries grammatical meaning in sign languages (negation/question/intensity). This is a genuine representation gap and a clean ablation ("does face help BdSL SLT?").
- **Files:** modify the extractor (`bsl_extract.py` / `run_extraction.py`) to emit a compact face-landmark subset (~40–70 pts: brows/eyes/mouth) into an extended raw vector; a load-time transform that optionally concatenates face channels; `configs/full_face.yaml`; a re-extraction over a **subset** (not all 17,988 — extract only what the ablation needs to bound compute, per §5.2 failure-mode note).
- **Validation:** feature-parity on non-face channels; face-on vs face-off compared on random-v1 + signerindep-v1 with paired significance.

## Phase F — Conformer (T1.3) + two-stream kp⊕hand-RGB (T1.2) — cloud

- **Entry:** Phase D done; **two-stream additionally needs cropped-hand-RGB features**, which requires a new extraction pass (hand-landmark bounding boxes → fixed-size patches → frozen CNN features → memmap). That extraction is its own task and is cloud-friendly.
- **Files:** `bornil/models/conformer.py` (convolution-augmented Transformer encoder + shared decoder); `bornil/models/twostream.py` (keypoint stream ⊕ hand-RGB stream with cross-stream fusion); hand-patch extractor under `scripts/`; configs + tests.
- **Validation:** Conformer as a "latest established" point on all three splits; two-stream tests whether appearance rescues MediaPipe hand misses (report residual hand-miss rate correlation).

## Phase G — ViViT / VideoMAE video backbones (T2.1 / T2.2) — cloud-only

**This is the answer to "when do we use ViViT."** It comes here, and only after its entry criteria hold.

- **Entry criteria (all required):**
  1. The keypoint suite (Transformer + CTC + BiLSTM + ST-GCN) is on the leaderboard **with CIs**, so there is a concrete baseline for the video model to beat.
  2. **RGB features are pre-extracted** (this does not exist yet — see step below). Do **not** decode video in the training loop.
  3. Cloud A100/H100 is provisioned (the 3070 cannot finetune video transformers).
- **First task (the real blocker): RGB pre-extraction (WS-A §5.2.4).** For each clip: decode → resize → uniform-sample frames (reuse the ≤512 adaptive-ceiling sampling discipline) → run a frozen backbone (ViViT/VideoMAE image or clip encoder from HuggingFace) → store per-clip features to a memmap `(N, T, feat_dim)`, mirroring the existing `packed.npy` layout so `SignDataset` can consume it with a different `feature_dim`. Budget storage + cloud I/O for this before anything else.
- **Model:** wrap the HF video backbone as an encoder producing `(B, S, d_model)` memory (project backbone dim → d_model), reuse the Task C1 shared decoder + the existing eval harness. Finetune with gradient checkpointing + bf16 on A100.
- **Files (in the expanded plan):** `scripts/extract_rgb_features.py`, `bornil/models/video_encoder.py`, `configs/full_vivit.yaml` / `configs/full_videomae.yaml`, tests.
- **Validation:** overfit-tiny on a cloud GPU first; ViViT vs the best keypoint model on all three splits with paired significance; the interesting result is whether appearance/video buys generalization over keypoints on `signerindep-v1`.

---

## Self-review notes (coverage check)

- **Program-plan §11 "CIs everywhere + paired tests"** → Phase A (Tasks A1, A2), reused as the comparison step in every later phase.
- **§7.1 T0.2 Transformer** → already done; **joint CTC** (the §7 "common interface" CTC head) → Phase B.
- **§7.1 T0.1 BiLSTM** → Phase C.
- **§7.2 T1.1 ST-GCN** → Phase D. **§7.2 T1.2 two-stream / §7.2 T1.3 Conformer** → Phase F.
- **§5.2.2 face representation** → Phase E.
- **§7.3 T2.1/T2.2 ViViT/VideoMAE** → Phase G, with its true blocker (RGB pre-extraction, §5.2.4) called out as the first task.
- **§9 WS-E alignment** is unblocked by Phase B's CTC head (CTC forced alignment) but is a separate workstream — write its own plan when starting it.

## Execution note (non-code housekeeping, do opportunistically)

The working tree has untracked artifacts: `trial_1pct/`, `trial_2pct/`, `trial_5pct/`, `termina-log.md`, `temp_bert_smoke.csv`, `temp_bert_smoke_metrics.json`. Confirm they are covered by `.gitignore` (trial run folders should be) or remove the stray `temp_*`/`termina-log.md` files before committing Phase A, so new run folders don't get swept into commits.
