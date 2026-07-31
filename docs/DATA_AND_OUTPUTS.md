# Data & Outputs — What's Not in Git, and Where to Look Instead

This project generates far more data than belongs in a git repository. This
doc says exactly what's excluded, how big it actually is, what it looks like,
and — most importantly — where to find a documented summary of it that *is*
in the repo, so you don't need the heavy files to see real results.

## What's excluded, and why

| Path | What it is | Size | Why excluded |
|---|---|---|---|
| `chunks/` | Raw `.webm` video, one per clip | ~180 GB | Far too large for git; distributed separately from the code |
| `data_full/` | Packed keypoint dataset + every training run's output | ~15 GB | Large, and fully regenerable from `chunks/` via the extraction pipeline (see `README.md`'s Pipeline section) |
| `features_mediapipe*/`, `outputs/`, `trial_*/` heavy files | Intermediate per-clip features, trial-run artifacts | varies | Same reasoning — regenerable, not source-of-truth |

None of this is a judgment call made once and forgotten — it's enforced by
`.gitignore` and has been true since this project's disk-space incident
(`docs/logs/FAILURES.md`, FAIL-001): committing large binaries here has
already caused real problems once.

## What's actually inside `data_full/` (~15 GB, measured 2026-07-31)

```
data_full/
├── packed.npy            # 8.9 GB — the packed keypoint dataset:
│                          #   shape (17988, 512, 258) float32, one row per
│                          #   clip, raw (unnormalized, all-landmark) features.
│                          #   This is what every SignDataset load reads from.
├── lengths.npy            # true per-clip frame count (before the 512 pad)
├── sample_manifest.csv    # clip ↔ label ↔ pack_idx join table
└── runs/
    └── <config-hash>/     # 26 of these as of 2026-07-31, ~50-100 MB each
        ├── config.json    # the exact config that produced this run
        ├── metrics.json   # test-set BLEU/chrF/chrF++/ROUGE-L
        ├── predictions.csv         # every (reference, hypothesis) pair
        ├── best.pt        # the checkpoint (best validation BLEU-4)
        └── ...             # some runs also have alignments.csv,
                             #   diagnostics.json, input_ablation/, etc.,
                             #   depending on which analysis scripts ran
                             #   against that checkpoint
```

Every one of those 26 `<config-hash>` directories corresponds to exactly one
row in [`docs/logs/EXPERIMENTS.md`](logs/EXPERIMENTS.md) — that file is the
full ledger (what was trained, on what split, with what config, scoring
what) without needing the checkpoint itself.

## Where to look instead of the heavy files

You don't need any of the above to understand what this project found. In
order of "how much detail do you want":

1. **[`docs/logs/RESULTS_OVERVIEW.md`](logs/RESULTS_OVERVIEW.md)** — the
   report-facing summary. Headline tables (official baselines, ablations,
   trivial floors, the input-ablation grounding control, diversity/length
   diagnostics), each with the significance test that backs it, each
   pointing at the run hash that produced it.
2. **[`docs/logs/QUALITATIVE_EXAMPLES.md`](logs/QUALITATIVE_EXAMPLES.md)** —
   real sample outputs. Actual Bangla (reference, hypothesis) pairs for
   multiple models and splits: exact matches, "typical" (median-chrF++)
   examples, and failure cases (e.g. repeated-token runaway decoding) — this
   *is* the "chunk of it" you can read without downloading anything.
3. **[`docs/logs/EXPERIMENTS.md`](logs/EXPERIMENTS.md)** — the complete,
   append-only ledger of every run ever trained: architecture, split,
   config hash, compute time, and test metrics, with a `Checkpoint:` line
   pointing at where the (untracked) `best.pt` would live if you regenerated
   it.
4. **[`docs/logs/CHANGELOG.md`](logs/CHANGELOG.md)** — the narrative version:
   what changed, why, and what it meant, newest-first, across the whole
   project's history.
5. **[`docs/logs/FAILURES.md`](logs/FAILURES.md)** — every dead end and
   negative result, with root-cause hypotheses and what was tried next.
   Several of this project's most important findings (that grounding is
   static-pose rather than motion; that a masked-pretraining objective and a
   pure-PyTorch Mamba encoder both didn't pan out) live here, not in a
   success-only results table.
6. **[`docs/logs/SIGNIFICANCE_*.md`](logs/)** — the raw paired-bootstrap
   output backing every "significantly better/worse" claim in the overview.

## If you want to regenerate the heavy files yourself

1. Get the raw video (`chunks/`) — distributed separately from this repo.
2. Follow `README.md`'s Pipeline section: extract → pack → train, using
   `scripts/run_trial.py` for a fast small-scale check, or
   `scripts/train_sample.py --config configs/<name>.yaml` for a full
   official run matching one of the `EXPERIMENTS.md` rows exactly (every
   config is committed, so any run is exactly reproducible given the same
   video and split files — the split files themselves *are* committed,
   under `splits/`).

No step of this pipeline requires anything that isn't either in this repo or
in the separately-distributed video set.
