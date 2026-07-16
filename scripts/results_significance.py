"""Confidence intervals + paired significance over saved run folders.

Reads each run's predictions.csv (reference,hypothesis), reports a point
estimate with a bootstrap 95% CI per metric, and a paired bootstrap p-value
against a chosen baseline run. This is the report-facing companion to
docs/logs/RESULTS_OVERVIEW.md: it turns point-estimate tables into CI +
significance tables.

    python scripts/results_significance.py \
        --run "vanilla=data_full/runs/c19d0ea08824" \
        --run "tconv=data_full/runs/c26953197929" \
        --baseline vanilla --metrics bleu4 chrf++ \
        --out docs/logs/SIGNIFICANCE.md
"""
from __future__ import annotations

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
    lines = [
        "# Significance & Confidence Intervals",
        "",
        f"Bootstrap n={n}, seed={seed}. Paired p-values vs `{baseline_label}`.",
        "",
    ]
    header = "| Run | Metric | Point | 95% CI | p vs " + str(baseline_label) + " |"
    lines += [header, "|---|---|---:|---|---:|"]
    for label, _ in runs:
        hyps, refs = data[label]
        for metric in metrics:
            point, lo, hi = metric_ci(hyps, refs, metric=metric, n=n, seed=seed)
            if base is not None and label != baseline_label:
                p = paired_bootstrap_pvalue(base[0], hyps, refs,
                                            metric=metric, n=n, seed=seed)
                pcol = f"{p:.3f}"
            else:
                pcol = "-"
            lines.append(
                f"| {label} | {metric} | {point:.3f} | "
                f"[{lo:.3f}, {hi:.3f}] | {pcol} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help="label=path/to/run_dir (repeatable)")
    ap.add_argument("--baseline", default=None,
                    help="label to compare others against")
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
