"""Create lightweight diagnostics for a predictions.csv file.

The input must have:
    reference,hypothesis

Outputs a Markdown report and JSON summary by default. This is intentionally
metric-light: it explains model behavior around length, repetition, exact
matches, and representative examples after the main scoring script has already
computed corpus metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sacrebleu.metrics import CHRF

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bornil.benchmark import metrics as M


def _read_predictions(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    missing = {"reference", "hypothesis"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path} missing required column(s): {sorted(missing)}")
    return rows


def _tok_len(text: str) -> int:
    return len(str(text).split())


def _length_bins(rows):
    bins = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 999)]
    out = []
    for lo, hi in bins:
        bucket = [
            r for r in rows
            if lo <= _tok_len(r["reference"]) <= hi
        ]
        if not bucket:
            continue
        refs = [r["reference"] for r in bucket]
        hyps = [r["hypothesis"] for r in bucket]
        m = M.corpus_metrics(hyps, refs)
        out.append({
            "bin": f"{lo}-{hi if hi < 999 else 'plus'}",
            "rows": len(bucket),
            "avg_ref_tokens": sum(_tok_len(r) for r in refs) / len(refs),
            "avg_hyp_tokens": sum(_tok_len(h) for h in hyps) / len(hyps),
            "bleu4": m["bleu4"],
            "chrf++": m["chrf++"],
            "rougeL": m["rougeL"],
        })
    return out


def _sentence_scores(rows):
    chrfpp = CHRF(word_order=2)
    scored = []
    for i, row in enumerate(rows, start=1):
        ref = row["reference"]
        hyp = row["hypothesis"]
        scored.append({
            "row": i,
            "reference": ref,
            "hypothesis": hyp,
            "ref_tokens": _tok_len(ref),
            "hyp_tokens": _tok_len(hyp),
            "exact": ref == hyp,
            "sentence_chrf++": chrfpp.sentence_score(hyp, [ref]).score,
            "rougeL": M.rouge_l(hyp, ref),
        })
    return scored


def _summary(rows):
    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    counts = Counter(hyps)
    empty = sum(not h.strip() for h in hyps)
    exact = sum(r == h for r, h in zip(refs, hyps))
    repeated_rows = sum(n for n in counts.values() if n > 1)
    return {
        "rows": len(rows),
        "unique_hypotheses": len(counts),
        "empty_hypotheses": empty,
        "exact_matches": exact,
        "exact_match_rate": exact / len(rows),
        "avg_ref_tokens": sum(_tok_len(r) for r in refs) / len(refs),
        "avg_hyp_tokens": sum(_tok_len(h) for h in hyps) / len(hyps),
        "repeated_hypothesis_rows": repeated_rows,
        "repeated_hypothesis_rate": repeated_rows / len(rows),
        "top_repeated_hypotheses": [
            {"count": n, "hypothesis": h}
            for h, n in counts.most_common(15)
        ],
    }


def _write_markdown(path: Path, pred_path: Path, data):
    lines = [
        f"# Prediction Diagnostics: {pred_path}",
        "",
        "## Summary",
        "",
        "| Rows | Unique hyps | Empty | Exact | Exact % | Avg ref tokens | Avg hyp tokens | Repeated rows % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    s = data["summary"]
    lines.append(
        f"| {s['rows']} | {s['unique_hypotheses']} | {s['empty_hypotheses']} | "
        f"{s['exact_matches']} | {s['exact_match_rate'] * 100:.2f} | "
        f"{s['avg_ref_tokens']:.2f} | {s['avg_hyp_tokens']:.2f} | "
        f"{s['repeated_hypothesis_rate'] * 100:.2f} |"
    )

    lines.extend([
        "",
        "## Length Bins",
        "",
        "| Ref token bin | Rows | Avg ref tokens | Avg hyp tokens | BLEU-4 | chrF++ | ROUGE-L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in data["length_bins"]:
        lines.append(
            f"| {row['bin']} | {row['rows']} | {row['avg_ref_tokens']:.2f} | "
            f"{row['avg_hyp_tokens']:.2f} | {row['bleu4']:.3f} | "
            f"{row['chrf++']:.3f} | {row['rougeL']:.3f} |"
        )

    lines.extend([
        "",
        "## Top Repeated Hypotheses",
        "",
        "| Count | Hypothesis |",
        "|---:|---|",
    ])
    for row in s["top_repeated_hypotheses"][:10]:
        hyp = row["hypothesis"].replace("|", "\\|")
        lines.append(f"| {row['count']} | {hyp} |")

    for title, key in [
        ("Best Examples", "best_examples"),
        ("Worst Examples", "worst_examples"),
    ]:
        lines.extend([
            "",
            f"## {title}",
            "",
            "| Row | chrF++ | ROUGE-L | Reference | Hypothesis |",
            "|---:|---:|---:|---|---|",
        ])
        for row in data[key]:
            ref = row["reference"].replace("|", "\\|")
            hyp = row["hypothesis"].replace("|", "\\|")
            lines.append(
                f"| {row['row']} | {row['sentence_chrf++']:.2f} | "
                f"{row['rougeL']:.3f} | {ref} | {hyp} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.csv path")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    pred_path = Path(args.pred)
    rows = _read_predictions(pred_path)
    scored = _sentence_scores(rows)
    scored_by_chrf = sorted(scored, key=lambda r: r["sentence_chrf++"])
    data = {
        "predictions": str(pred_path),
        "summary": _summary(rows),
        "length_bins": _length_bins(rows),
        "best_examples": list(reversed(scored_by_chrf[-args.examples:])),
        "worst_examples": scored_by_chrf[:args.examples],
    }

    out_md = Path(args.out_md) if args.out_md else pred_path.with_name("diagnostics.md")
    out_json = Path(args.out_json) if args.out_json else pred_path.with_name("diagnostics.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(out_md, pred_path, data)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved -> {out_md}")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
