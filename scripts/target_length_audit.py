"""Audit target-text length distribution and long examples.

This helps decide whether failures on long references reflect model weakness or
noisy/out-of-scope target text. The default input is the packed full-dataset
manifest written by the extraction pipeline.

Example:
    python scripts/target_length_audit.py --manifest data_full/sample_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def _token_len(text) -> int:
    return len(str(text).split())


def _bucket_counts(lengths, buckets):
    out = []
    for lo, hi in buckets:
        n = int(((lengths >= lo) & (lengths <= hi)).sum())
        out.append({"bucket": f"{lo}-{hi if hi < 999 else 'plus'}", "rows": n})
    return out


def _write_examples_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["recording", "tok_len", "clean_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, manifest: Path, data):
    lines = [
        f"# Target Length Audit: {manifest}",
        "",
        "## Summary",
        "",
        "| Rows | Mean | Std | Min | 25% | 50% | 75% | 90% | 95% | 99% | Max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    s = data["summary"]
    lines.append(
        f"| {s['rows']} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']} | "
        f"{s['p25']:.1f} | {s['p50']:.1f} | {s['p75']:.1f} | "
        f"{s['p90']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | {s['max']} |"
    )

    lines.extend([
        "",
        "## Length Buckets",
        "",
        "| Token-length bucket | Rows | Percent |",
        "|---|---:|---:|",
    ])
    total = s["rows"]
    for row in data["buckets"]:
        pct = row["rows"] / total * 100 if total else 0
        lines.append(f"| {row['bucket']} | {row['rows']} | {pct:.2f}% |")

    lines.extend([
        "",
        "## Longest Examples",
        "",
        "| Token length | Recording | Text |",
        "|---:|---|---|",
    ])
    for row in data["long_examples"]:
        text = str(row["clean_text"]).replace("|", "\\|")
        lines.append(f"| {row['tok_len']} | `{row['recording']}` | {text} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_full/sample_manifest.csv")
    ap.add_argument("--text-col", default="clean_text")
    ap.add_argument("--recording-col", default="recording")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--out-md", default="docs/logs/TARGET_LENGTH_AUDIT.md")
    ap.add_argument("--out-json", default="docs/logs/TARGET_LENGTH_AUDIT.json")
    ap.add_argument("--long-csv", default="docs/logs/TARGET_LENGTH_LONG_EXAMPLES.csv")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    df = pd.read_csv(manifest)
    for col in (args.text_col, args.recording_col):
        if col not in df.columns:
            raise SystemExit(f"{manifest} missing required column: {col}")

    df = df.copy()
    df["tok_len"] = df[args.text_col].map(_token_len)
    desc = df["tok_len"].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    buckets = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 20), (21, 40), (41, 999)]

    long_rows = []
    for _, row in df.sort_values("tok_len", ascending=False).head(args.top_n).iterrows():
        long_rows.append({
            "recording": row[args.recording_col],
            "tok_len": int(row["tok_len"]),
            "clean_text": row[args.text_col],
        })

    data = {
        "manifest": str(manifest),
        "summary": {
            "rows": int(desc["count"]),
            "mean": float(desc["mean"]),
            "std": float(desc["std"]),
            "min": int(desc["min"]),
            "p25": float(desc["25%"]),
            "p50": float(desc["50%"]),
            "p75": float(desc["75%"]),
            "p90": float(desc["90%"]),
            "p95": float(desc["95%"]),
            "p99": float(desc["99%"]),
            "max": int(desc["max"]),
        },
        "buckets": _bucket_counts(df["tok_len"], buckets),
        "long_examples": long_rows,
    }

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    long_csv = Path(args.long_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(out_md, manifest, data)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_examples_csv(long_csv, long_rows)

    print(f"rows={data['summary']['rows']} mean={data['summary']['mean']:.2f} "
          f"median={data['summary']['p50']:.1f} p95={data['summary']['p95']:.1f} "
          f"max={data['summary']['max']}")
    for row in data["buckets"]:
        print(f"{row['bucket']}: {row['rows']}")
    print(f"saved -> {out_md}")
    print(f"saved -> {out_json}")
    print(f"saved -> {long_csv}")


if __name__ == "__main__":
    main()
