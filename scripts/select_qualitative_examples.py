"""Select qualitative prediction examples from saved per-example scores.

The output is a Markdown table intended for report writing. Selection is
algorithmic, so the final paper should still include human review of the chosen
examples.

Examples:
    python scripts/select_qualitative_examples.py
    python scripts/select_qualitative_examples.py --out docs/logs/QUALITATIVE_EXAMPLES.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List


DEFAULT_RUNS = (
    "c19d0ea08824",  # official random-v1
    "60320f4c6a60",  # official unseen-sentence-v1
    "46a706570712",  # official signerindep-v1
    "bc38907bece6",  # LTE12 random-v1
    "58269a7b1003",  # LTE12 unseen-sentence-v1
    "2ff734ae1aae",  # LTE12 signerindep-v1
)


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["row_id"] = int(row["row_id"])
        row["exact_match"] = str(row["exact_match"]).lower() == "true"
        row["ref_tokens"] = int(row["ref_tokens"])
        row["hyp_tokens"] = int(row["hyp_tokens"])
        row["sentence_chrfpp"] = float(row["sentence_chrfpp"])
        row["rougeL"] = float(row["rougeL"])
    return rows


def _max_repeat_run(text: str) -> int:
    toks = str(text).split()
    best = cur = 0
    prev = None
    for tok in toks:
        if tok == prev:
            cur += 1
        else:
            prev = tok
            cur = 1
        best = max(best, cur)
    return best


def _clip(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _md(text: str) -> str:
    return _clip(text).replace("|", "\\|")


def _pick_examples(rows: List[Dict]) -> List[Dict]:
    examples = []

    exact = [r for r in rows if r["exact_match"]]
    if exact:
        best = max(exact, key=lambda r: (r["ref_tokens"], r["sentence_chrfpp"]))
        best_case = "exact/best"
        best_note = "Exact match; use as a positive example."
    else:
        best = max(rows, key=lambda r: (r["sentence_chrfpp"], r["rougeL"]))
        best_case = "best-overlap"
        best_note = "Highest automatic overlap; needs human semantic check."
    examples.append({**best, "case": best_case, "note": best_note})

    scores = [r["sentence_chrfpp"] for r in rows]
    med = median(scores)
    typical = min(
        rows,
        key=lambda r: (
            abs(r["sentence_chrfpp"] - med),
            abs(r["ref_tokens"] - r["hyp_tokens"]),
        ),
    )
    examples.append({
        **typical,
        "case": "representative",
        "note": "Near median chrF++; shows typical model behavior.",
    })

    repeat_rows = [r for r in rows if _max_repeat_run(r["hypothesis"]) >= 8]
    if repeat_rows:
        failure = max(repeat_rows, key=lambda r: _max_repeat_run(r["hypothesis"]))
        case = "repetition failure"
        note = "Repeated-token runaway; useful for decode-stability discussion."
    else:
        failure = min(rows, key=lambda r: (r["sentence_chrfpp"], r["rougeL"]))
        case = "worst-overlap"
        note = "Lowest automatic overlap; likely mistranslation."
    examples.append({**failure, "case": case, "note": note})

    return examples


def _run_label(cfg: Dict) -> str:
    if cfg.get("data", {}).get("max_target_tokens") is not None:
        return "LTE12 diagnostic"
    return "official full-data"


def _write_markdown(path: Path, examples: Iterable[Dict]) -> None:
    lines = [
        "# Qualitative Prediction Examples",
        "",
        "Generated from `predictions_per_example.csv` files.",
        "",
        "Selection is automatic: exact/best, median-like representative, and",
        "failure/pathology examples are selected by overlap scores and repetition",
        "patterns. Human review is still required before using these in a paper.",
        "",
        "| Track | Split | Run | Case | Row | ref/hyp tokens | chrF++ | ROUGE-L | Reference | Hypothesis | Note |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for ex in examples:
        lines.append(
            f"| {ex['track']} | {ex['split']} | `{ex['run']}` | {ex['case']} | "
            f"{ex['row_id']} | {ex['ref_tokens']} / {ex['hyp_tokens']} | "
            f"{ex['sentence_chrfpp']:.2f} | {ex['rougeL']:.3f} | "
            f"{_md(ex['reference'])} | {_md(ex['hypothesis'])} | {ex['note']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="data_full/runs")
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--out", default="docs/logs/QUALITATIVE_EXAMPLES.md")
    args = ap.parse_args()

    examples = []
    root = Path(args.runs_root)
    for run in args.runs:
        run_dir = root / run
        cfg = _read_json(run_dir / "config.json")
        rows = _read_rows(run_dir / "predictions_per_example.csv")
        for ex in _pick_examples(rows):
            examples.append({
                **ex,
                "track": _run_label(cfg),
                "split": cfg["split"]["kind"],
                "run": run,
            })

    out = Path(args.out)
    _write_markdown(out, examples)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
