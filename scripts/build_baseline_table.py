"""Build the main full-dataset baseline table from saved run artifacts.

The script scans data_full/runs/*/config.json, finds the latest completed
unfiltered `data_full` run for each official split, reads score_metrics.json
when available, and writes a paper-friendly Markdown summary.

Example:
    python scripts/build_baseline_table.py
    python scripts/build_baseline_table.py --out docs/logs/BASELINE_TABLE.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_SPLITS = ("random-v1", "unseen-sentence-v1", "signerindep-v1")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_mtime(run_dir: Path) -> float:
    mtimes = [p.stat().st_mtime for p in run_dir.glob("*") if p.is_file()]
    return max(mtimes) if mtimes else run_dir.stat().st_mtime


def _find_runs(runs_root: Path, splits: Iterable[str],
               trial_name: Optional[str] = "data_full",
               include_filtered: bool = False) -> Dict[str, Optional[Path]]:
    wanted = set(splits)
    found = {split: [] for split in wanted}

    if not runs_root.exists():
        return {split: None for split in wanted}

    for run_dir in runs_root.iterdir():
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        pred_path = run_dir / "predictions.csv"
        if not run_dir.is_dir() or not cfg_path.exists() or not metrics_path.exists():
            continue
        if not pred_path.exists():
            continue
        try:
            cfg = _load_json(cfg_path)
        except json.JSONDecodeError:
            continue
        if trial_name and cfg.get("trial", {}).get("name") != trial_name:
            continue
        if not include_filtered and cfg.get("data", {}).get("max_target_tokens") is not None:
            continue
        split = cfg.get("split", {}).get("kind")
        if split in wanted:
            found[split].append(run_dir)

    return {
        split: max(paths, key=_run_mtime) if paths else None
        for split, paths in found.items()
    }


def _read_result(run_dir: Path) -> Dict[str, Any]:
    score_path = run_dir / "score_metrics.json"
    metrics_path = run_dir / "metrics.json"
    cfg = _load_json(run_dir / "config.json")

    if score_path.exists():
        score = _load_json(score_path)
        metrics = score["corpus_metrics"]
        summary = score.get("summary", {})
        ci = score.get("bootstrap_ci", {})
        semantic = score.get("semantic_metrics", {})
    else:
        raw = _load_json(metrics_path)
        metrics = raw["test"]
        summary = {}
        ci = {}
        semantic = {}

    raw = _load_json(metrics_path)
    return {
        "hash": run_dir.name,
        "path": run_dir.as_posix(),
        "cfg": cfg,
        "split_counts": _split_counts(cfg),
        "metrics": metrics,
        "summary": summary,
        "ci": ci,
        "semantic": semantic,
        "best_val_bleu4": raw.get("best_val_bleu4"),
        "train_min": raw.get("train_min"),
        "has_score_metrics": score_path.exists(),
    }


def _count_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _split_counts(cfg: Dict[str, Any]) -> Dict[str, Optional[int]]:
    after_filter = cfg.get("split", {}).get("filter", {}).get("after")
    if after_filter:
        return {
            "train": after_filter.get("train"),
            "val": after_filter.get("val"),
            "test": after_filter.get("test"),
        }

    source = cfg.get("split", {}).get("source", "")
    if not source.startswith("frozen:"):
        return {"train": None, "val": None, "test": None}

    source = source.split("|", 1)[0]
    split_dir = Path(source.removeprefix("frozen:"))
    return {
        "train": _count_lines(split_dir / "train.txt"),
        "val": _count_lines(split_dir / "val.txt"),
        "test": _count_lines(split_dir / "test.txt"),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(ci: Dict[str, Any], metric: str) -> str:
    item = ci.get(metric)
    if not item:
        return "-"
    return f"[{item['lo']:.3f}, {item['hi']:.3f}]"


def _row(split: str, result: Optional[Dict[str, Any]]) -> str:
    if result is None:
        return f"| {split} | pending | - | - | - | - | - | - | - | - | - | - |"

    metrics = result["metrics"]
    cfg = result["cfg"]
    counts = result["split_counts"]
    summary = result["summary"]
    semantic = result["semantic"]
    train_n = counts.get("train") or "-"
    test_n = summary.get("rows") or counts.get("test") or "-"
    source = cfg.get("split", {}).get("source", "unknown")
    decode = cfg.get("train", {}).get("decode_mode", "unknown")
    bertscore = semantic.get("bertscore_f1")
    exact = summary.get("exact_match_rate")
    unique = summary.get("unique_hypotheses")
    if exact is not None:
        exact = exact * 100

    return (
        f"| {split} | {result['hash']} | {source} | {decode} | "
        f"{train_n} | {test_n} | "
        f"{metrics['bleu4']:.3f} | {metrics['chrf++']:.3f} | "
        f"{metrics['rougeL']:.3f} | {_fmt(bertscore, 3)} | "
        f"{_fmt(exact, 2)} | {unique if unique is not None else '-'} |"
    )


def _detail(split: str, result: Optional[Dict[str, Any]]) -> str:
    if result is None:
        return f"### {split}\n\nPending.\n"

    metrics = result["metrics"]
    summary = result["summary"]
    lines = [
        f"### {split}",
        "",
        f"- Run: `{result['path']}`",
        f"- Train time: {_fmt(result['train_min'], 1)} min",
        f"- Best validation BLEU-4: {_fmt(result['best_val_bleu4'], 3)}",
        f"- BLEU-4 CI: {_ci(result['ci'], 'bleu4')}",
        f"- chrF++ CI: {_ci(result['ci'], 'chrf++')}",
        f"- ROUGE-L CI: {_ci(result['ci'], 'rougeL')}",
        f"- BLEU-1/2/3/4: {metrics['bleu1']:.3f} / {metrics['bleu2']:.3f} / "
        f"{metrics['bleu3']:.3f} / {metrics['bleu4']:.3f}",
    ]
    if summary:
        lines.extend([
            f"- Empty hypotheses: {summary.get('empty_hypotheses', '-')}",
            f"- Exact matches: {summary.get('exact_matches', '-')}",
            f"- Average reference/hypothesis tokens: "
            f"{_fmt(summary.get('avg_ref_tokens'), 2)} / "
            f"{_fmt(summary.get('avg_hyp_tokens'), 2)}",
        ])
    if not result["has_score_metrics"]:
        lines.append("- Note: detailed scoring is missing; run score_predictions.py for CI/BERTScore.")
    lines.append("")
    return "\n".join(lines)


def build_table(args: argparse.Namespace) -> str:
    splits = tuple(args.splits)
    run_dirs = _find_runs(Path(args.runs_root), splits,
                          trial_name=args.trial_name,
                          include_filtered=args.include_filtered)
    results = {
        split: _read_result(run_dir) if run_dir is not None else None
        for split, run_dir in run_dirs.items()
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Bornil Full-Dataset Baseline Table",
        "",
        f"Generated: {now}",
        "",
        "| Split | Run | Source | Decode | Train clips | Test rows | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Exact % | Unique hyps |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_row(split, results[split]) for split in splits)
    lines.extend(["", "## Details", ""])
    lines.extend(_detail(split, results[split]) for split in splits)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="data_full/runs")
    ap.add_argument("--out", default="docs/logs/BASELINE_TABLE.md")
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--trial-name", default="data_full",
                    help="trial.name to select; empty string allows any trial")
    ap.add_argument("--include-filtered", action="store_true",
                    help="allow filtered runs such as data.max_target_tokens experiments")
    args = ap.parse_args()
    if args.trial_name == "":
        args.trial_name = None

    text = build_table(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
