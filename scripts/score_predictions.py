"""Score a saved Bornil predictions.csv file.

Use this instead of pasting large Python blocks into PowerShell. It reads the
standard run artifact written by `train_sample.py`:

    reference,hypothesis

and writes a metrics JSON next to it by default.

Examples:
    python scripts/score_predictions.py --pred data_full/runs/c19d0ea08824/predictions.csv
    python scripts/score_predictions.py --pred data_full/runs/c19d0ea08824/predictions.csv --ci 300
    python scripts/score_predictions.py --pred data_full/runs/c19d0ea08824/predictions.csv --per-example
    python scripts/score_predictions.py --pred data_full/runs/c19d0ea08824/predictions.csv --bertscore
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sacrebleu.metrics import BLEU, CHRF

from bornil.benchmark import metrics as M


def _read_predictions(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    missing = {"reference", "hypothesis"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path} missing required column(s): {sorted(missing)}")
    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    return rows, refs, hyps


def _length_stats(refs, hyps):
    ref_lens = [len(str(r).split()) for r in refs]
    hyp_lens = [len(str(h).split()) for h in hyps]
    exact = sum(r == h for r, h in zip(refs, hyps))
    empty = sum(not str(h).strip() for h in hyps)
    return {
        "rows": len(refs),
        "unique_hypotheses": len(set(hyps)),
        "empty_hypotheses": empty,
        "exact_matches": exact,
        "exact_match_rate": exact / len(refs),
        "avg_ref_tokens": sum(ref_lens) / len(ref_lens),
        "avg_hyp_tokens": sum(hyp_lens) / len(hyp_lens),
    }


def _write_per_example(path: Path, rows):
    bleu = BLEU(tokenize=M.BLEU_TOKENIZE, effective_order=True)
    chrf = CHRF(word_order=0)
    chrfpp = CHRF(word_order=2)
    out_path = path.with_name(path.stem + "_per_example.csv")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "row_id", "reference", "hypothesis", "exact_match",
            "ref_tokens", "hyp_tokens", "sentence_bleu4",
            "sentence_chrf", "sentence_chrfpp", "rougeL",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            ref = row["reference"]
            hyp = row["hypothesis"]
            writer.writerow({
                "row_id": i,
                "reference": ref,
                "hypothesis": hyp,
                "exact_match": ref == hyp,
                "ref_tokens": len(str(ref).split()),
                "hyp_tokens": len(str(hyp).split()),
                "sentence_bleu4": bleu.sentence_score(hyp, [ref]).score,
                "sentence_chrf": chrf.sentence_score(hyp, [ref]).score,
                "sentence_chrfpp": chrfpp.sentence_score(hyp, [ref]).score,
                "rougeL": M.rouge_l(hyp, ref),
            })
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="path to predictions.csv")
    ap.add_argument("--out", default=None, help="metrics JSON output path")
    ap.add_argument("--ci", type=int, default=0,
                    help="bootstrap sample count; 0 disables CI")
    ap.add_argument("--ci-seed", type=int, default=42)
    ap.add_argument("--per-example", action="store_true",
                    help="write predictions_per_example.csv next to predictions.csv")
    ap.add_argument("--bertscore", action="store_true",
                    help="compute optional BanglaBERT BERTScore F1")
    ap.add_argument("--bertscore-model", default="csebuetnlp/banglabert")
    ap.add_argument("--bertscore-num-layers", type=int, default=12,
                    help="layer count for custom BERTScore models")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    rows, refs, hyps = _read_predictions(pred_path)

    result = {
        "predictions": str(pred_path),
        "corpus_metrics": M.corpus_metrics(hyps, refs),
        "summary": _length_stats(refs, hyps),
    }

    if args.ci:
        result["bootstrap_ci"] = {}
        for metric in ("bleu4", "chrf++", "rougeL"):
            lo, hi = M.bootstrap_ci(hyps, refs, metric=metric,
                                    n=args.ci, seed=args.ci_seed)
            result["bootstrap_ci"][metric] = {
                "samples": args.ci,
                "seed": args.ci_seed,
                "lo": lo,
                "hi": hi,
            }

    if args.bertscore:
        result["semantic_metrics"] = {
            "bertscore_f1": M.bertscore(
                hyps, refs,
                model=args.bertscore_model,
                num_layers=args.bertscore_num_layers,
            ),
            "model": args.bertscore_model,
            "num_layers": args.bertscore_num_layers,
        }

    if args.per_example:
        result["per_example_csv"] = str(_write_per_example(pred_path, rows))

    out_path = Path(args.out) if args.out else pred_path.with_name("score_metrics.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
