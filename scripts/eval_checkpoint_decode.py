"""Evaluate an existing checkpoint with alternate decode settings.

This is for cheap decode ablations after a model has already been trained. It
rebuilds the test loader from the saved run config, loads `best.pt`, and writes
new prediction/metric files without touching the original training artifacts.

Examples:
    python scripts/eval_checkpoint_decode.py --run-dir data_full/runs/c19d0ea08824 --beam-size 1 5
    python scripts/eval_checkpoint_decode.py --run-dir data_full/runs/46a706570712 --beam-size 5 --length-penalty 0.8 1.0 1.2
    python scripts/eval_checkpoint_decode.py --run-dir data_full/runs/c19d0ea08824 --beam-size 5 --limit-test 64
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # import torch before pandas (FAIL-002)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bornil.benchmark.splits import filter_split_by_recordings, resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.data.transforms import feature_dim
from bornil.models.transformer import SignTransformer
from bornil.train.loop import evaluate


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return pref


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _subset(manifest, recordings, targets_by_rec):
    rows = manifest[manifest["recording"].isin(set(recordings))]
    pack_idx = rows["pack_idx"].tolist()
    targets = [targets_by_rec[r] for r in rows["recording"]]
    return pack_idx, targets


def _apply_target_length_filter(cfg, manifest, split, split_source):
    max_tokens = cfg.get("data", {}).get("max_target_tokens")
    if max_tokens is None:
        return split, split_source
    max_tokens = int(max_tokens)
    lengths = manifest["clean_text"].astype(str).str.split().map(len)
    allowed = set(manifest.loc[lengths <= max_tokens, "recording"])
    filtered = filter_split_by_recordings(split, allowed)
    split_source = f"{split_source}|max_target_tokens<={max_tokens}"
    return filtered, split_source


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
        "exact_match_rate": exact / len(refs) if refs else 0.0,
        "avg_ref_tokens": sum(ref_lens) / len(ref_lens) if ref_lens else 0.0,
        "avg_hyp_tokens": sum(hyp_lens) / len(hyp_lens) if hyp_lens else 0.0,
    }


def _write_predictions(path: Path, refs, hyps):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reference", "hypothesis"])
        writer.writeheader()
        for ref, hyp in zip(refs, hyps):
            writer.writerow({"reference": ref, "hypothesis": hyp})


def _decode_name(beam_size: int, length_penalty: float, max_len: int,
                 repetition_stop: int = 0) -> str:
    mode = "greedy" if beam_size <= 1 else f"beam{beam_size}_lp{length_penalty:g}"
    suffix = f"_repstop{repetition_stop}" if repetition_stop else ""
    return f"{mode}_max{max_len}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="existing run directory containing config.json and best.pt")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint path; defaults to <run-dir>/best.pt")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--split-root", default="splits")
    ap.add_argument("--beam-size", type=int, nargs="+", default=[1],
                    help="one or more beam sizes to evaluate")
    ap.add_argument("--length-penalty", type=float, nargs="+", default=[1.0],
                    help="one or more beam length penalties; used only for beam_size > 1")
    ap.add_argument("--max-decode-len", type=int, default=None,
                    help="override config train.max_decode_len")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override config train.batch_size for evaluation")
    ap.add_argument("--limit-test", type=int, default=None,
                    help="evaluate only the first N test recordings for a quick smoke run")
    ap.add_argument("--out-dir", default=None,
                    help="output directory; defaults to <run-dir>/decode_ablations")
    ap.add_argument("--progress", action="store_true",
                    help="show tqdm decode progress")
    ap.add_argument("--repetition-stop", type=int, default=0,
                    help="stop a hypothesis when one token repeats this many times; 0 disables")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = _read_json(run_dir / "config.json")
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "decode_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cfg["train"]["device"] = device
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    max_len = args.max_decode_len or int(cfg["train"]["max_decode_len"])

    data_root = Path(cfg["trial"]["out_dir"])
    man = pd.read_csv(data_root / "sample_manifest.csv")
    features = np.load(data_root / "packed.npy", mmap_mode="r")
    lengths = np.load(data_root / "lengths.npy")
    tok = Tokenizer(cfg["data"]["tokenizer"])
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man,
        kind=cfg["split"]["kind"],
        seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]),
        split_root=args.split_root,
        prefer_frozen=True,
    )
    sp, split_source = _apply_target_length_filter(cfg, man, sp, split_source)
    test_recs = sp["test"][:args.limit_test] if args.limit_test else sp["test"]
    pack_idx, targets = _subset(man, test_recs, targets_by_rec)
    ds = SignDataset(features, lengths, pack_idx, targets,
                     normalize=cfg["data"]["normalize"],
                     upper_body=cfg["data"]["upper_body"])
    sampler = LengthBucketBatchSampler(
        ds.item_lengths(),
        int(cfg["train"]["batch_size"]),
        shuffle=False,
        seed=cfg["trial"]["seed"],
    )
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(tok.pad_id))

    m = cfg["model"]
    model = SignTransformer(
        input_dim=feature_dim(cfg["data"]["upper_body"]),
        vocab_size=tok.vocab_size,
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        pad_id=tok.pad_id,
        temporal_conv_layers=int(m.get("temporal_conv_layers", 0)),
        temporal_conv_kernel=int(m.get("temporal_conv_kernel", 5)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    print(f"run={run_dir} | checkpoint={checkpoint}")
    print(f"device={device} | split={cfg['split']['kind']} ({split_source}) | "
          f"test={len(test_recs)} | batch_size={cfg['train']['batch_size']} | max_len={max_len}")

    summary_rows = []
    for beam_size in args.beam_size:
        penalties = [1.0] if beam_size <= 1 else args.length_penalty
        for length_penalty in penalties:
            name = _decode_name(beam_size, length_penalty, max_len, args.repetition_stop)
            t0 = time.time()
            metrics, hyps, refs = evaluate(
                model,
                loader,
                tok,
                device=device,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                repetition_stop=args.repetition_stop,
                progress=args.progress,
                desc=f"decode {name}",
            )
            mins = (time.time() - t0) / 60
            result = {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "split": cfg["split"]["kind"],
                "split_source": split_source,
                "decode": {
                    "beam_size": beam_size,
                    "length_penalty": length_penalty,
                    "max_decode_len": max_len,
                    "repetition_stop": args.repetition_stop,
                },
                "minutes": mins,
                "corpus_metrics": metrics,
                "summary": _length_stats(refs, hyps),
            }

            pred_path = out_dir / f"{name}_predictions.csv"
            metrics_path = out_dir / f"{name}_metrics.json"
            _write_predictions(pred_path, refs, hyps)
            metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
            summary_rows.append({
                "decode": name,
                "minutes": f"{mins:.2f}",
                "bleu4": f"{metrics['bleu4']:.4f}",
                "chrf++": f"{metrics['chrf++']:.4f}",
                "rougeL": f"{metrics['rougeL']:.4f}",
                "avg_hyp_tokens": f"{result['summary']['avg_hyp_tokens']:.2f}",
                "unique_hypotheses": result["summary"]["unique_hypotheses"],
                "predictions": str(pred_path),
                "metrics": str(metrics_path),
            })
            print(f"{name}: BLEU-4={metrics['bleu4']:.3f} "
                  f"chrF++={metrics['chrf++']:.3f} ROUGE-L={metrics['rougeL']:.3f} "
                  f"avg_hyp_tokens={result['summary']['avg_hyp_tokens']:.2f} "
                  f"time={mins:.1f}m")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "decode", "minutes", "bleu4", "chrf++", "rougeL",
            "avg_hyp_tokens", "unique_hypotheses", "predictions", "metrics",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
