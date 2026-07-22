"""Measure how much a trained checkpoint actually uses its visual input.

Re-decodes the test split with the keypoint stream corrupted in ways that
destroy visual information while leaving the decoder untouched. If scores hold
up under corruption, the encoder is decorative and the decoder's Bangla prior is
producing the output.

Corruption modes:
    none          control; should reproduce the run's recorded metrics
    mismatch      pair each reference with a DIFFERENT clip's video (derangement).
                  Feature statistics stay real; only the correspondence dies.
                  This is the primary test.
    frame_shuffle shuffle frames within each clip -- kills temporal order,
                  preserves the pose distribution. Separates "uses motion" from
                  "uses average pose".
    zero          all-zero features -- pure decoder prior.
    noise         Gaussian noise matched to the split's per-dim mean/std.

    python scripts/input_ablation.py --run-dir data_full/runs/b0a590ea3317 \
        --mode none mismatch frame_shuffle zero noise
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
from torch.utils.data import DataLoader, Dataset

from bornil.benchmark.splits import filter_split_by_recordings, resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.models.build import build_model
from bornil.train.loop import evaluate

MODES = ("none", "mismatch", "frame_shuffle", "zero", "noise")


class CorruptedDataset(Dataset):
    """Wraps SignDataset and corrupts only the feature tensor, never the target."""

    def __init__(self, base: SignDataset, mode: str, seed: int = 42,
                 stats: tuple | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.base = base
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.base)

    def item_lengths(self):
        return self.base.item_lengths()

    def __getitem__(self, i: int):
        x, y = self.base[i]
        if self.mode in ("none", "mismatch"):
            # `mismatch` is applied by permuting pack_idx upstream, not here.
            return x, y
        if self.mode == "zero":
            return torch.zeros_like(x), y
        if self.mode == "frame_shuffle":
            perm = torch.from_numpy(self.rng.permutation(x.shape[0]))
            return x[perm], y
        if self.mode == "noise":
            mean, std = self.stats
            noise = self.rng.normal(mean, std, size=tuple(x.shape)).astype(np.float32)
            return torch.from_numpy(noise), y
        raise AssertionError(self.mode)


def _derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation with no fixed points, so no clip keeps its own target."""
    if n < 2:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def _feature_stats(ds: SignDataset, rng: np.random.Generator, sample: int = 200):
    """Per-dimension mean/std over a sample of clips, for the `noise` mode."""
    idx = rng.choice(len(ds), size=min(sample, len(ds)), replace=False)
    frames = np.concatenate([ds[int(i)][0].numpy() for i in idx], axis=0)
    return frames.mean(axis=0), frames.std(axis=0) + 1e-6


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return pref


def _apply_target_length_filter(cfg, manifest, split, split_source):
    max_tokens = cfg.get("data", {}).get("max_target_tokens")
    if max_tokens is None:
        return split, split_source
    max_tokens = int(max_tokens)
    lengths = manifest["clean_text"].astype(str).str.split().map(len)
    allowed = set(manifest.loc[lengths <= max_tokens, "recording"])
    return filter_split_by_recordings(split, allowed), \
        f"{split_source}|max_target_tokens<={max_tokens}"


def _length_stats(refs, hyps):
    exact = sum(r == h for r, h in zip(refs, hyps))
    return {
        "rows": len(refs),
        "unique_hypotheses": len(set(hyps)),
        "empty_hypotheses": sum(not str(h).strip() for h in hyps),
        "exact_matches": exact,
        "avg_ref_tokens": float(np.mean([len(str(r).split()) for r in refs])) if refs else 0.0,
        "avg_hyp_tokens": float(np.mean([len(str(h).split()) for h in hyps])) if hyps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split-root", default="splits")
    ap.add_argument("--mode", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "input_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cfg["train"]["device"] = device
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    max_len = int(cfg["train"]["max_decode_len"])

    data_root = Path(cfg["trial"]["out_dir"])
    man = pd.read_csv(data_root / "sample_manifest.csv")
    features = np.load(data_root / "packed.npy", mmap_mode="r")
    lengths = np.load(data_root / "lengths.npy")
    tok = Tokenizer(cfg["data"]["tokenizer"])
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]), split_root=args.split_root, prefer_frozen=True,
    )
    sp, split_source = _apply_target_length_filter(cfg, man, sp, split_source)
    test_recs = sp["test"][:args.limit_test] if args.limit_test else sp["test"]

    rows = man[man["recording"].isin(set(test_recs))]
    pack_idx = np.array(rows["pack_idx"].tolist())
    targets = [targets_by_rec[r] for r in rows["recording"]]

    model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    print(f"run={run_dir} | checkpoint={checkpoint}")
    print(f"device={device} | split={cfg['split']['kind']} ({split_source}) | "
          f"test={len(test_recs)} | modes={args.mode}")

    summary_rows = []
    for mode in args.mode:
        rng = np.random.default_rng(args.seed)
        # `mismatch` breaks the video<->text correspondence by deranging which
        # clip's features sit opposite each target; everything else is identical.
        idx = pack_idx[_derange(len(pack_idx), rng)] if mode == "mismatch" else pack_idx

        base = SignDataset(features, lengths, idx.tolist(), targets,
                           normalize=cfg["data"]["normalize"],
                           upper_body=cfg["data"]["upper_body"])
        stats = _feature_stats(base, rng) if mode == "noise" else None
        ds = CorruptedDataset(base, mode, seed=args.seed, stats=stats)

        sampler = LengthBucketBatchSampler(
            ds.item_lengths(), int(cfg["train"]["batch_size"]),
            shuffle=False, seed=cfg["trial"]["seed"],
        )
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(tok.pad_id))

        t0 = time.time()
        metrics, hyps, refs = evaluate(
            model, loader, tok, device=device, max_len=max_len,
            beam_size=1, length_penalty=1.0, repetition_stop=0,
            progress=args.progress, desc=f"ablation {mode}",
        )
        mins = (time.time() - t0) / 60

        summary = _length_stats(refs, hyps)
        (out_dir / f"{mode}_metrics.json").write_text(json.dumps({
            "run_dir": str(run_dir), "mode": mode, "split": cfg["split"]["kind"],
            "split_source": split_source, "seed": args.seed, "minutes": mins,
            "corpus_metrics": metrics, "summary": summary,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        with (out_dir / f"{mode}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["reference", "hypothesis"])
            w.writeheader()
            for ref, hyp in zip(refs, hyps):
                w.writerow({"reference": ref, "hypothesis": hyp})

        summary_rows.append({
            "mode": mode, "minutes": f"{mins:.2f}",
            "bleu4": f"{metrics['bleu4']:.4f}", "chrf++": f"{metrics['chrf++']:.4f}",
            "rougeL": f"{metrics['rougeL']:.4f}",
            "avg_hyp_tokens": f"{summary['avg_hyp_tokens']:.2f}",
            "unique_hypotheses": summary["unique_hypotheses"],
            "exact_matches": summary["exact_matches"],
        })
        print(f"{mode:14s} BLEU-4={metrics['bleu4']:.3f} chrF++={metrics['chrf++']:.3f} "
              f"ROUGE-L={metrics['rougeL']:.3f} uniq={summary['unique_hypotheses']} "
              f"exact={summary['exact_matches']} time={mins:.1f}m")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
