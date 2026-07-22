"""WS-E Task A2 — extract CTC forced alignments from a trained checkpoint.

Runs the A1 Viterbi forced alignment (bornil/align/ctc_forced) over the test
split of a joint-CTC run and writes one CSV row per aligned label occurrence:
which subword sits on which frame span, plus a per-label peak-posterior
confidence. The confidence exists because A0 found the head is sparse (97.4%
blank) and only ~2.7 of ~12.2 target subwords per clip have a confident spike —
so forced alignment yields a few well-anchored labels and many weak placements,
and Phase B must be able to tell them apart. See docs/logs/WSE_A0_POSTERIOR.md.

    python scripts/align_checkpoint.py --run data_full/runs/b0a590ea3317 \
        --device cuda --out data_full/runs/b0a590ea3317/alignments.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # torch before pandas (FAIL-002)

import numpy as np
import pandas as pd
import torch

from bornil.align.ctc_forced import AlignmentError, align_clip
from bornil.benchmark.splits import filter_split_by_recordings, resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.models.build import build_model

Span = Tuple[int, int, int]


def span_confidences(log_probs, spans: Sequence[Span],
                     labels: Sequence[int]) -> List[float]:
    """Peak posterior of each span's own label token over its assigned frames.

    High when the head actually favours the label somewhere in the span (a real
    anchor); near zero when forced alignment parked the label on blank frames.
    """
    if isinstance(log_probs, torch.Tensor):
        lp = log_probs.detach().float().cpu().numpy()
    else:
        lp = np.asarray(log_probs, dtype=np.float64)
    out: List[float] = []
    for li, s, e in spans:
        token_id = int(labels[li])
        window = lp[s:e + 1, token_id]
        out.append(float(np.exp(window.max())) if window.size else 0.0)
    return out


def spans_to_rows(row, spans: Sequence[Span], labels: Sequence[int],
                  tokenizer, n_frames: int,
                  peak_conf: Sequence[float] = None) -> List[Dict]:
    """One dict per span: (label_index, token, frame span, n_frames, confidence)."""
    rows: List[Dict] = []
    for i, (li, start, end) in enumerate(spans):
        token_id = int(labels[li])
        rows.append({
            "row": row,
            "label_index": li,
            "token_id": token_id,
            "token_str": tokenizer.decode([token_id]),
            "start_frame": start,
            "end_frame": end,
            "n_frames": n_frames,
            "peak_conf": (float(peak_conf[i]) if peak_conf is not None else None),
        })
    return rows


def _labels_from_target(target: Sequence[int], specials) -> List[int]:
    """CTC labels are the target ids with BOS/EOS/pad removed."""
    return [int(t) for t in target if int(t) not in specials]


@torch.no_grad()
def align_split(model, dataset, sampler, tokenizer, device, blank,
                recordings) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Forced-align every clip in `dataset`, iterating `sampler`'s index batches.

    Returns the alignment rows and a summary dict (clips, aligned, infeasible,
    empty_target, labels).
    """
    collate = make_collate_fn(tokenizer.pad_id)
    specials = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}
    all_rows: List[Dict] = []
    stats = {"clips": 0, "aligned": 0, "infeasible": 0, "empty_target": 0, "labels": 0}

    for batch_idx in sampler:
        items = [dataset[i] for i in batch_idx]
        x, _y, x_lengths, _yl = collate(items)
        x = x.to(device)
        src_kpm = model.src_padding_mask(x_lengths.to(device), x.size(1))
        log_probs = model.ctc_logits(x, src_key_padding_mask=src_kpm).log_softmax(-1)

        for b, ds_idx in enumerate(batch_idx):
            stats["clips"] += 1
            t = int(x_lengths[b])
            labels = _labels_from_target(dataset.targets[ds_idx], specials)
            if not labels:
                stats["empty_target"] += 1
                continue
            clip_lp = log_probs[b, :t]
            try:
                spans = align_clip(clip_lp, labels, blank=blank)
            except AlignmentError:
                stats["infeasible"] += 1
                continue
            conf = span_confidences(clip_lp, spans, labels)
            all_rows.extend(spans_to_rows(
                row=recordings[ds_idx], spans=spans, labels=labels,
                tokenizer=tokenizer, n_frames=t, peak_conf=conf))
            stats["aligned"] += 1
            stats["labels"] += len(spans)

    df = pd.DataFrame(all_rows, columns=[
        "row", "label_index", "token_id", "token_str",
        "start_frame", "end_frame", "n_frames", "peak_conf"])
    return df, stats


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with config.json and best.pt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split-root", default="splits")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not bool(cfg["model"].get("ctc_head", False)):
        raise SystemExit(f"run {run_dir} was not trained with a CTC head; A2 needs one")
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    device = resolve_device(args.device)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    data_root = Path(cfg["trial"]["out_dir"])
    man = pd.read_csv(data_root / "sample_manifest.csv")
    features = np.load(data_root / "packed.npy", mmap_mode="r")
    lengths = np.load(data_root / "lengths.npy")
    tok = Tokenizer(cfg["data"]["tokenizer"])
    blank = tok.pad_id
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]), split_root=args.split_root, prefer_frozen=True,
    )
    sp, split_source = _apply_target_length_filter(cfg, man, sp, split_source)
    test_recs = sp["test"][:args.limit_test] if args.limit_test else sp["test"]

    rows = man[man["recording"].isin(set(test_recs))]
    pack_idx = rows["pack_idx"].tolist()
    recordings = rows["recording"].tolist()
    targets = [targets_by_rec[r] for r in recordings]
    ds = SignDataset(features, lengths, pack_idx, targets,
                     normalize=cfg["data"]["normalize"], upper_body=cfg["data"]["upper_body"])
    sampler = LengthBucketBatchSampler(ds.item_lengths(), int(cfg["train"]["batch_size"]),
                                       shuffle=False, seed=cfg["trial"]["seed"])

    model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    print(f"run={run_dir} | device={device} | split={cfg['split']['kind']} "
          f"({split_source}) | test={len(test_recs)} | blank={blank}")

    df, stats = align_split(model, ds, sampler, tok, device, blank, recordings)

    infeas_pct = 100.0 * stats["infeasible"] / max(stats["clips"], 1)
    print(f"aligned {stats['aligned']}/{stats['clips']} clips | {stats['labels']} labels | "
          f"infeasible: {stats['infeasible']} ({infeas_pct:.1f}%) | "
          f"empty target: {stats['empty_target']}")
    if not df.empty:
        anchored = (df["peak_conf"] >= 0.5).mean()
        print(f"per-label confidence: median={df['peak_conf'].median():.3f} "
              f"mean={df['peak_conf'].mean():.3f} | anchored (>=0.5)={anchored*100:.1f}% "
              f"| avg labels/clip={stats['labels']/max(stats['aligned'],1):.1f}")
    if infeas_pct > 5.0:
        print("WARNING: infeasible rate > 5% — likely the frame ceiling truncating "
              "long targets. Investigate before Phase B (log to FAILURES.md).")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
