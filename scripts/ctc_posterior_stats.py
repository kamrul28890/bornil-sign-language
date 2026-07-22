"""WS-E Task A0 — is the CTC head temporally informative? (gate for Phase B)

Re-uses a trained joint-CTC checkpoint to report per-frame CTC posterior
statistics over the test split, and contrasts them against frame-shuffled input.
If the head is degenerate (blank almost everywhere, ~1 distinct label per clip),
forced alignment (A1) produces arbitrary segmentations and Phase B cannot yield
an interpretable agreement-vs-chance result. See the plan amendments in
docs/plan/2026-07-16-ws-e-alignment.md.

    python scripts/ctc_posterior_stats.py --run data_full/runs/b0a590ea3317 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # torch before pandas (FAIL-002)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bornil.align.posterior import aggregate_stats, clip_posterior_stats
from bornil.benchmark.splits import filter_split_by_recordings, resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.models.build import build_model


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


@torch.no_grad()
def _collect_stats(model, loader, device, blank, shuffle_frames, seed):
    """Per-clip posterior stats over the loader; optionally shuffle frames first."""
    rng = np.random.default_rng(seed)
    per_clip = []
    for x, _y, x_lengths, _yl in loader:
        x = x.to(device)
        if shuffle_frames:
            # Permute only the real frames of each clip, leaving padding in place.
            x = x.clone()
            for b in range(x.size(0)):
                t = int(x_lengths[b])
                perm = torch.from_numpy(rng.permutation(t)).to(device)
                x[b, :t] = x[b, :t][perm]
        src_kpm = model.src_padding_mask(x_lengths.to(device), x.size(1))
        logits = model.ctc_logits(x, src_key_padding_mask=src_kpm)   # (B, S, V)
        log_probs = logits.log_softmax(-1)
        for b in range(x.size(0)):
            t = int(x_lengths[b])
            per_clip.append(clip_posterior_stats(log_probs[b, :t], blank=blank))
    return aggregate_stats(per_clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with config.json and best.pt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split-root", default="splits")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not bool(cfg["model"].get("ctc_head", False)):
        raise SystemExit(f"run {run_dir} was not trained with a CTC head; A0 needs one")
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
    blank = tok.pad_id      # CTC blank == pad_id == 0 (WS-C Phase B convention)
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]), split_root=args.split_root, prefer_frozen=True,
    )
    sp, split_source = _apply_target_length_filter(cfg, man, sp, split_source)
    test_recs = sp["test"][:args.limit_test] if args.limit_test else sp["test"]

    rows = man[man["recording"].isin(set(test_recs))]
    pack_idx = rows["pack_idx"].tolist()
    targets = [targets_by_rec[r] for r in rows["recording"]]
    ds = SignDataset(features, lengths, pack_idx, targets,
                     normalize=cfg["data"]["normalize"], upper_body=cfg["data"]["upper_body"])
    sampler = LengthBucketBatchSampler(ds.item_lengths(), int(cfg["train"]["batch_size"]),
                                       shuffle=False, seed=cfg["trial"]["seed"])
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(tok.pad_id))

    model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    print(f"run={run_dir} | device={device} | split={cfg['split']['kind']} "
          f"({split_source}) | test={len(test_recs)} | vocab={tok.vocab_size} | blank={blank}")

    real = _collect_stats(model, loader, device, blank, shuffle_frames=False, seed=args.seed)
    shuf = _collect_stats(model, loader, device, blank, shuffle_frames=True, seed=args.seed)

    def line(tag, s):
        print(f"{tag:14s} blank_rate={s['blank_rate']:.3f} "
              f"distinct_nonblank={s['distinct_nonblank']:.2f} "
              f"mean_run_len={s['mean_run_length']:.2f} "
              f"entropy={s['mean_entropy']:.3f} ratio={s['entropy_ratio']:.3f}")

    line("real", real)
    line("frame_shuffle", shuf)

    # Gate (plan A0): degenerate if blank dominates AND ~1 label per clip.
    degenerate = real["blank_rate"] > 0.95 and real["distinct_nonblank"] < 1.5
    verdict = ("DEGENERATE - alignment uninterpretable; log to FAILURES.md and stop"
               if degenerate else
               "OK - CTC head is informative enough to proceed to A2/Phase B")
    # Time-sensitivity: does the head respond to frame ORDER? blank_rate is the
    # wrong probe -- most frames are blank either way, so it barely moves. The
    # responsive statistic is how many distinct labels fire: if that collapses
    # under shuffling, temporal order genuinely drives which labels the head emits.
    denom = max(real["distinct_nonblank"], 1e-9)
    label_drop = (real["distinct_nonblank"] - shuf["distinct_nonblank"]) / denom
    dblank = abs(real["blank_rate"] - shuf["blank_rate"])
    time_sensitive = label_drop > 0.15
    print(f"\nGATE: {verdict}")
    print(f"time-sensitivity: distinct-label count drops {label_drop*100:.0f}% under "
          f"frame shuffle ({real['distinct_nonblank']:.2f} -> {shuf['distinct_nonblank']:.2f}) "
          f"-> {'temporally responsive' if time_sensitive else 'near time-invariant'}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "run": str(run_dir), "split_source": split_source, "n_test": len(test_recs),
            "blank": blank, "real": real, "frame_shuffle": shuf,
            "degenerate": degenerate, "blank_rate_shift": dblank,
            "distinct_label_drop": label_drop, "time_sensitive": time_sensitive,
        }, indent=2), encoding="utf-8")
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
