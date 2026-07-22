"""WS-E Task B3 — validate CTC forced alignments against an independent signal.

No gold alignment exists for Bornil, so the defensible claim is: CTC label
placements coincide with keypoint-motion changepoints more often than chance.
This script computes that, per run, over the test split, and writes a Markdown
verdict.

Two things this script does beyond the plan, both forced by earlier findings:

- Anchors vs all labels. A2 found only ~0.85 of ~12 forced labels/clip land on a
  frame the CTC head actually favours (peak_conf >= 0.5); the rest are Viterbi
  tie-breaks on blank frames. Agreement is reported for the anchor subset AND all
  labels, because mixing ~11 noise placements into ~1 real one would mask signal.
- Minima and maxima references. The plan assumes labels fall at velocity minima
  (between-sign rests). If they instead track articulation peaks, a minima-only
  test would read as chance for the wrong reason, so peaks are reported too.

    python scripts/alignment_report.py --run data_full/runs/b0a590ea3317 \
        --out docs/logs/ALIGNMENT_VALIDATION.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # torch before pandas (FAIL-002)

import numpy as np
import pandas as pd

from bornil.align.agreement import boundary_f1, chance_boundary_f1, shift_null_f1
from bornil.align.motion import changepoints, frame_velocity, smooth_velocity
from bornil.benchmark.splits import resolve_split
from bornil.data.transforms import prepare_features

ANCHOR_MIN_CONF = 0.5


def pred_boundaries(df_clip: pd.DataFrame,
                    anchor_min_conf: Optional[float] = None) -> List[int]:
    """Sorted CTC label placement frames for one clip (optionally anchors only)."""
    df = df_clip
    if anchor_min_conf is not None:
        df = df[df["peak_conf"] >= anchor_min_conf]
    return sorted(int(f) for f in df["start_frame"].tolist())


def _agreement_record(pred: List[int], ref: List[int], n_frames: int,
                      tolerance: int, trials: int, seed: int) -> Optional[Dict]:
    """Real F1 and both nulls for one clip; None if not evaluable."""
    if not pred or not ref:
        return None
    real = boundary_f1(pred, ref, tolerance)["f1"]
    chance = chance_boundary_f1(len(pred), len(ref), n_frames, tolerance, trials, seed)
    shift = shift_null_f1(pred, ref, n_frames, tolerance, trials, seed)
    return {"real": real, "chance": chance, "shift": shift}


def summarize(records: List[Dict]) -> Dict[str, float]:
    """Mean real/chance/shift F1 over evaluable clips, with ratios."""
    if not records:
        return {"n_clips": 0, "real": 0.0, "chance": 0.0, "shift": 0.0,
                "ratio_chance": 0.0, "ratio_shift": 0.0}
    real = float(np.mean([r["real"] for r in records]))
    chance = float(np.mean([r["chance"] for r in records]))
    shift = float(np.mean([r["shift"] for r in records]))
    return {
        "n_clips": len(records), "real": real, "chance": chance, "shift": shift,
        "ratio_chance": real / chance if chance > 0 else float("inf"),
        "ratio_shift": real / shift if shift > 0 else float("inf"),
    }


def _feats_for(features, lengths, pack_idx, normalize, upper_body):
    t = int(lengths[pack_idx])
    raw = np.asarray(features[pack_idx, :t])
    return prepare_features(raw, normalize=normalize, upper_body=upper_body)


def build_records(align_df, man, features, lengths, cfg, *, tolerance, min_distance,
                  trials, seed, smooth_window=5, quantile=0.5, progress_every=0):
    """Per-clip agreement records for {anchors, all} x {minima, maxima}.

    Velocity is smoothed and minima are restricted to the deepest `quantile` of
    the clip, so the changepoint reference does not tile the timeline (a dense
    reference saturates the agreement metric — see docs/logs/ALIGNMENT_VALIDATION).
    """
    rec_to_pack = dict(zip(man["recording"], man["pack_idx"]))
    normalize, upper_body = cfg["data"]["normalize"], cfg["data"]["upper_body"]
    out = {k: [] for k in ("anchor_min", "all_min", "anchor_max", "all_max")}
    span_widths, omitted, labels_per_clip, frames_per_clip = [], [], [], []
    n_changepoints = []

    groups = list(align_df.groupby("row", sort=False))
    for gi, (rec, df_clip) in enumerate(groups):
        pack_idx = int(rec_to_pack[rec])
        feats = _feats_for(features, lengths, pack_idx, normalize, upper_body)
        n_frames = feats.shape[0]
        vel = smooth_velocity(frame_velocity(feats), window=smooth_window)
        # Rests = deepest minima; peaks = shallowest maxima (deepest minima of -v).
        minima = changepoints(vel, min_distance=min_distance, quantile=quantile)
        maxima = changepoints(-vel, min_distance=min_distance, quantile=quantile)
        n_changepoints.append(len(minima))

        all_p = pred_boundaries(df_clip)
        anchor_p = pred_boundaries(df_clip, ANCHOR_MIN_CONF)

        for tag, pred, ref in (
            ("anchor_min", anchor_p, minima), ("all_min", all_p, minima),
            ("anchor_max", anchor_p, maxima), ("all_max", all_p, maxima),
        ):
            r = _agreement_record(pred, ref, n_frames, tolerance, trials, seed)
            if r is not None:
                out[tag].append(r)

        w = (df_clip["end_frame"] - df_clip["start_frame"] + 1)
        span_widths.extend(int(x) for x in w)
        labels_per_clip.append(len(df_clip))
        frames_per_clip.append(n_frames)
        # A label "received zero frames" would be dropped by path_to_spans; the
        # target length minus aligned label count estimates omissions.
        omitted.append(max(0, int(df_clip["label_index"].max()) + 1 - len(df_clip)))
        if progress_every and (gi + 1) % progress_every == 0:
            print(f"  ...{gi + 1}/{len(groups)} clips", flush=True)

    diagnostics = {
        "mean_span_width": float(np.mean(span_widths)) if span_widths else 0.0,
        "mean_labels_per_clip": float(np.mean(labels_per_clip)) if labels_per_clip else 0.0,
        "mean_changepoints_per_clip": float(np.mean(n_changepoints)) if n_changepoints else 0.0,
        "omitted_label_rate": float(np.sum(omitted) / max(np.sum(labels_per_clip), 1)),
        "length_label_corr": float(np.corrcoef(frames_per_clip, labels_per_clip)[0, 1])
        if len(frames_per_clip) > 1 else 0.0,
    }
    return out, diagnostics


def _verdict(anchor_min: Dict, all_min: Dict, anchor_max: Dict) -> str:
    best = max(anchor_min["ratio_shift"], anchor_max["ratio_shift"])
    if best >= 1.3 and (anchor_min["real"] > anchor_min["shift"]
                        or anchor_max["real"] > anchor_max["shift"]):
        return ("ABOVE CHANCE — CTC anchor placements agree with motion "
                "changepoints beyond both nulls. Alignments carry real temporal "
                "signal; WS-E is viable and this corroborates the encoder-grounding "
                "claim. Phase C (attention cross-check) is unlocked.")
    if best >= 1.1:
        return ("WEAKLY ABOVE CHANCE — a small but consistent margin over the "
                "nulls. Matches the pre-registered prediction. Report as a modest "
                "positive; do not overclaim, and gate Phase C on the attention "
                "cross-check agreeing.")
    return ("AT CHANCE — CTC subword placements do not beat the shift null against "
            "velocity minima. This is a negative result for the specific hypothesis "
            "that CTC labels localize to sign-boundary rests; it does NOT overturn "
            "the grounding evidence from the input ablation (mismatch −4.71 chrF++) "
            "or A0 (the head is temporally responsive). The likely reconciliation is "
            "granularity — subwords are not signs — and/or that anchors fall away "
            "from motion extrema (note anchors actively avoid velocity peaks). Log to "
            "FAILURES.md; do NOT proceed to Phase C or pseudo-glosses on this signal.")


def render_markdown(run_dir, split_kind, split_source, n_test, cfg_summary,
                    summaries, diagnostics, params) -> str:
    am, alm, amx = summaries["anchor_min"], summaries["all_min"], summaries["anchor_max"]
    L = []
    L.append("# WS-E Alignment Validation")
    L.append("")
    L.append(f"Run `{Path(run_dir).name}` · split **{split_kind}** (`{split_source}`) · "
             f"test n={n_test} · {cfg_summary}")
    L.append("")
    L.append(f"Params: tolerance=±{params['tolerance']} frames, "
             f"changepoint min_distance={params['min_distance']}, "
             f"velocity smoothing window={params['smooth_window']}, "
             f"minima kept below clip quantile={params['quantile']}, "
             f"null trials={params['trials']}, seed={params['seed']}. "
             f"Anchors are labels with peak_conf ≥ {ANCHOR_MIN_CONF}.")
    L.append("")
    L.append("## Agreement vs motion changepoints (velocity minima)")
    L.append("")
    L.append("| Label set | clips | mean F1 | uniform null | shift null | F1 / shift |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for tag, label in (("anchor_min", "anchors (conf≥0.5)"), ("all_min", "all labels")):
        s = summaries[tag]
        L.append(f"| {label} | {s['n_clips']} | {s['real']:.4f} | {s['chance']:.4f} | "
                 f"{s['shift']:.4f} | {s['ratio_shift']:.2f}× |")
    L.append("")
    L.append("## Diagnostic: agreement vs velocity maxima (articulation peaks)")
    L.append("")
    L.append("| Label set | clips | mean F1 | shift null | F1 / shift |")
    L.append("|---|---:|---:|---:|---:|")
    for tag, label in (("anchor_max", "anchors (conf≥0.5)"), ("all_max", "all labels")):
        s = summaries[tag]
        L.append(f"| {label} | {s['n_clips']} | {s['real']:.4f} | {s['shift']:.4f} | "
                 f"{s['ratio_shift']:.2f}× |")
    L.append("")
    L.append("## Sanity diagnostics")
    L.append("")
    d = diagnostics
    L.append(f"- Mean span width: {d['mean_span_width']:.2f} frames "
             "(≈1 → point events, expected from the sparse head)")
    L.append(f"- Mean aligned labels/clip: {d['mean_labels_per_clip']:.1f}")
    L.append(f"- Mean motion changepoints/clip: {d['mean_changepoints_per_clip']:.1f}")
    L.append(f"- Omitted-label rate (labels receiving no frame): {d['omitted_label_rate']*100:.2f}% "
             "(should be ~0 by construction)")
    L.append(f"- Corr(clip frames, aligned labels): {d['length_label_corr']:.3f} "
             "(positive → alignment tracks content)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(_verdict(am, alm, amx))
    L.append("")
    return "\n".join(L)


def resolve_split_for(cfg, man, split_root):
    sp, src = resolve_split(man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
                            fracs=tuple(cfg["split"]["fracs"]),
                            split_root=split_root, prefer_frozen=True)
    return sp, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--alignments", default=None, help="defaults to <run>/alignments.csv")
    ap.add_argument("--split-root", default="splits")
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--min-distance", type=int, default=8)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="keep only velocity minima at/below this clip quantile")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    align_path = Path(args.alignments) if args.alignments else run_dir / "alignments.csv"
    if not align_path.exists():
        raise SystemExit(f"alignments not found: {align_path} (run align_checkpoint.py first)")

    align_df = pd.read_csv(align_path)
    data_root = Path(cfg["trial"]["out_dir"])
    man = pd.read_csv(data_root / "sample_manifest.csv")
    features = np.load(data_root / "packed.npy", mmap_mode="r")
    lengths = np.load(data_root / "lengths.npy")
    _, split_source = resolve_split_for(cfg, man, args.split_root)

    n_test = align_df["row"].nunique()
    print(f"run={run_dir} | split={cfg['split']['kind']} ({split_source}) | "
          f"clips in alignments={n_test} | trials={args.trials}")

    records, diagnostics = build_records(
        align_df, man, features, lengths, cfg,
        tolerance=args.tolerance, min_distance=args.min_distance,
        trials=args.trials, seed=args.seed,
        smooth_window=args.smooth_window, quantile=args.quantile, progress_every=400)
    summaries = {k: summarize(v) for k, v in records.items()}

    for tag in ("anchor_min", "all_min", "anchor_max", "all_max"):
        s = summaries[tag]
        print(f"{tag:11s} n={s['n_clips']:4d} real={s['real']:.4f} "
              f"chance={s['chance']:.4f} shift={s['shift']:.4f} "
              f"ratio_shift={s['ratio_shift']:.2f}")

    cfg_summary = "TConv+CTC d256"
    md = render_markdown(run_dir, cfg["split"]["kind"], split_source, n_test,
                         cfg_summary, summaries, diagnostics,
                         {"tolerance": args.tolerance, "min_distance": args.min_distance,
                          "trials": args.trials, "seed": args.seed,
                          "smooth_window": args.smooth_window, "quantile": args.quantile})
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"saved report -> {args.out}")
    else:
        print("\n" + md)


if __name__ == "__main__":
    main()
