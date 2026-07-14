"""Flag clips with abnormal frame counts (data-quality audit, WS-A).

Uses two independent signals so we catch both kinds of bad clip:
  * absolute frame_count (very long clips), and
  * implied FPS = frame_count / duration (normal video is ~25-30; a huge value
    means the frame_count is bogus / the clip is concatenated).

Writes a sorted report to docs/logs/frame_outliers.csv and prints the worst.

    python scripts/check_frame_outliers.py
    python scripts/check_frame_outliers.py --max-fps 60 --min-frames 1500
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def _duration(meta):
    try:
        return float(json.loads(meta).get("duration", 0) or 0)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="video_data.csv")
    ap.add_argument("--max-fps", type=float, default=60.0,
                    help="implied fps above this is flagged as bogus")
    ap.add_argument("--min-frames", type=int, default=1500,
                    help="frame_count above this is flagged as very long")
    ap.add_argument("--out", default="docs/logs/frame_outliers.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df["duration"] = df["metadata"].map(_duration)
    df["implied_fps"] = df.apply(
        lambda r: (r["frame_count"] / r["duration"]) if r["duration"] > 0 else float("inf"),
        axis=1)
    # True fps from the clips whose implied fps is physically plausible.
    sane = df[(df["implied_fps"] > 10) & (df["implied_fps"] < 60)]
    true_fps = float(sane["implied_fps"].median())
    # Real frame estimate: trust frame_count when its fps is sane, else duration*true_fps.
    ok = (df["implied_fps"] >= 10) & (df["implied_fps"] <= 60)
    df["real_frames_est"] = ok.where(ok, other=False)
    df["real_frames_est"] = df["frame_count"].where(ok, (df["duration"] * true_fps)).round().astype(int)
    df["reason"] = ok.map({True: "", False: "fps_artifact(frame_count_unreliable)"})
    long_mask = df["real_frames_est"] > args.min_frames
    df.loc[long_mask & (df["reason"] == ""), "reason"] = "long_clip"
    df.loc[long_mask & (df["reason"] != ""), "reason"] += "+long_clip"
    df["path"] = df["chunk_path"].str.replace("/", os.sep, regex=False) + os.sep + df["recording"]

    flagged = df[(df["implied_fps"] > args.max_fps) | (long_mask)].copy()
    flagged = flagged.sort_values("real_frames_est", ascending=False)

    cols = ["path", "frame_count", "duration", "implied_fps", "real_frames_est", "reason"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    flagged[cols].to_csv(args.out, index=False, encoding="utf-8")

    print(f"total clips              : {len(df)}")
    print(f"true fps (median, sane)  : {true_fps:.2f}")
    print(f"flagged                  : {len(flagged)}")
    print(f"  fps artifact (bogus frame_count) : {(df['implied_fps'] > args.max_fps).sum()}")
    print(f"  genuinely long (>{args.min_frames} real frames) : {long_mask.sum()}")
    print(f"report -> {args.out}")
    print("\n=== 15 longest by REAL (estimated) frames ===")
    print(f"{'real~':>6} {'raw_fc':>7} {'dur(s)':>7} {'impl.fps':>9}  path")
    for r in flagged.head(15).itertuples(index=False):
        print(f"{r.real_frames_est:6d} {r.frame_count:7d} {r.duration:7.1f} {r.implied_fps:9.1f}  {r.path}")


if __name__ == "__main__":
    main()
