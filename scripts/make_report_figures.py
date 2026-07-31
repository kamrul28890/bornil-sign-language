"""Generate EDA figures + a stats summary for docs/report/PROJECT_REPORT.md.

Pure numpy/pandas/matplotlib -- no torch, so the torch-before-pandas import
order rule (FAIL-002) does not apply here (this script never imports torch).

    python scripts/make_report_figures.py

Writes PNGs to docs/report/assets/ and a JSON stats dump to
docs/report/assets/eda_stats.json (so report prose can cite exact numbers
without re-deriving them by hand).
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force UTF-8 stdout so Bangla text in prints/errors never trips the
# Windows console's cp1252 default codec.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT_DIR = os.path.join("docs", "report", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# Dataviz skill reference palette (light mode), applied per the skill's
# "assign color by the job it does" rule: single-series histograms use the
# sequential blue hue; the one multi-category chart (session diversity, if
# faceted) would use the categorical order -- not needed here since every
# EDA chart in this script is a single series.
BLUE = "#2a78d6"
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    stats = {}

    man = pd.read_csv("video_data.csv")
    stats["n_clips"] = int(len(man))

    # --- sentence length distribution (whitespace tokens) ---
    tok_len = man["text"].astype(str).str.split().map(len)
    stats["sentence_len"] = {
        "mean": float(tok_len.mean()), "std": float(tok_len.std()),
        "min": int(tok_len.min()), "median": float(tok_len.median()),
        "p90": float(tok_len.quantile(0.90)), "p95": float(tok_len.quantile(0.95)),
        "p99": float(tok_len.quantile(0.99)), "max": int(tok_len.max()),
    }
    bins = [0, 3, 6, 9, 12, 20, 40, 10_000]
    labels = ["0-3", "4-6", "7-9", "10-12", "13-20", "21-40", "41+"]
    bucket = pd.cut(tok_len, bins=bins, labels=labels, right=True)
    counts = bucket.value_counts().reindex(labels)
    stats["sentence_len_buckets"] = {k: int(v) for k, v in counts.items()}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts.values, color=BLUE, width=0.65)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.015, f"{v:,}\n({v/len(man)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("Sentence length (whitespace tokens)")
    ax.set_ylabel("Number of clips")
    ax.set_title("Sentence length distribution (n=17,988 clips)")
    ax.set_ylim(0, max(counts.values) * 1.18)
    savefig(fig, "eda_sentence_length.png")

    # --- unique sentence count ---
    import unicodedata
    def norm(s):
        return " ".join(unicodedata.normalize("NFC", str(s)).strip().split())
    stats["unique_sentences"] = int(man["text"].map(norm).nunique())

    # --- clip duration / frame-count distribution ---
    lengths_path = os.path.join("data_full", "lengths.npy")
    if os.path.exists(lengths_path):
        lengths = np.load(lengths_path)
    else:
        lengths = man["frame_count"].to_numpy()
    stats["frame_count"] = {
        "mean": float(lengths.mean()), "median": float(np.median(lengths)),
        "min": int(lengths.min()), "max": int(lengths.max()),
        "p90": float(np.percentile(lengths, 90)), "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "pct_at_cap_512": float((lengths >= 512).mean() * 100),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(lengths, 0, 600), bins=40, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(512, color="#eb6834", linewidth=1.5, linestyle="--")
    ax.text(512, ax.get_ylim()[1] * 0.92, " 512-frame\n processing cap",
            color="#eb6834", fontsize=9, va="top")
    ax.set_xlabel("Frames per clip (clipped at 600 for display)")
    ax.set_ylabel("Number of clips")
    ax.set_title("Clip length distribution, in video frames")
    savefig(fig, "eda_frame_length.png")

    # --- vocabulary Zipf plot ---
    words = man["text"].astype(str).str.split().explode()
    freq = words.value_counts()
    stats["vocab"] = {
        "unique_words": int(len(freq)),
        "pct_hapax": float((freq == 1).mean() * 100),
        "top5": [[w, int(c)] for w, c in freq.head(5).items()],
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ranks = np.arange(1, len(freq) + 1)
    ax.plot(ranks, freq.values, color=BLUE, linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Word rank (most frequent → rarest, log scale)")
    ax.set_ylabel("Occurrences in the dataset (log scale)")
    ax.set_title(f"Word-frequency distribution: {len(freq):,} unique words, "
                 f"{(freq==1).mean()*100:.0f}% appear only once")
    savefig(fig, "eda_word_frequency.png")

    # --- session / chunk diversity ---
    per_chunk = man["chunk_path"].value_counts().sort_values(ascending=False)
    stats["sessions"] = {
        "n_chunks": int(len(per_chunk)),
        "mean_clips_per_chunk": float(per_chunk.mean()),
        "min_clips": int(per_chunk.min()), "max_clips": int(per_chunk.max()),
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(per_chunk)), per_chunk.values, color=BLUE, width=0.8)
    ax.set_xlabel(f"Recording session (chunk), sorted by clip count -- {len(per_chunk)} sessions total")
    ax.set_ylabel("Number of clips")
    ax.set_title("Clips per recording session")
    ax.set_xticks([])
    savefig(fig, "eda_session_diversity.png")

    # --- landmark visibility (pose block, subsampled clips for speed) ---
    packed_path = os.path.join("data_full", "packed.npy")
    if os.path.exists(packed_path) and os.path.exists(lengths_path):
        packed = np.load(packed_path, mmap_mode="r")
        rng = np.random.default_rng(0)
        n_total = packed.shape[0]
        sample_idx = rng.choice(n_total, size=min(1500, n_total), replace=False)
        sample_idx.sort()
        vis_sums = np.zeros(33)
        vis_counts = 0
        for i in sample_idx:
            t = int(lengths[i])
            if t <= 0:
                continue
            clip = np.asarray(packed[i, :t])          # (t, 258)
            pose = clip[:, :132].reshape(t, 33, 4)     # 33 pose landmarks x (x,y,z,vis)
            vis_sums += pose[:, :, 3].mean(axis=0)
            vis_counts += 1
        mean_vis = vis_sums / max(vis_counts, 1)
        stats["landmark_visibility"] = {
            "sampled_clips": int(vis_counts),
            "kept_landmarks_0_22_mean": float(mean_vis[:23].mean()),
            "dropped_landmarks_23_32_mean": float(mean_vis[23:].mean()),
            "per_landmark": [float(v) for v in mean_vis],
        }
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = [BLUE if i < 23 else "#eb6834" for i in range(33)]
        ax.bar(range(33), mean_vis, color=colors, width=0.7)
        ax.axhline(0.5, color=GRID, linewidth=1)
        ax.set_xlabel("MediaPipe pose landmark index (0-22 kept, 23-32 dropped)")
        ax.set_ylabel("Mean visibility score")
        ax.set_title(f"Pose landmark visibility (sampled {vis_counts:,} clips) -- "
                     f"why landmarks 23-32 are dropped")
        savefig(fig, "eda_landmark_visibility.png")
    else:
        print("WARNING: data_full/packed.npy or lengths.npy not found -- "
              "skipping landmark-visibility figure. Run this script where "
              "the full dataset is available.")

    # --- split composition ---
    split_rows = []
    for kind in ("random-v1", "signerindep-v1", "unseen-sentence-v1"):
        row = {"split": kind}
        for part in ("train", "val", "test"):
            p = os.path.join("splits", kind, f"{part}.txt")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    row[part] = sum(1 for ln in f if ln.strip())
            else:
                row[part] = None
        split_rows.append(row)
    stats["splits"] = split_rows

    with open(os.path.join(OUT_DIR, "eda_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.join(OUT_DIR, 'eda_stats.json')}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
