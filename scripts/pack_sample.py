"""Pack per-clip features into one memmap (WS-A, plan §5).

Consolidates `<out_dir>/features/*.npy` into `packed.npy` (N, MAXF, 258) + a
`lengths.npy` of true frame counts, and adds a `pack_idx` column to the manifest.
Only clips whose feature file exists are kept, so it is safe to run after a
partial (resumable) extraction — re-run it any time to (re)build the pack from
whatever is on disk.

MAXF is auto-detected from the longest extracted clip (so it adapts to the
extraction frame cap: 128, 256, ...), or set explicitly with --max-frames.
Writes via a memmap + atomic rename, so a 256-frame full-dataset pack
(~4.7 GB) never has to fit in RAM.

    python scripts/pack_sample.py --out-dir data_full
    python scripts/pack_sample.py --out-dir data_full --max-frames 256
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DIM = 258


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="trial_5pct")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="pad/truncate length (default: longest extracted clip)")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.out_dir, "sample_manifest.csv"))
    feat_dir = os.path.join(args.out_dir, "features")

    def fpath(rec):
        return os.path.join(feat_dir, rec.replace(".webm", ".npy"))

    man = man[man["recording"].map(lambda r: os.path.exists(fpath(r)))].reset_index(drop=True)
    n = len(man)
    if n == 0:
        raise SystemExit("no feature files found — run extract_sample.py first")

    # Determine MAXF: explicit, or the longest clip actually on disk.
    if args.max_frames is not None:
        maxf = args.max_frames
    else:
        maxf = 0
        for rec in man["recording"]:
            maxf = max(maxf, int(np.load(fpath(rec), mmap_mode="r").shape[0]))
    print(f"packing {n} clips -> (N={n}, MAXF={maxf}, DIM={DIM}) "
          f"~{n * maxf * DIM * 4 / 1e9:.2f} GB", flush=True)

    out = os.path.join(args.out_dir, "packed.npy")
    arr = np.lib.format.open_memmap(out + ".tmp", mode="w+", dtype=np.float32,
                                    shape=(n, maxf, DIM))
    lengths = np.zeros(n, dtype=np.int32)
    for i, rec in enumerate(man["recording"]):
        a = np.load(fpath(rec))
        t = min(a.shape[0], maxf)
        arr[i, :t] = a[:t]
        lengths[i] = t
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    arr.flush()
    del arr
    os.replace(out + ".tmp", out)

    man["pack_idx"] = range(n)
    np.save(os.path.join(args.out_dir, "lengths.npy"), lengths)
    man.to_csv(os.path.join(args.out_dir, "sample_manifest.csv"), index=False)
    print(f"packed {n} clips -> {out} "
          f"(lengths min={int(lengths.min())} max={int(lengths.max())})", flush=True)


if __name__ == "__main__":
    main()
