"""Pack the trial sample's per-clip features into one memmap (WS-A, plan §5).

Consolidates `<out_dir>/features/*.npy` into `packed.npy` (N,128,258) + a
`lengths.npy` of true frame counts, and adds a `pack_idx` column to the sample
manifest. Only clips whose feature file exists are kept (so it works after a
partial extraction). Mirrors pack_features.py for the full set.
"""
import argparse
import os

import numpy as np
import pandas as pd

MAXF, DIM = 128, 258


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="trial_5pct")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.out_dir, "sample_manifest.csv"))
    feat_dir = os.path.join(args.out_dir, "features")

    def fpath(rec):
        return os.path.join(feat_dir, rec.replace(".webm", ".npy"))

    man = man[man["recording"].map(lambda r: os.path.exists(fpath(r)))].reset_index(drop=True)
    n = len(man)
    if n == 0:
        raise SystemExit("no feature files found — run extract_sample.py first")

    arr = np.zeros((n, MAXF, DIM), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    for i, rec in enumerate(man["recording"]):
        a = np.load(fpath(rec))
        t = min(a.shape[0], MAXF)
        arr[i, :t] = a[:t]
        lengths[i] = t

    man["pack_idx"] = range(n)
    np.save(os.path.join(args.out_dir, "packed.npy"), arr)
    np.save(os.path.join(args.out_dir, "lengths.npy"), lengths)
    man.to_csv(os.path.join(args.out_dir, "sample_manifest.csv"), index=False)
    print(f"packed {n} clips -> {args.out_dir}/packed.npy "
          f"(lengths min={int(lengths.min())} max={int(lengths.max())})")


if __name__ == "__main__":
    main()
