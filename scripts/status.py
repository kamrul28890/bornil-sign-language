"""Report a run's progress so a long, resumable job can be checked / resumed.

Read-only. Shows how many clips are sampled, how many keypoint features are
already extracted (i.e. how much a re-run would skip), whether the packed array
and split exist, and the estimated remaining extraction time. Use it before or
during a full-dataset run to know exactly where things stand.

    python scripts/status.py --out-dir data_full
    python scripts/status.py --out-dir data_full --rate 1.6   # clips/sec estimate
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_full")
    ap.add_argument("--rate", type=float, default=1.6,
                    help="assumed extraction clips/sec for the ETA estimate")
    args = ap.parse_args()
    d = args.out_dir

    man_path = os.path.join(d, "sample_manifest.csv")
    if not os.path.exists(man_path):
        print(f"[{d}] no sample_manifest.csv yet — run make_sample first.")
        return
    man = pd.read_csv(man_path)
    total = len(man)

    feat_dir = os.path.join(d, "features")
    done = len(glob.glob(os.path.join(feat_dir, "*.npy"))) if os.path.isdir(feat_dir) else 0
    remaining = max(0, total - done)
    pct = 100.0 * done / total if total else 0.0

    print(f"=== status: {d} ===")
    print(f"  sampled clips     : {total}")
    print(f"  extracted (.npy)  : {done}  ({pct:.1f}%)")
    print(f"  remaining         : {remaining}")
    if remaining:
        eta_h = remaining / args.rate / 3600 if args.rate else 0
        print(f"  est. time left    : {eta_h:.1f} h  (at {args.rate:.2f} clip/s)")

    packed = os.path.join(d, "packed.npy")
    if os.path.exists(packed):
        import numpy as np
        shp = np.load(packed, mmap_mode="r").shape
        print(f"  packed.npy        : yes  shape={shp}")
    else:
        print(f"  packed.npy        : no   (run pack_sample after extraction)")

    has_pack_idx = "pack_idx" in man.columns
    print(f"  manifest pack_idx : {'yes' if has_pack_idx else 'no'}")

    if remaining == 0 and total:
        print("  -> extraction COMPLETE. Next: pack_sample, then train_sample.")
    elif done:
        print("  -> extraction PARTIAL. Re-run extract_sample to resume "
              "(finished clips are skipped).")
    else:
        print("  -> not started. Run extract_sample to begin.")


if __name__ == "__main__":
    main()
