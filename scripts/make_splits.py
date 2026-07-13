"""Generate the official Bornil benchmark splits from the manifest.

Reads `video_data.csv`, writes UUID-list split files under `splits/<name>/`, and
prints per-partition statistics after verifying no leakage. Regenerable and
deterministic (fixed seeds), so the split files are a released, citeable artifact.

    python scripts/make_splits.py                 # uses ./video_data.csv
    python scripts/make_splits.py --manifest path/to/video_data.csv --seed 42
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from bornil.benchmark import splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="video_data.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    n = len(df)
    print(f"manifest: {n} rows | chunks: {df['chunk_path'].nunique()} | "
          f"unique texts: {df['text'].map(splits.normalize_text).nunique()}")

    builders = {
        "random-v1": splits.make_random_split,
        "signerindep-v1": splits.make_signerindep_split,
        "unseen-sentence-v1": splits.make_unseen_sentence_split,
    }
    checks = {"signerindep-v1": "chunk_path"}

    for name, build in builders.items():
        sp = build(df, seed=args.seed, fracs=(0.8, 0.1, 0.1))
        splits.check_no_leakage(sp, df=df, by=checks.get(name))
        stats = splits.split_stats(sp, df)
        meta = {"name": name, "seed": args.seed, "manifest_rows": n,
                "fracs": [0.8, 0.1, 0.1], "stats": stats}
        splits.write_split(sp, Path(args.out) / name, meta=meta)
        print(f"\n[{name}]  (leakage check: OK)")
        for part, s in stats.items():
            print(f"  {part:5s}: clips={s['clips']:6d}  chunks={s['chunks']:3d}  "
                  f"unique_texts={s['unique_texts']:6d}")
        print(f"  -> {Path(args.out) / name}")

    print("\nAll official splits written and leakage-checked.")


if __name__ == "__main__":
    main()
