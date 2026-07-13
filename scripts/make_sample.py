"""Select a stratified sample of clips for a trial run (plan §3, trial tier).

Samples `frac` of clips from each chunk (deterministic) so every chunk/session
is represented and the split logic behaves like it will on the full dataset.
Writes `<out_dir>/sample_manifest.csv` with a normalized `clean_text` column.

    python scripts/make_sample.py                       # 5% per config default
    python scripts/make_sample.py --frac 0.05
    python scripts/make_sample.py --limit 8             # tiny smoke sample
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from bornil.benchmark.splits import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="video_data.csv")
    ap.add_argument("--out-dir", default="trial_5pct")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="hard cap on #clips (smoke test); overrides frac")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        sample = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    else:
        sample = (df.groupby("chunk_path", group_keys=False)
                    .apply(lambda g: g.sample(frac=args.frac, random_state=args.seed)))
    sample = sample.reset_index(drop=True)
    sample["clean_text"] = sample["text"].map(normalize_text)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "sample_manifest.csv")
    sample.to_csv(out, index=False)
    print(f"sampled {len(sample)}/{len(df)} clips "
          f"({sample['chunk_path'].nunique()} chunks) -> {out}")


if __name__ == "__main__":
    main()
