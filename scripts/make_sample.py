"""Select a stratified sample of clips for a trial run (plan §3, trial tier).

Samples a percentage of clips from each chunk (deterministic) so every
chunk/session is represented and the split logic behaves like it will on the
full dataset. Writes `<out_dir>/sample_manifest.csv` with a `clean_text` column.

    python scripts/make_sample.py --percent 2           # 2% -> trial_2pct/
    python scripts/make_sample.py --percent 5 --out-dir myrun
    python scripts/make_sample.py --frac 0.02           # fraction form still works
    python scripts/make_sample.py --limit 8             # tiny smoke sample
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from bornil.benchmark.splits import normalize_text
from bornil.trial import trial_dirname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="video_data.csv")
    ap.add_argument("--percent", type=float, default=None,
                    help="percentage of the dataset to sample, e.g. 2 for 2%%")
    ap.add_argument("--frac", type=float, default=None,
                    help="fraction form of --percent (0.02 == 2%%)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default derived from percent: trial_<p>pct)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="hard cap on #clips (smoke test); overrides percent/frac")
    args = ap.parse_args()

    # Resolve percent/frac/out_dir (percent wins; default 5%).
    if args.percent is not None:
        percent = args.percent
    elif args.frac is not None:
        percent = args.frac * 100
    else:
        percent = 5.0
    frac = percent / 100.0
    out_dir = args.out_dir or trial_dirname(percent)

    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        sample = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    else:
        sample = (df.groupby("chunk_path", group_keys=False)
                    .apply(lambda g: g.sample(frac=frac, random_state=args.seed)))
    sample = sample.reset_index(drop=True)
    sample["clean_text"] = sample["text"].map(normalize_text)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "sample_manifest.csv")
    sample.to_csv(out, index=False)
    print(f"sampled {len(sample)}/{len(df)} clips ({percent:g}%, "
          f"{sample['chunk_path'].nunique()} chunks) -> {out}")


if __name__ == "__main__":
    main()
