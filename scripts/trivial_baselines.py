"""Compute the trivial baseline floors for a run's split (plan §6.4).

Prints most-frequent, random, and nearest-neighbour-retrieval scores on the SAME
train/test split the model used, so the model's number can be read in context: a
model that barely beats these hasn't learned to translate.

    python scripts/trivial_baselines.py --out-dir data_full --config configs/full.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from bornil.config import load_config
from bornil.benchmark import metrics as M
from bornil.benchmark import trivial
from bornil.benchmark.splits import resolve_split
from bornil.data.transforms import prepare_features


def _pool(features, lengths, pack_idx, normalize, upper_body):
    """Mean-pool each clip's (normalized) keypoints -> one vector per clip."""
    out = []
    for p in pack_idx:
        p = int(p)
        x = np.asarray(features[p, :int(lengths[p])])
        x = prepare_features(x, normalize=normalize, upper_body=upper_body)
        out.append(x.mean(axis=0))
    return np.stack(out) if out else np.zeros((0, 1), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_full")
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--split", default=None,
                    help="override split.kind: random-v1|signerindep-v1|unseen-sentence-v1")
    ap.add_argument("--split-root", default="splits",
                    help="directory containing frozen split files")
    ap.add_argument("--no-frozen-splits", action="store_true",
                    help="generate the split from the current manifest instead of loading splits/<kind>")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.split is not None:
        cfg.setdefault("split", {})["kind"] = args.split

    man = pd.read_csv(os.path.join(args.out_dir, "sample_manifest.csv"))
    features = np.load(os.path.join(args.out_dir, "packed.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(args.out_dir, "lengths.npy"))

    sp, split_source = resolve_split(
        man,
        kind=cfg["split"]["kind"],
        seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]),
        split_root=args.split_root,
        prefer_frozen=not args.no_frozen_splits,
    )
    rec2text = dict(zip(man["recording"], man["clean_text"].astype(str)))
    rec2pack = dict(zip(man["recording"], man["pack_idx"]))

    train_texts = [rec2text[r] for r in sp["train"]]
    test_refs = [rec2text[r] for r in sp["test"]]
    print(f"split={cfg['split']['kind']} ({split_source})")
    print(f"train {len(train_texts)} / test {len(test_refs)} clips")

    def report(name, hyps):
        m = M.corpus_metrics(hyps, test_refs)
        print(f"  {name:12s}  BLEU-4={m['bleu4']:5.2f}  chrF={m['chrf']:5.2f}  "
              f"chrF++={m['chrf++']:5.2f}  ROUGE-L={m['rougeL']:.3f}")
        return m

    print("=== trivial baselines (test) ===")
    report("most-freq", trivial.most_frequent_hypotheses(train_texts, len(test_refs)))
    report("random", trivial.random_hypotheses(train_texts, len(test_refs),
                                               seed=cfg["trial"]["seed"]))

    print("  computing retrieval (pooling features)...", flush=True)
    norm, upper = cfg["data"]["normalize"], cfg["data"]["upper_body"]
    tr_feats = _pool(features, lengths, [rec2pack[r] for r in sp["train"]], norm, upper)
    te_feats = _pool(features, lengths, [rec2pack[r] for r in sp["test"]], norm, upper)
    report("retrieval", trivial.retrieval_hypotheses(tr_feats, train_texts, te_feats))


if __name__ == "__main__":
    main()
