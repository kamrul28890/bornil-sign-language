"""Official Bornil benchmark splits (WS-B, plan §6.2).

Three versioned splits, all derived deterministically from the manifest
(`video_data.csv`) so they can be regenerated and released as UUID lists:

  * random-v1          — stratified-free random 80/10/10 over clips.
  * signerindep-v1     — whole `chunk_path` (our session proxy) kept on one
                         side, so eval sessions are unseen.
  * unseen-sentence-v1 — whole normalized-sentence groups kept on one side, so
                         no sentence appears in both train and eval.

The only correctness guarantee that matters is **no leakage** — enforced by
`check_no_leakage` and covered by tests. Absolute fractions are approximate for
the grouped splits (whole groups are indivisible).
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Split = Dict[str, List[str]]
_NAMES = ("train", "val", "test")
_KINDS = ("random-v1", "signerindep-v1", "unseen-sentence-v1")


class LeakageError(AssertionError):
    """Raised when a split violates a disjointness invariant."""


def normalize_text(text) -> str:
    """Whitespace-collapse and NFC-normalize a Bangla label.

    Matches the manifest's `clean_text` (collapse runs of whitespace) and adds
    Unicode NFC normalization so visually identical Bangla strings compare equal
    (critical for the unseen-sentence split and for metric scoring).
    """
    s = unicodedata.normalize("NFC", str(text))
    return " ".join(s.strip().split())


def _targets(n: int, fracs: Sequence[float]) -> Dict[str, float]:
    return {name: n * f for name, f in zip(_NAMES, fracs)}


def make_random_split(df: pd.DataFrame, seed: int = 42,
                      fracs: Sequence[float] = (0.8, 0.1, 0.1)) -> Split:
    """Random 80/10/10 over individual clips (recordings)."""
    recs = df["recording"].tolist()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(recs))
    n_val = max(1, int(round(fracs[1] * len(recs))))
    n_test = max(1, int(round(fracs[2] * len(recs))))
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return {
        "train": [recs[i] for i in train_idx],
        "val": [recs[i] for i in val_idx],
        "test": [recs[i] for i in test_idx],
    }


def _group_split(df: pd.DataFrame, group_col_values: pd.Series, seed: int,
                 fracs: Sequence[float]) -> Split:
    """Assign whole groups to partitions, greedily balancing clip counts.

    Groups are shuffled (seeded) then processed largest-first; each group goes to
    the partition with the largest remaining deficit (target minus current). This
    keeps whole groups intact while landing near the requested fractions.
    """
    df = df.copy()
    df["_grp"] = group_col_values.values
    sizes = df.groupby("_grp").size().to_dict()

    rng = np.random.default_rng(seed)
    groups = list(sizes.keys())
    rng.shuffle(groups)
    groups.sort(key=lambda g: sizes[g], reverse=True)

    targets = _targets(len(df), fracs)
    current = {name: 0 for name in _NAMES}
    assign: Dict[str, str] = {}
    for g in groups:
        name = max(_NAMES, key=lambda nm: targets[nm] - current[nm])
        assign[g] = name
        current[name] += sizes[g]

    out: Split = {name: [] for name in _NAMES}
    for rec, g in zip(df["recording"], df["_grp"]):
        out[assign[g]].append(rec)
    return out


def make_signerindep_split(df: pd.DataFrame, seed: int = 42,
                           fracs: Sequence[float] = (0.8, 0.1, 0.1)) -> Split:
    """Signer-independent split: whole `chunk_path` groups stay on one side."""
    return _group_split(df, df["chunk_path"], seed, fracs)


def make_unseen_sentence_split(df: pd.DataFrame, seed: int = 42,
                               fracs: Sequence[float] = (0.8, 0.1, 0.1)) -> Split:
    """Unseen-sentence split: whole normalized-sentence groups stay on one side."""
    norm = df["text"].map(normalize_text)
    return _group_split(df, norm, seed, fracs)


def make_split(df: pd.DataFrame, kind: str, seed: int = 42,
               fracs: Sequence[float] = (0.8, 0.1, 0.1)) -> Split:
    """Create a split by benchmark kind."""
    if kind == "random-v1":
        return make_random_split(df, seed=seed, fracs=fracs)
    if kind == "signerindep-v1":
        return make_signerindep_split(df, seed=seed, fracs=fracs)
    if kind == "unseen-sentence-v1":
        return make_unseen_sentence_split(df, seed=seed, fracs=fracs)
    raise ValueError(f"unknown split kind {kind!r}; expected one of {_KINDS}")


def _recording_set(split: Split) -> set:
    recs = set()
    for name in _NAMES:
        recs.update(split.get(name, []))
    return recs


def filter_split_by_recordings(split: Split, allowed_recordings) -> Split:
    """Keep only allowed recordings while preserving existing partition membership.

    This is for controlled filtered-data experiments: resolve the official split
    first, then remove rows inside each partition. That keeps train/val/test
    provenance fixed instead of creating a new split over the filtered manifest.
    """
    allowed = set(allowed_recordings)
    return {
        name: [rec for rec in split.get(name, []) if rec in allowed]
        for name in _NAMES
    }


def filter_split_by_fraction(split: Split, fraction: float, seed: int = 42) -> Split:
    """Randomly subsample the train partition to `fraction` of its clips.

    val/test are returned unchanged (as new lists, same contents) so every
    point on a data-efficiency curve evaluates against the same held-out
    data. Unlike `filter_split_by_recordings`, this filters by a random
    fraction, not by a predicate on the manifest.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    train = list(split.get("train", []))
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = max(1, int(round(fraction * len(train))))
        keep_idx = sorted(rng.permutation(len(train))[:n].tolist())
        train = [train[i] for i in keep_idx]
    return {
        "train": train,
        "val": list(split.get("val", [])),
        "test": list(split.get("test", [])),
    }


def resolve_split(df: pd.DataFrame, kind: str, seed: int = 42,
                  fracs: Sequence[float] = (0.8, 0.1, 0.1),
                  split_root="splits",
                  prefer_frozen: bool = True) -> Tuple[Split, str]:
    """Load the frozen split when it exactly matches `df`; otherwise generate.

    Full benchmark runs should use the released files in `splits/<kind>/`.
    Trial/sample manifests only contain a subset of recordings, so they fall back
    to a deterministic local split instead of accidentally mixing in full-corpus
    recording ids.

    Returns `(split, source)` where source is `frozen:<path>` or `generated`.
    """
    if prefer_frozen:
        split_dir = Path(split_root) / kind
        if all((split_dir / f"{name}.txt").exists() for name in _NAMES):
            sp = load_split(split_dir)
            if _recording_set(sp) == set(df["recording"]):
                return sp, f"frozen:{split_dir.as_posix()}"
    return make_split(df, kind=kind, seed=seed, fracs=fracs), "generated"


def check_no_leakage(split: Split, df: Optional[pd.DataFrame] = None,
                     by: Optional[str] = None) -> None:
    """Verify partitions are disjoint by recording, and optionally by a column.

    Always checks that no recording appears in two partitions. If `df` and `by`
    are given (e.g. by='chunk_path'), also checks that the grouping value never
    straddles partitions. Raises LeakageError on any violation.
    """
    sets = {name: set(split.get(name, [])) for name in _NAMES}
    for a in _NAMES:
        for b in _NAMES:
            if a < b and sets[a] & sets[b]:
                raise LeakageError(
                    f"{len(sets[a] & sets[b])} recording(s) shared between "
                    f"{a} and {b}: {sorted(sets[a] & sets[b])[:3]}..."
                )
    if df is not None and by is not None:
        rec2grp = dict(zip(df["recording"], df[by]))
        grp_sets = {name: {rec2grp[r] for r in recs if r in rec2grp}
                    for name, recs in split.items()}
        for a in _NAMES:
            for b in _NAMES:
                if a < b and grp_sets[a] & grp_sets[b]:
                    raise LeakageError(
                        f"'{by}' value(s) straddle {a}/{b}: "
                        f"{sorted(grp_sets[a] & grp_sets[b])[:3]}..."
                    )


def write_split(split: Split, out_dir, meta: Optional[dict] = None) -> None:
    """Write a split as one recording-id-per-line file per partition + meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in _NAMES:
        recs = split.get(name, [])
        (out / f"{name}.txt").write_text("\n".join(recs) + ("\n" if recs else ""),
                                         encoding="utf-8")
    if meta is not None:
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")


def load_split(split_dir) -> Split:
    """Load a split written by `write_split`."""
    d = Path(split_dir)
    out: Split = {}
    for name in _NAMES:
        text = (d / f"{name}.txt").read_text(encoding="utf-8")
        out[name] = [ln for ln in text.splitlines() if ln]
    return out


def split_stats(split: Split, df: pd.DataFrame) -> Dict[str, dict]:
    """Per-partition counts: clips, distinct chunks, distinct normalized texts."""
    rec2chunk = dict(zip(df["recording"], df["chunk_path"]))
    rec2text = {r: normalize_text(t) for r, t in zip(df["recording"], df["text"])}
    stats = {}
    for name in _NAMES:
        recs = split.get(name, [])
        stats[name] = {
            "clips": len(recs),
            "chunks": len({rec2chunk[r] for r in recs if r in rec2chunk}),
            "unique_texts": len({rec2text[r] for r in recs if r in rec2text}),
        }
    return stats
