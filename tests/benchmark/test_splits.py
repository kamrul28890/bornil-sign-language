"""Tests for the official Bornil benchmark split generator.

Correctness here means: no data leakage. Every recording appears in exactly one
partition; the signer-independent split never lets a chunk (our session proxy)
straddle partitions; the unseen-sentence split never lets a sentence appear in
both train and eval. These invariants are what make reported numbers trustworthy.
"""
import pandas as pd
import pytest

from bornil.benchmark import splits


def _toy_manifest():
    """A tiny manifest with repeated sentences and several chunks/sessions."""
    rows = []
    # 10 chunks, chunk k has (k+1) clips -> 55 clips total.
    rec = 0
    for k in range(10):
        for _ in range(k + 1):
            rows.append(
                {
                    "text": f"sentence {rec % 7}",   # 7 distinct texts, so many repeat
                    "recording": f"{rec:04d}.webm",
                    "chunk_path": f"chunks/chunk_{k}",
                    "frame_count": 100 + rec,
                }
            )
            rec += 1
    return pd.DataFrame(rows)


# --- normalize_text ---------------------------------------------------------

def test_normalize_text_collapses_whitespace():
    assert splits.normalize_text("  আমি   ভালো\tআছি \n") == "আমি ভালো আছি"


def test_normalize_text_is_nfc():
    import unicodedata
    # A decomposed sequence must come back composed (NFC).
    decomposed = unicodedata.normalize("NFD", "নিয়ে")
    assert splits.normalize_text(decomposed) == unicodedata.normalize("NFC", "নিয়ে")


# --- random split -----------------------------------------------------------

def test_random_split_covers_all_exactly_once():
    df = _toy_manifest()
    sp = splits.make_random_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    all_recs = sorted(sp["train"] + sp["val"] + sp["test"])
    assert all_recs == sorted(df["recording"].tolist())
    # disjoint
    assert set(sp["train"]) & set(sp["val"]) == set()
    assert set(sp["train"]) & set(sp["test"]) == set()
    assert set(sp["val"]) & set(sp["test"]) == set()


def test_random_split_is_deterministic():
    df = _toy_manifest()
    a = splits.make_random_split(df, seed=7, fracs=(0.8, 0.1, 0.1))
    b = splits.make_random_split(df, seed=7, fracs=(0.8, 0.1, 0.1))
    assert a == b


def test_random_split_roughly_respects_fractions():
    df = _toy_manifest()
    sp = splits.make_random_split(df, seed=1, fracs=(0.8, 0.1, 0.1))
    n = len(df)
    assert abs(len(sp["train"]) / n - 0.8) < 0.05


# --- signer-independent split (chunk-as-session) ----------------------------

def test_signerindep_chunks_are_disjoint():
    df = _toy_manifest()
    sp = splits.make_signerindep_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    rec2chunk = dict(zip(df["recording"], df["chunk_path"]))
    chunks = {name: {rec2chunk[r] for r in recs} for name, recs in sp.items()}
    assert chunks["train"] & chunks["val"] == set()
    assert chunks["train"] & chunks["test"] == set()
    assert chunks["val"] & chunks["test"] == set()


def test_signerindep_covers_all_and_nonempty():
    df = _toy_manifest()
    sp = splits.make_signerindep_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    all_recs = sorted(sp["train"] + sp["val"] + sp["test"])
    assert all_recs == sorted(df["recording"].tolist())
    assert all(len(sp[k]) > 0 for k in ("train", "val", "test"))


# --- unseen-sentence split --------------------------------------------------

def test_unseen_sentence_has_no_text_overlap():
    df = _toy_manifest()
    sp = splits.make_unseen_sentence_split(df, seed=42, fracs=(0.6, 0.2, 0.2))
    rec2text = {r: splits.normalize_text(t) for r, t in zip(df["recording"], df["text"])}
    texts = {name: {rec2text[r] for r in recs} for name, recs in sp.items()}
    assert texts["train"] & texts["val"] == set()
    assert texts["train"] & texts["test"] == set()
    assert texts["val"] & texts["test"] == set()


def test_unseen_sentence_covers_all_exactly_once():
    df = _toy_manifest()
    sp = splits.make_unseen_sentence_split(df, seed=3, fracs=(0.6, 0.2, 0.2))
    all_recs = sorted(sp["train"] + sp["val"] + sp["test"])
    assert all_recs == sorted(df["recording"].tolist())


# --- leakage checker --------------------------------------------------------

def test_check_no_leakage_passes_on_clean_split():
    df = _toy_manifest()
    sp = splits.make_random_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    splits.check_no_leakage(sp)  # should not raise


def test_check_no_leakage_raises_on_overlap():
    bad = {"train": ["a.webm", "b.webm"], "val": ["b.webm"], "test": ["c.webm"]}
    with pytest.raises(splits.LeakageError):
        splits.check_no_leakage(bad)


def test_check_no_leakage_detects_chunk_straddle():
    df = _toy_manifest()
    # Force a bad split: put one chunk's clips in both train and test.
    chunk0 = df[df["chunk_path"] == "chunks/chunk_0"]["recording"].tolist()
    chunk1 = df[df["chunk_path"] == "chunks/chunk_1"]["recording"].tolist()
    bad = {"train": chunk1[:1] + chunk0[:1], "val": [], "test": chunk1[1:]}
    with pytest.raises(splits.LeakageError):
        splits.check_no_leakage(bad, df=df, by="chunk_path")


# --- split file I/O ---------------------------------------------------------

def test_write_then_load_split_roundtrips(tmp_path):
    df = _toy_manifest()
    sp = splits.make_random_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    splits.write_split(sp, tmp_path / "random-v1", meta={"seed": 42})
    loaded = splits.load_split(tmp_path / "random-v1")
    assert loaded == sp


def test_resolve_split_loads_matching_frozen_split(tmp_path):
    df = _toy_manifest()
    sp = splits.make_random_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    splits.write_split(sp, tmp_path / "random-v1", meta={"seed": 42})

    loaded, source = splits.resolve_split(
        df, "random-v1", seed=99, split_root=tmp_path, prefer_frozen=True
    )

    assert loaded == sp
    assert source.startswith("frozen:")


def test_resolve_split_falls_back_when_frozen_split_does_not_match_manifest(tmp_path):
    df = _toy_manifest()
    full = splits.make_random_split(df, seed=42, fracs=(0.8, 0.1, 0.1))
    splits.write_split(full, tmp_path / "random-v1", meta={"seed": 42})
    sample = df.head(12).copy()

    resolved, source = splits.resolve_split(
        sample, "random-v1", seed=42, split_root=tmp_path, prefer_frozen=True
    )

    assert source == "generated"
    assert set(resolved["train"] + resolved["val"] + resolved["test"]) == set(sample["recording"])


def test_filter_split_by_recordings_preserves_partition_membership():
    split = {
        "train": ["a.webm", "b.webm", "c.webm"],
        "val": ["d.webm", "e.webm"],
        "test": ["f.webm", "g.webm"],
    }

    filtered = splits.filter_split_by_recordings(split, {"b.webm", "d.webm", "g.webm"})

    assert filtered == {
        "train": ["b.webm"],
        "val": ["d.webm"],
        "test": ["g.webm"],
    }
