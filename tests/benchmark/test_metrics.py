"""Tests for the locked metric protocol (plan §6.3).

The protocol must be reproducible and Bangla-appropriate: NFC normalization on
both sides, a declared BLEU tokenizer (`intl`, handles non-Latin script without
external downloads), chrF/chrF++ (character-level, robust for Bangla morphology),
ROUGE-L, and bootstrap confidence intervals. BERTScore is optional (needs torch).
"""
import unicodedata

import pytest

from bornil.benchmark import metrics


REF = ["আমি বাংলায় গান গাই", "সে স্কুলে যায়"]


def test_nfc_normalizes():
    decomposed = unicodedata.normalize("NFD", "নিয়ে")
    assert metrics.nfc(decomposed) == unicodedata.normalize("NFC", "নিয়ে")


def test_identical_hypothesis_scores_near_perfect():
    m = metrics.corpus_metrics(REF, REF)
    assert m["bleu4"] == pytest.approx(100.0, abs=1e-6)
    assert m["chrf"] == pytest.approx(100.0, abs=1e-6)
    assert m["chrf++"] == pytest.approx(100.0, abs=1e-6)
    assert m["rougeL"] == pytest.approx(1.0, abs=1e-6)


def test_all_metric_keys_present():
    m = metrics.corpus_metrics(REF, REF)
    for k in ("bleu1", "bleu2", "bleu3", "bleu4", "chrf", "chrf++", "rougeL"):
        assert k in m


def test_worse_hypothesis_scores_lower_bleu():
    good = list(REF)
    bad = ["সম্পূর্ণ ভুল বাক্য এখানে", "আরেকটি ভুল"]
    assert metrics.corpus_metrics(bad, REF)["bleu4"] < metrics.corpus_metrics(good, REF)["bleu4"]


def test_metrics_are_nfc_invariant():
    # Decomposed hypotheses must score the same as composed (NFC) references.
    decomposed = [unicodedata.normalize("NFD", r) for r in REF]
    m = metrics.corpus_metrics(decomposed, REF)
    assert m["chrf"] == pytest.approx(100.0, abs=1e-6)


# --- ROUGE-L (pure-python LCS) ----------------------------------------------

def test_rouge_l_full_match_is_one():
    assert metrics.rouge_l("a b c d", "a b c d") == pytest.approx(1.0)


def test_rouge_l_partial_lcs():
    # LCS of "a b c" and "a x c" is "a c" (len 2); P=R=2/3 -> F=2/3.
    assert metrics.rouge_l("a b c", "a x c") == pytest.approx(2 / 3, abs=1e-6)


def test_rouge_l_no_overlap_is_zero():
    assert metrics.rouge_l("a b c", "x y z") == pytest.approx(0.0)


# --- bootstrap CI -----------------------------------------------------------

def test_bootstrap_ci_brackets_point_estimate():
    hyps = ["আমি বাংলায় গান গাই", "সে স্কুলে যায় না"]
    point = metrics.corpus_metrics(hyps, REF)["bleu4"]
    lo, hi = metrics.bootstrap_ci(hyps, REF, metric="bleu4", n=200, seed=0)
    assert lo <= point <= hi
    assert lo <= hi
