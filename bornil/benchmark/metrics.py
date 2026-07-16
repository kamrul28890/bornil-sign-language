"""Locked metric protocol for Bornil SLT (plan §6.3).

Reproducibility rules baked in:
  * NFC-normalize hypotheses and references before scoring (Bangla scripts have
    multiple byte encodings for the same glyph — a silent source of BLEU noise).
  * BLEU via SacreBLEU with a DECLARED tokenizer: `intl` (Unicode-aware, handles
    Bangla without external model downloads). We report BLEU-1..4.
  * chrF and chrF++ (character n-gram F) — primary metrics for morphologically
    rich Bangla; tokenization-independent.
  * ROUGE-L (LCS F1) computed in-house (no NLTK dependency).
  * Bootstrap confidence intervals for every headline number.
  * BERTScore(BanglaBERT) is optional and lives behind a lazy import (needs torch).

Scales follow SacreBLEU convention: BLEU and chrF in [0, 100]; ROUGE-L in [0, 1].
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

# Declared, frozen tokenizer for BLEU. Changing this changes the numbers, so it
# is a benchmark constant, not a knob.
BLEU_TOKENIZE = "intl"


def nfc(text) -> str:
    """NFC-normalize a string (applied to every hypothesis and reference)."""
    return unicodedata.normalize("NFC", str(text))


def _bleu_k(hyps: Sequence[str], refs: Sequence[str], k: int) -> float:
    scorer = BLEU(max_ngram_order=k, tokenize=BLEU_TOKENIZE, effective_order=True)
    return scorer.corpus_score(list(hyps), [list(refs)]).score


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 over whitespace tokens (LCS-based), in [0, 1]."""
    h = nfc(hyp).split()
    r = nfc(ref).split()
    if not h or not r:
        return 0.0
    # LCS length via DP.
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            if h[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(h)][len(r)]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(h), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def corpus_metrics(hypotheses: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """All locked metrics over a corpus of (hypothesis, reference) pairs."""
    if len(hypotheses) != len(references):
        raise ValueError(f"count mismatch: {len(hypotheses)} hyps vs {len(references)} refs")
    hyps = [nfc(h) for h in hypotheses]
    refs = [nfc(r) for r in references]

    chrf = CHRF(word_order=0)         # chrF
    chrfpp = CHRF(word_order=2)       # chrF++
    out = {
        "bleu1": _bleu_k(hyps, refs, 1),
        "bleu2": _bleu_k(hyps, refs, 2),
        "bleu3": _bleu_k(hyps, refs, 3),
        "bleu4": _bleu_k(hyps, refs, 4),
        "chrf": chrf.corpus_score(hyps, [refs]).score,
        "chrf++": chrfpp.corpus_score(hyps, [refs]).score,
        "rougeL": float(np.mean([rouge_l(h, r) for h, r in zip(hyps, refs)])),
    }
    return out


def _corpus_metric(hypotheses: Sequence[str], references: Sequence[str], metric: str) -> float:
    """Compute one locked corpus metric without building the full metric suite."""
    if len(hypotheses) != len(references):
        raise ValueError(f"count mismatch: {len(hypotheses)} hyps vs {len(references)} refs")
    hyps = [nfc(h) for h in hypotheses]
    refs = [nfc(r) for r in references]
    if metric.startswith("bleu") and metric[4:].isdigit():
        return _bleu_k(hyps, refs, int(metric[4:]))
    if metric == "chrf":
        return CHRF(word_order=0).corpus_score(hyps, [refs]).score
    if metric == "chrf++":
        return CHRF(word_order=2).corpus_score(hyps, [refs]).score
    if metric == "rougeL":
        return float(np.mean([rouge_l(h, r) for h, r in zip(hyps, refs)]))
    raise KeyError(metric)


def bootstrap_ci(hypotheses: Sequence[str], references: Sequence[str],
                 metric: str = "bleu4", n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for one metric by resampling sentence pairs."""
    hyps = list(hypotheses)
    refs = list(references)
    m = len(hyps)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        vh = [hyps[i] for i in idx]
        vr = [refs[i] for i in idx]
        vals.append(_corpus_metric(vh, vr, metric))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def metric_ci(hypotheses, references, metric: str = "bleu4",
              n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Point estimate plus percentile bootstrap CI for one metric.

    Returns (point, lo, hi). Thin wrapper over corpus_metrics + bootstrap_ci
    so callers get all three numbers from one call with a single seed.
    """
    point = _corpus_metric(hypotheses, references, metric)
    lo, hi = bootstrap_ci(hypotheses, references, metric=metric,
                          n=n, seed=seed, alpha=alpha)
    return float(point), lo, hi


def paired_bootstrap_pvalue(hyps_a, hyps_b, references, metric: str = "bleu4",
                            n: int = 1000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for H0: metric(A) == metric(B).

    Paired: the same resampled sentence indices score both systems, so the
    test controls for which sentences are easy/hard. p is the fraction of
    resampled deltas (B-A) that fall on the opposite side of zero from the
    observed delta, doubled for a two-sided test and clipped to 1.0.
    """
    a, b, r = list(hyps_a), list(hyps_b), list(references)
    if not (len(a) == len(b) == len(r)):
        raise ValueError("hyps_a, hyps_b, references must have equal length")
    m = len(r)
    observed = _corpus_metric(b, r, metric) - _corpus_metric(a, r, metric)
    rng = np.random.default_rng(seed)
    opposite = 0
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        va = [a[i] for i in idx]
        vb = [b[i] for i in idx]
        vr = [r[i] for i in idx]
        delta = _corpus_metric(vb, vr, metric) - _corpus_metric(va, vr, metric)
        if (observed >= 0 and delta <= 0) or (observed < 0 and delta >= 0):
            opposite += 1
    return float(min(1.0, 2.0 * opposite / n))


def bertscore(hypotheses, references, model="csebuetnlp/banglabert", num_layers=12):
    """BERTScore F1 with BanglaBERT. Optional — requires `bert-score` + torch."""
    try:
        import importlib
        import torch  # noqa: F401  # Windows native DLL/import-order guard
        _bscore_mod = importlib.import_module("bert_score.score")
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "BERTScore requires the 'bert-score' package and torch. "
            "Install them in the cloud/full environment; skipped in dev."
        ) from e
    hyps = [nfc(h) for h in hypotheses]
    refs = [nfc(r) for r in references]
    original_get_tokenizer = _bscore_mod.get_tokenizer

    def get_tokenizer_with_sane_max_length(model_type, use_fast=False):
        tokenizer = original_get_tokenizer(model_type, use_fast)
        max_len = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len > 10000:
            tokenizer.model_max_length = 512
        return tokenizer

    _bscore_mod.get_tokenizer = get_tokenizer_with_sane_max_length
    try:
        _, _, f1 = _bscore_mod.score(hyps, refs, model_type=model,
                                     num_layers=num_layers, lang="bn",
                                     verbose=False)
    finally:
        _bscore_mod.get_tokenizer = original_get_tokenizer
    return float(f1.mean())
