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
        vals.append(corpus_metrics(vh, vr)[metric])
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def bertscore(hypotheses, references, model="csebuetnlp/banglabert"):
    """BERTScore F1 with BanglaBERT. Optional — requires `bert-score` + torch."""
    try:
        from bert_score import score as _bscore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "BERTScore requires the 'bert-score' package and torch. "
            "Install them in the cloud/full environment; skipped in dev."
        ) from e
    hyps = [nfc(h) for h in hypotheses]
    refs = [nfc(r) for r in references]
    _, _, f1 = _bscore(hyps, refs, model_type=model, lang="bn", verbose=False)
    return float(f1.mean())
