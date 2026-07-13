"""Tests for the SentencePiece target tokenizer wrapper (WS-A).

Uses the real trained model (`bsl_sp.model`, 4000 BPE, pad/bos/eos = 0/2/3).
The wrapper owns the BOS/EOS convention and special-id stripping so the model
and eval code never re-derive it.
"""
import pytest

from bornil.data.tokenizer import Tokenizer

MODEL = "bsl_sp.model"
SENT = "আমি বাংলায় গান গাই"


@pytest.fixture(scope="module")
def tok():
    return Tokenizer(MODEL)


def test_special_ids(tok):
    assert tok.pad_id == 0
    assert tok.bos_id == 2
    assert tok.eos_id == 3
    assert tok.vocab_size == 4000


def test_encode_wraps_with_bos_eos(tok):
    ids = tok.encode(SENT)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert len(ids) > 2


def test_decode_strips_specials_and_roundtrips(tok):
    ids = tok.encode(SENT)
    assert tok.decode(ids) == SENT


def test_decode_ignores_pad(tok):
    ids = tok.encode(SENT) + [tok.pad_id, tok.pad_id]
    assert tok.decode(ids) == SENT


def test_encode_accepts_non_str(tok):
    # numbers/other types are coerced to str, not crash.
    assert isinstance(tok.encode(12345), list)
