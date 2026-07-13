"""SentencePiece target tokenizer wrapper (WS-A).

Owns the target-side convention in one place: encode wraps subword ids with
BOS/EOS; decode strips pad/bos/eos before detokenizing. The model and eval code
consume this rather than re-implementing the special-token handling (which is
how off-by-one BLEU bugs creep in).
"""
from __future__ import annotations

from typing import List

import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_file: str):
        self._sp = spm.SentencePieceProcessor(model_file=model_file)
        self.pad_id = self._sp.pad_id()
        self.bos_id = self._sp.bos_id()
        self.eos_id = self._sp.eos_id()
        self.vocab_size = self._sp.get_piece_size()
        self._specials = {self.pad_id, self.bos_id, self.eos_id}

    def encode(self, text) -> List[int]:
        """Subword-encode `text`, wrapped as [BOS] ... [EOS]."""
        return [self.bos_id] + self._sp.encode(str(text)) + [self.eos_id]

    def decode(self, token_ids) -> str:
        """Detokenize, dropping pad/bos/eos ids."""
        ids = [int(i) for i in token_ids if int(i) not in self._specials]
        return self._sp.decode(ids)
