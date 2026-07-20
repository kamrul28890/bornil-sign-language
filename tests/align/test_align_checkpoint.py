import bornil.runtime  # noqa: F401

import numpy as np
import torch

from scripts.align_checkpoint import span_confidences, spans_to_rows


class _FakeTok:
    def decode(self, ids):
        return "tok" + "_".join(str(i) for i in ids)


def test_spans_to_rows_shapes_and_fields():
    spans = [(0, 1, 3), (1, 5, 5)]
    rows = spans_to_rows(row=7, spans=spans, labels=[11, 12],
                         tokenizer=_FakeTok(), n_frames=9)
    assert len(rows) == 2
    assert rows[0] == {"row": 7, "label_index": 0, "token_id": 11,
                       "token_str": "tok11", "start_frame": 1,
                       "end_frame": 3, "n_frames": 9, "peak_conf": None}
    assert rows[1]["start_frame"] == 5 and rows[1]["end_frame"] == 5


def test_spans_to_rows_attaches_confidence_in_order():
    spans = [(0, 1, 3), (1, 5, 5)]
    rows = spans_to_rows(row=0, spans=spans, labels=[11, 12],
                         tokenizer=_FakeTok(), n_frames=9,
                         peak_conf=[0.8, 0.02])
    assert rows[0]["peak_conf"] == 0.8
    assert rows[1]["peak_conf"] == 0.02


def test_spans_to_rows_empty():
    assert spans_to_rows(0, [], [], _FakeTok(), 5) == []


def _peaked(argmax_seq, vocab=6):
    T = len(argmax_seq)
    p = torch.full((T, vocab), 1e-6)
    for t, v in enumerate(argmax_seq):
        p[t, v] = 1.0
    return (p / p.sum(-1, keepdim=True)).log()


def test_span_confidences_reads_peak_posterior_of_the_label_token():
    # label 3 spans frames 1..2 (confident), label 5 spans frame 4 (confident)
    lp = _peaked([0, 3, 3, 0, 5])
    spans = [(0, 1, 2), (1, 4, 4)]
    conf = span_confidences(lp, spans, labels=[3, 5])
    assert conf[0] > 0.99 and conf[1] > 0.99


def test_span_confidences_low_when_label_placed_off_its_evidence():
    # The head never favours label 5; a span placed on blank frames scores ~0.
    lp = _peaked([0, 3, 3, 0, 0])
    spans = [(0, 1, 2), (1, 4, 4)]      # label 5 forced onto blank frame 4
    conf = span_confidences(lp, spans, labels=[3, 5])
    assert conf[0] > 0.99
    assert conf[1] < 0.01


def test_span_confidences_takes_the_max_over_the_span():
    # label 3 spans frames 1..3, but only frame 2 is confident for it.
    lp = _peaked([0, 0, 3, 0])
    spans = [(0, 1, 3)]
    conf = span_confidences(lp, spans, labels=[3])
    assert conf[0] > 0.99


def test_span_confidences_accepts_numpy():
    lp = _peaked([0, 3, 0]).numpy()
    conf = span_confidences(lp, [(0, 1, 1)], labels=[3])
    assert conf[0] > 0.99


def test_span_confidences_length_matches_spans():
    lp = _peaked([0, 3, 5, 2, 0])
    spans = [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
    conf = span_confidences(lp, spans, labels=[3, 5, 2])
    assert len(conf) == 3
