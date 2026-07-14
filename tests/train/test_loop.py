"""Integration test for the train/eval loop (WS-C).

Trains the Transformer baseline to overfit a handful of (random clip -> real
Bangla sentence) pairs, then evaluates through greedy decode + the locked metric
protocol. If the whole path (teacher forcing, masking, generate, detokenize,
scoring) is correct, memorized examples score near-perfect BLEU.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.tokenizer import Tokenizer
from bornil.models.transformer import SignTransformer
from bornil.train.loop import train_epoch, evaluate

SENTENCES = ["আমি ভালো আছি", "সে স্কুলে যায়", "আজ আকাশ পরিষ্কার", "পাখি উড়ে যায়"]


def _setup():
    tok = Tokenizer("bsl_sp.model")
    rng = np.random.default_rng(0)
    n = len(SENTENCES)
    feats = rng.standard_normal((n, 128, 258)).astype(np.float32)
    lengths = np.array([16, 24, 20, 12], dtype=np.int32)
    targets = [tok.encode(s) for s in SENTENCES]
    ds = SignDataset(feats, lengths, list(range(n)), targets)
    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        collate_fn=make_collate_fn(tok.pad_id))
    model = SignTransformer(input_dim=218, vocab_size=tok.vocab_size, d_model=64,
                            nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                            dim_feedforward=128, dropout=0.0, pad_id=tok.pad_id)
    return tok, loader, model


def test_train_epoch_reduces_loss():
    torch.manual_seed(0)
    tok, loader, model = _setup()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = train_epoch(model, loader, opt, device="cpu", pad_id=tok.pad_id)
    for _ in range(5):
        last = train_epoch(model, loader, opt, device="cpu", pad_id=tok.pad_id)
    assert last < first


def test_train_epoch_with_scheduler_and_label_smoothing():
    import torch
    from bornil.train.schedule import noam_lambda
    torch.manual_seed(0)
    tok, loader, model = _setup()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam_lambda(5))
    loss = train_epoch(model, loader, opt, device="cpu", pad_id=tok.pad_id,
                       label_smoothing=0.1, scheduler=sched)
    assert loss == loss  # finite (not NaN)
    assert sched.last_epoch > 0  # scheduler advanced


def test_evaluate_with_beam_search():
    import torch
    torch.manual_seed(0)
    tok, loader, model = _setup()
    metrics, hyps, refs = evaluate(model, loader, tok, device="cpu", max_len=16, beam_size=3)
    assert "bleu4" in metrics and len(hyps) == len(refs)


def test_evaluate_returns_metrics_and_overfits():
    torch.manual_seed(0)
    tok, loader, model = _setup()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(120):
        train_epoch(model, loader, opt, device="cpu", pad_id=tok.pad_id)
    metrics, hyps, refs = evaluate(model, loader, tok, device="cpu", max_len=32)
    assert set(["bleu4", "chrf", "chrf++", "rougeL"]).issubset(metrics.keys())
    assert len(hyps) == len(refs) == len(SENTENCES)
    # Memorized -> high BLEU and references recovered from the targets.
    assert sorted(refs) == sorted(SENTENCES)
    assert metrics["bleu4"] > 50.0
