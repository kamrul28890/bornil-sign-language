import torch

from bornil.pretrain.masking import sample_span_mask


def test_output_shape_and_dtype():
    lengths = torch.tensor([10, 6, 10])
    mask = sample_span_mask(seq_len=10, lengths=lengths, ratio=0.5,
                            generator=torch.Generator().manual_seed(0))
    assert mask.shape == (3, 10)
    assert mask.dtype == torch.bool


def test_never_masks_beyond_length():
    lengths = torch.tensor([4, 10])
    mask = sample_span_mask(seq_len=10, lengths=lengths, ratio=0.9,
                            min_span=1, max_span=2,
                            generator=torch.Generator().manual_seed(1))
    assert not mask[0, 4:].any()


def test_ratio_is_approximately_respected():
    lengths = torch.full((1,), 1000, dtype=torch.long)
    mask = sample_span_mask(seq_len=1000, lengths=lengths, ratio=0.6,
                            min_span=3, max_span=15,
                            generator=torch.Generator().manual_seed(2))
    frac = mask[0].float().mean().item()
    assert 0.5 <= frac <= 0.7


def test_masked_positions_form_contiguous_spans():
    lengths = torch.tensor([200])
    mask = sample_span_mask(seq_len=200, lengths=lengths, ratio=0.4,
                            min_span=5, max_span=5,
                            generator=torch.Generator().manual_seed(3))
    row = mask[0].tolist()
    # count contiguous True-runs; with min_span==max_span==5 every run's
    # length must be a multiple of 5 give or take the final trim.
    runs = []
    cur = 0
    for v in row:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    assert all(r <= 5 for r in runs)
    assert len(runs) > 1


def test_deterministic_with_seeded_generator():
    lengths = torch.tensor([50])
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    m1 = sample_span_mask(seq_len=50, lengths=lengths, ratio=0.5, generator=g1)
    m2 = sample_span_mask(seq_len=50, lengths=lengths, ratio=0.5, generator=g2)
    assert torch.equal(m1, m2)


def test_zero_length_row_is_never_masked():
    lengths = torch.tensor([0, 10])
    mask = sample_span_mask(seq_len=10, lengths=lengths, ratio=0.5,
                            generator=torch.Generator().manual_seed(4))
    assert not mask[0].any()
