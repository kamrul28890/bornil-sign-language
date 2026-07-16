# TConv Length-Pressure Result

Generated: 2026-07-15

Run: `data_full/runs/a459d1148d0c`

Config: `configs/full_tconv_len.yaml`

Split: `random-v1`

## Comparison

| Model | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Unique hyps | Repeated rows | Avg hyp tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla SignTransformer | 0.520 | 9.646 | 0.055 | 0.862 | 1616 | 15.6% | 6.50 |
| TConv2k5 | 0.655 | 9.482 | 0.054 | 0.864 | 1644 | 14.0% | 6.30 |
| TConv2k5 + early EOS penalty 0.2 | 0.113 | 7.262 | 0.031 | 0.857 | 414 | 83.6% | 6.21 |

## Length-Bin Behavior

| Ref token bin | TConv BLEU-4 | TConv+EOS BLEU-4 | TConv ROUGE-L | TConv+EOS ROUGE-L |
|---|---:|---:|---:|---:|
| 0-3 | 2.178 | 0.330 | 0.084 | 0.038 |
| 4-6 | 1.121 | 0.285 | 0.062 | 0.044 |
| 7-9 | 0.596 | 0.162 | 0.046 | 0.021 |
| 10-12 | 0.146 | 0.144 | 0.022 | 0.013 |
| 13-plus | 0.135 | 0.067 | 0.022 | 0.016 |

## Interpretation

The early-EOS penalty setting failed. It did not improve long references; instead
it collapsed output diversity and made the model repeat a small set of generic
sentences. The top repeated hypotheses account for hundreds of rows, and one
diagnostic example reached the full `max_decode_len` with token repetition.

Conclusion: abandon `early_eos_penalty: 0.2`. Do not run this setting on
`unseen-sentence-v1` or `signerindep-v1`.

The useful result from this branch remains the plain TConv2k5 model, which
improved BLEU-4 and output diversity but did not improve ROUGE-L or long-bin
behavior enough to justify harder-split runs yet.
