# Decode Ablation Notes

Generated: 2026-07-15

Run: data_full/runs/c19d0ea08824
Split: random-v1
Checkpoint: best.pt

| Decode | BLEU-4 | chrF++ | ROUGE-L | Avg hyp tokens | Unique hyps | Time |
|---|---:|---:|---:|---:|---:|---:|
| greedy_max256 | 0.520 | 9.646 | 0.055 | 6.50 | 1616 | 0.3 min |
| beam5_lp0.8_max256 | 0.517 | 9.462 | 0.053 | 6.19 | 1430 | 97.0 min |
| beam5_lp1_max256 | 0.516 | 9.738 | 0.053 | 6.48 | 1430 | 631.6 min |
| beam5_lp1.2_max256 | 0.505 | 9.872 | 0.052 | 7.35 | 1431 | 100.2 min |

Conclusion: greedy remains the official decode setting. Beam search does not improve BLEU-4 or ROUGE-L and reduces output diversity. Higher length penalty raises chrF++ mostly by generating longer text, not better grounded translations.
