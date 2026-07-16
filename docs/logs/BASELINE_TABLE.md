# Bornil Full-Dataset Baseline Table

Generated: 2026-07-16 01:55

| Split | Run | Source | Decode | Train clips | Test rows | BLEU-4 | chrF++ | ROUGE-L | BERTScore F1 | Exact % | Unique hyps |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random-v1 | c19d0ea08824 | frozen:splits/random-v1 | greedy | 14390 | 1799 | 0.520 | 9.646 | 0.055 | 0.862 | 0.11 | 1616 |
| unseen-sentence-v1 | 60320f4c6a60 | frozen:splits/unseen-sentence-v1 | greedy | 14390 | 1799 | 0.419 | 9.187 | 0.048 | 0.860 | 0.00 | 1692 |
| signerindep-v1 | 46a706570712 | frozen:splits/signerindep-v1 | greedy | 14500 | 1500 | 0.351 | 8.382 | 0.048 | 0.855 | 0.13 | 1365 |

## Details

### random-v1

- Run: `data_full/runs/c19d0ea08824`
- Train time: 42.0 min
- Best validation BLEU-4: 0.467
- BLEU-4 CI: [0.219, 0.834]
- chrF++ CI: [9.342, 9.914]
- ROUGE-L CI: [0.050, 0.061]
- BLEU-1/2/3/4: 11.723 / 3.325 / 1.110 / 0.520
- Empty hypotheses: 0
- Exact matches: 2
- Average reference/hypothesis tokens: 7.50 / 6.50

### unseen-sentence-v1

- Run: `data_full/runs/60320f4c6a60`
- Train time: 37.9 min
- Best validation BLEU-4: 0.625
- BLEU-4 CI: [0.229, 0.601]
- chrF++ CI: [8.940, 9.467]
- ROUGE-L CI: [0.044, 0.053]
- BLEU-1/2/3/4: 11.087 / 3.025 / 0.986 / 0.419
- Empty hypotheses: 0
- Exact matches: 0
- Average reference/hypothesis tokens: 7.62 / 6.38

### signerindep-v1

- Run: `data_full/runs/46a706570712`
- Train time: 42.0 min
- Best validation BLEU-4: 0.603
- BLEU-4 CI: [0.164, 0.551]
- chrF++ CI: [8.106, 8.641]
- ROUGE-L CI: [0.041, 0.053]
- BLEU-1/2/3/4: 8.777 / 2.371 / 0.800 / 0.351
- Empty hypotheses: 0
- Exact matches: 2
- Average reference/hypothesis tokens: 9.40 / 6.85
