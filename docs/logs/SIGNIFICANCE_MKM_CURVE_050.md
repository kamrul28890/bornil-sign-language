# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `scratch`.

| Run | Metric | Point | 95% CI | p vs scratch |
|---|---|---:|---|---:|
| scratch | bleu4 | 0.220 | [0.159, 0.263] | - |
| scratch | chrf++ | 8.965 | [8.681, 9.241] | - |
| pretrained | bleu4 | 0.135 | [0.054, 0.257] | 0.192 |
| pretrained | chrf++ | 7.673 | [7.496, 7.831] | 0.000 |
