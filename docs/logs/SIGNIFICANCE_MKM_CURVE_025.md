# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `scratch`.

| Run | Metric | Point | 95% CI | p vs scratch |
|---|---|---:|---|---:|
| scratch | bleu4 | 0.142 | [0.092, 0.180] | - |
| scratch | chrf++ | 8.790 | [8.544, 9.034] | - |
| pretrained | bleu4 | 0.136 | [0.076, 0.180] | 0.810 |
| pretrained | chrf++ | 4.539 | [4.409, 4.676] | 0.000 |
