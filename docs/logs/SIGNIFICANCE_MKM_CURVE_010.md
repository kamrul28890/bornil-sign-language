# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `scratch`.

| Run | Metric | Point | 95% CI | p vs scratch |
|---|---|---:|---|---:|
| scratch | bleu4 | 0.110 | [0.059, 0.146] | - |
| scratch | chrf++ | 7.237 | [7.086, 7.379] | - |
| pretrained | bleu4 | 0.053 | [0.043, 0.061] | 0.004 |
| pretrained | chrf++ | 6.737 | [6.589, 6.883] | 0.000 |
