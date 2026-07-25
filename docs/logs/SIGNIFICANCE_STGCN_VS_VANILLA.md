# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `vanilla`.

| Run | Metric | Point | 95% CI | p vs vanilla |
|---|---|---:|---|---:|
| vanilla | bleu4 | 0.520 | [0.217, 0.840] | - |
| vanilla | chrf++ | 9.646 | [9.342, 9.953] | - |
| stgcn | bleu4 | 0.356 | [0.160, 0.557] | 0.484 |
| stgcn | chrf++ | 8.265 | [8.041, 8.520] | 0.000 |
