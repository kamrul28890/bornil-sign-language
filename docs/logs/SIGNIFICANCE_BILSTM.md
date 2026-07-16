# Significance & Confidence Intervals

Bootstrap n=200, seed=0. Paired p-values vs `transformer`.

| Run | Metric | Point | 95% CI | p vs transformer |
|---|---|---:|---|---:|
| transformer | bleu4 | 0.520 | [0.217, 0.797] | - |
| transformer | chrf++ | 9.646 | [9.334, 9.868] | - |
| bilstm | bleu4 | 0.621 | [0.367, 0.851] | 0.490 |
| bilstm | chrf++ | 8.990 | [8.720, 9.289] | 0.000 |
