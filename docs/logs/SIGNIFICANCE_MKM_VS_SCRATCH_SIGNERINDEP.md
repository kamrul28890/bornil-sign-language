# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `scratch`.

| Run | Metric | Point | 95% CI | p vs scratch |
|---|---|---:|---|---:|
| scratch | bleu4 | 0.792 | [0.520, 1.061] | - |
| scratch | chrf++ | 10.361 | [10.029, 10.700] | - |
| pretrained | bleu4 | 0.737 | [0.424, 1.015] | 0.748 |
| pretrained | chrf++ | 10.716 | [10.382, 11.062] | 0.064 |
