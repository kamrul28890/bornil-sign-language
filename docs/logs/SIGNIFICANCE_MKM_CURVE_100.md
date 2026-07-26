# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `scratch`.

| Run | Metric | Point | 95% CI | p vs scratch |
|---|---|---:|---|---:|
| scratch | bleu4 | 0.684 | [0.348, 0.971] | - |
| scratch | chrf++ | 10.913 | [10.589, 11.243] | - |
| pretrained | bleu4 | 0.456 | [0.185, 0.771] | 0.332 |
| pretrained | chrf++ | 9.798 | [9.493, 10.068] | 0.000 |
