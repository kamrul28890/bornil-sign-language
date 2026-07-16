# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `vanilla_random`.

| Run | Metric | Point | 95% CI | p vs vanilla_random |
|---|---|---:|---|---:|
| vanilla_random | bleu4 | 0.520 | [0.217, 0.840] | - |
| vanilla_random | chrf++ | 9.646 | [9.342, 9.953] | - |
| tconv_random | bleu4 | 0.655 | [0.373, 0.917] | 0.486 |
| tconv_random | chrf++ | 9.482 | [9.200, 9.765] | 0.344 |
| eos_pen_random | bleu4 | 0.113 | [0.074, 0.160] | 0.000 |
| eos_pen_random | chrf++ | 7.262 | [7.001, 7.506] | 0.000 |
