# Significance & Confidence Intervals

Bootstrap n=200, seed=0. Paired p-values vs `vanilla_random`.

| Run | Metric | Point | 95% CI | p vs vanilla_random |
|---|---|---:|---|---:|
| vanilla_random | bleu4 | 0.520 | [0.217, 0.797] | - |
| vanilla_random | chrf++ | 9.646 | [9.334, 9.868] | - |
| tconv_random | bleu4 | 0.655 | [0.377, 0.895] | 0.450 |
| tconv_random | chrf++ | 9.482 | [9.199, 9.752] | 0.440 |
| eos_pen_random | bleu4 | 0.113 | [0.073, 0.163] | 0.000 |
| eos_pen_random | chrf++ | 7.262 | [6.988, 7.494] | 0.000 |
