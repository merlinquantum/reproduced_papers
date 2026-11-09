# S4 conditioning-vector stability vs accuracy

Metric: mean over epochs of ||z_t - z_(t-1)|| / ||z_(t-1)|| for the z used in training (EMA applied when enabled)

| arm | n | used-z relative step | raw-z relative step | test acc % |
|---|---|---|---|---|
| B0_zero | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 10.81 ± 0.93 |
| B4_fixed_photonic | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 99.14 ± 0.09 |
| B3f_random_fixed | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 99.18 ± 0.02 |
| pqkd | 3 | 0.0025 ± 0.0015 | 0.0247 ± 0.0140 | 99.13 ± 0.04 |
| B5_surrogate | 3 | 0.2651 ± 0.0070 | 1.3904 ± 0.0036 | 98.82 ± 0.09 |
| B6_classical_histogram | 3 | 0.2715 ± 0.0056 | 1.3915 ± 0.0014 | 98.48 ± 0.09 |
| B3r_random_resampled | 3 | 0.3153 ± 0.0047 | 1.4193 ± 0.0028 | 98.15 ± 0.35 |

Spearman rank correlation (used-z step vs test accuracy, per run, B0 excluded, n=18): -0.896
