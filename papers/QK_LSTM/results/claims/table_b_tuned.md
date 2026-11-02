# Table B - equal-budget validation-tuned, 80/20 split (S3 selection; S1 + S4 runs)

Scope: **paper-accurate data and split, validation-tuned lr/N/alpha** (S3, validation only); 5 seeds 0-4. Kernel models, ridge, persistence: S3 selected the S1 configuration (S1 runs reused). LSTMs: selected lr 0.01 (S4 runs).

| Model | Params | Seeds | RMSE | MAE | MAPE (%) | R² |
|---|---:|---:|---:|---:|---:|---:|
| B0 persistence | 0 | 5 | 11.42 ± 0.00 | 8.00 ± 0.00 | 10.11 ± 0.00 | 0.661 ± 0.000 |
| B2 LSTM(3) | 196 | 5 | 11.72 ± 2.14 | 9.29 ± 2.24 | 12.62 ± 3.65 | 0.633 ± 0.133 |
| B0b ridge | 34 | 5 | 12.42 ± 0.00 | 10.21 ± 0.00 | 14.15 ± 0.00 | 0.598 ± 0.000 |
| B1 LSTM(16) | 1873 | 5 | 12.88 ± 2.78 | 10.22 ± 3.05 | 13.92 ± 4.60 | 0.552 ± 0.200 |
| Q-A quantum (N=4) | 449 | 5 | 15.66 ± 3.77 | 12.20 ± 3.80 | 16.24 ± 5.67 | 0.332 ± 0.315 |
| Q-B quantum | 209 | 5 | 15.68 ± 3.97 | 12.38 ± 4.01 | 16.53 ± 5.96 | 0.327 ± 0.328 |
| B3-A RBF (N=4) | 449 | 5 | 15.93 ± 3.85 | 12.48 ± 3.91 | 16.62 ± 5.88 | 0.308 ± 0.327 |
| B3-B RBF | 209 | 5 | 16.01 ± 3.80 | 12.67 ± 3.92 | 16.90 ± 5.89 | 0.303 ± 0.322 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 |

Paired same-seed differences (A − B), test RMSE and R²; 95 % t-interval:

| Claim | A vs B | ΔRMSE mean [95 % CI] | A better (RMSE) | ΔR² mean [95 % CI] |
|---|---|---:|---:|---:|
| C1 (paper comparison) | Q-B quantum vs B1 LSTM(16) | +2.80 [+0.50, +5.10] | 0/5 | -0.225 [-0.443, -0.007] |
| C1 (paper comparison) | Q-A quantum (N=4) vs B1 LSTM(16) | +2.78 [+0.68, +4.88] | 0/5 | -0.220 [-0.422, -0.019] |
| C1/C2 iso-parameter | Q-B quantum vs B2 LSTM(3) | +3.96 [+1.37, +6.55] | 0/5 | -0.306 [-0.568, -0.044] |
| C4 quantum vs classical kernel | Q-B quantum vs B3-B RBF | -0.33 [-0.59, -0.06] | 5/5 | +0.024 [+0.011, +0.038] |
| C4 quantum vs classical kernel | Q-A quantum (N=4) vs B3-A RBF (N=4) | -0.27 [-0.45, -0.08] | 5/5 | +0.023 [+0.005, +0.041] |
| naive floor | Q-B quantum vs B0 persistence | +4.26 [-0.66, +9.19] | 0/5 | -0.334 [-0.741, +0.074] |
| naive floor | B1 LSTM(16) vs B0 persistence | +1.47 [-1.98, +4.92] | 2/5 | -0.109 [-0.358, +0.140] |
