# Table A - paper hyperparameters, 80/20 split (S1)

Scope: **paper-accurate** (full data, Table II hyperparameters, 5 seeds 0-4, final epoch). Test = 402 days 2019-05-27..2020-07-01; observed AQI targets.

| Model | Params | Seeds | RMSE | MAE | MAPE (%) | R² |
|---|---:|---:|---:|---:|---:|---:|
| B0 persistence | 0 | 5 | 11.42 ± 0.00 | 8.00 ± 0.00 | 10.11 ± 0.00 | 0.661 ± 0.000 |
| B0b ridge | 34 | 5 | 12.42 ± 0.00 | 10.21 ± 0.00 | 14.15 ± 0.00 | 0.598 ± 0.000 |
| B1 LSTM(16) | 1873 | 5 | 12.61 ± 2.66 | 10.06 ± 2.67 | 13.45 ± 4.05 | 0.571 ± 0.185 |
| B2 LSTM(3) | 196 | 5 | 13.35 ± 3.61 | 10.71 ± 3.67 | 14.28 ± 5.41 | 0.509 ± 0.261 |
| Q-A quantum (N=4) | 449 | 5 | 15.66 ± 3.77 | 12.20 ± 3.80 | 16.24 ± 5.67 | 0.332 ± 0.315 |
| Q-B quantum | 209 | 5 | 15.68 ± 3.97 | 12.38 ± 4.01 | 16.53 ± 5.96 | 0.327 ± 0.328 |
| B3-A RBF (N=4) | 449 | 5 | 15.93 ± 3.85 | 12.48 ± 3.91 | 16.62 ± 5.88 | 0.308 ± 0.327 |
| B3-B RBF | 209 | 5 | 16.01 ± 3.80 | 12.67 ± 3.92 | 16.90 ± 5.89 | 0.303 ± 0.322 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 |

Paired same-seed differences (A − B), test RMSE and R²; 95 % t-interval:

| Claim | A vs B | ΔRMSE mean [95 % CI] | A better (RMSE) | ΔR² mean [95 % CI] |
|---|---|---:|---:|---:|
| C1 (paper comparison) | Q-B quantum vs B1 LSTM(16) | +3.07 [+0.76, +5.38] | 0/5 | -0.244 [-0.457, -0.031] |
| C1 (paper comparison) | Q-A quantum (N=4) vs B1 LSTM(16) | +3.05 [+1.15, +4.95] | 0/5 | -0.239 [-0.427, -0.052] |
| C1/C2 iso-parameter | Q-B quantum vs B2 LSTM(3) | +2.33 [+1.60, +3.06] | 0/5 | -0.182 [-0.277, -0.086] |
| C4 quantum vs classical kernel | Q-B quantum vs B3-B RBF | -0.33 [-0.59, -0.06] | 5/5 | +0.024 [+0.011, +0.038] |
| C4 quantum vs classical kernel | Q-A quantum (N=4) vs B3-A RBF (N=4) | -0.27 [-0.45, -0.08] | 5/5 | +0.023 [+0.005, +0.041] |
| naive floor | Q-B quantum vs B0 persistence | +4.26 [-0.66, +9.19] | 0/5 | -0.334 [-0.741, +0.074] |
| naive floor | B1 LSTM(16) vs B0 persistence | +1.20 [-2.11, +4.50] | 3/5 | -0.090 [-0.319, +0.140] |
