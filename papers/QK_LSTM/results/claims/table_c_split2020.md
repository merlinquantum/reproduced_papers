# Table C - split sensitivity: calendar-2020 test (S2)

Scope: **paper-accurate** hyperparameters, calendar-2020 test (183 days 2020-01-01..2020-07-01) as stated in the paper text; 5 seeds 0-4.

| Model | Params | Seeds | RMSE | MAE | MAPE (%) | R² |
|---|---:|---:|---:|---:|---:|---:|
| B0b ridge | 34 | 5 | 9.94 ± 0.00 | 8.11 ± 0.00 | 11.02 ± 0.00 | 0.738 ± 0.000 |
| B0 persistence | 0 | 5 | 10.45 ± 0.00 | 7.78 ± 0.00 | 9.87 ± 0.00 | 0.710 ± 0.000 |
| B1 LSTM(16) | 1873 | 5 | 11.33 ± 1.66 | 9.02 ± 1.43 | 11.68 ± 2.12 | 0.654 ± 0.104 |
| B2 LSTM(3) | 196 | 5 | 11.62 ± 1.78 | 9.44 ± 1.49 | 12.61 ± 2.16 | 0.635 ± 0.104 |
| Q-B quantum | 209 | 5 | 12.32 ± 2.61 | 10.04 ± 2.44 | 13.54 ± 3.63 | 0.583 ± 0.162 |
| B3-B RBF | 209 | 5 | 12.33 ± 2.47 | 10.00 ± 2.28 | 13.52 ± 3.43 | 0.584 ± 0.155 |
| B3-A RBF (N=4) | 449 | 5 | 12.43 ± 2.73 | 10.11 ± 2.49 | 13.71 ± 3.69 | 0.574 ± 0.172 |
| Q-A quantum (N=4) | 449 | 5 | 12.62 ± 2.90 | 10.34 ± 2.68 | 14.01 ± 3.96 | 0.560 ± 0.184 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 |

Paired same-seed differences (A − B), test RMSE and R²; 95 % t-interval:

| Claim | A vs B | ΔRMSE mean [95 % CI] | A better (RMSE) | ΔR² mean [95 % CI] |
|---|---|---:|---:|---:|
| C1 (paper comparison) | Q-B quantum vs B1 LSTM(16) | +0.99 [-1.25, +3.24] | 2/5 | -0.071 [-0.217, +0.075] |
| C1 (paper comparison) | Q-A quantum (N=4) vs B1 LSTM(16) | +1.29 [-1.30, +3.89] | 1/5 | -0.094 [-0.267, +0.079] |
| C1/C2 iso-parameter | Q-B quantum vs B2 LSTM(3) | +0.70 [-0.42, +1.82] | 2/5 | -0.052 [-0.129, +0.024] |
| C4 quantum vs classical kernel | Q-B quantum vs B3-B RBF | -0.01 [-0.37, +0.36] | 2/5 | -0.001 [-0.026, +0.024] |
| C4 quantum vs classical kernel | Q-A quantum (N=4) vs B3-A RBF (N=4) | +0.19 [-0.25, +0.62] | 2/5 | -0.014 [-0.045, +0.016] |
| naive floor | Q-B quantum vs B0 persistence | +1.86 [-1.38, +5.11] | 2/5 | -0.127 [-0.329, +0.075] |
| naive floor | B1 LSTM(16) vs B0 persistence | +0.87 [-1.19, +2.93] | 2/5 | -0.056 [-0.186, +0.074] |
