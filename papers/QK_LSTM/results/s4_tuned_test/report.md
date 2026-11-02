# s4_tuned_test_20261021-152814: test metrics (observed AQI targets), mean ± std over seeds

Coverage: **COMPLETE SWEEP**

| Candidate | Params | Seeds | RMSE | MAE | MAPE (%) | R² | Train RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2_lstm3_lr0.01 | 196 | 5/5 | 11.72 ± 2.14 | 9.29 ± 2.24 | 12.62 ± 3.65 | 0.633 ± 0.133 | 30.40 |
| B1_lstm16_lr0.01 | 1873 | 5/5 | 12.88 ± 2.78 | 10.22 ± 3.05 | 13.92 ± 4.60 | 0.552 ± 0.200 | 29.67 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 | — |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 | — |

Paired test-RMSE difference vs `B1_lstm16_lr0.01` (same seed; negative = better):

| Candidate | Mean Δ RMSE | Std | Seeds better |
|---|---:|---:|---:|
| B2_lstm3_lr0.01 | -1.17 | 0.81 | 5/5 |
