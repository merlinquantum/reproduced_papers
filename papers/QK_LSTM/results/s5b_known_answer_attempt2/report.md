# s5b_known_answer_attempt2_20261021-154126: test metrics (observed AQI targets), mean ± std over seeds

Coverage: **COMPLETE SWEEP**

| Candidate | Params | Seeds | RMSE | MAE | MAPE (%) | R² | Train RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| KA_pm25_QB_quantum | 209 | 3/3 | 6.90 ± 5.38 | 6.48 ± 5.43 | 27.07 ± 23.55 | 0.607 ± 0.532 | 15.90 |
| KA_pm25_B1_lstm16 | 1873 | 3/3 | 29.00 ± 6.01 | 12.90 ± 3.31 | 47.43 ± 12.14 | -4.073 ± 1.930 | 15.25 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 | — |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 | — |

Paired test-RMSE difference vs `KA_pm25_B1_lstm16` (same seed; negative = better):

| Candidate | Mean Δ RMSE | Std | Seeds better |
|---|---:|---:|---:|
| KA_pm25_QB_quantum | -22.10 | 4.79 | 3/3 |
