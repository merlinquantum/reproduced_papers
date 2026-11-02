# s5_gap_diagnostics_20261021-153326: test metrics (observed AQI targets), mean ± std over seeds

Coverage: **PARTIAL SWEEP**; missing [('KA_pm25_B1_lstm16', 0), ('KA_pm25_B1_lstm16', 1), ('KA_pm25_B1_lstm16', 2), ('KA_pm25_QB_quantum', 0), ('KA_pm25_QB_quantum', 1), ('KA_pm25_QB_quantum', 2)]

| Candidate | Params | Seeds | RMSE | MAE | MAPE (%) | R² | Train RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0_aqi_B0b_ridge | 34 | 3/3 | 9.43 ± 0.00 | 7.54 ± 0.00 | 9.96 ± 0.00 | 0.768 ± 0.000 | 27.70 |
| H0_aqi_B1_lstm16 | 1873 | 3/3 | 10.34 ± 2.86 | 8.40 ± 2.97 | 11.05 ± 4.17 | 0.707 ± 0.166 | 25.78 |
| H0_aqi_B2_lstm3 | 196 | 3/3 | 13.06 ± 3.85 | 10.44 ± 3.95 | 13.64 ± 5.43 | 0.530 ± 0.284 | 27.93 |
| H0_aqi_QB_quantum | 209 | 3/3 | 15.33 ± 4.86 | 12.18 ± 5.22 | 16.16 ± 7.29 | 0.347 ± 0.421 | 28.38 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 | — |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 | — |

Paired test-RMSE difference vs `H0_aqi_B1_lstm16` (same seed; negative = better):

| Candidate | Mean Δ RMSE | Std | Seeds better |
|---|---:|---:|---:|
| H0_aqi_B0b_ridge | -0.90 | 2.86 | 1/3 |
| H0_aqi_B2_lstm3 | +2.72 | 1.03 | 0/3 |
| H0_aqi_QB_quantum | +5.00 | 2.00 | 0/3 |
