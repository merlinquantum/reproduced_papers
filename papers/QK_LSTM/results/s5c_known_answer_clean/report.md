# s5c_known_answer_clean_20261021-155457: test metrics (observed AQI targets), mean ± std over seeds

Coverage: **COMPLETE SWEEP**

| Candidate | Params | Seeds | RMSE | MAE | MAPE (%) | R² | Train RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| KAc_pm25_QB_quantum_lr0.001 | 209 | 3/3 | 0.81 ± 0.16 | 0.63 ± 0.13 | 2.57 ± 0.63 | 0.996 ± 0.001 | 2.43 |
| KAc_pm25_B1_lstm16_lr0.01 | 1873 | 3/3 | 2.51 ± 1.83 | 1.85 ± 1.08 | 6.55 ± 1.95 | 0.950 ± 0.064 | 5.05 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 | — |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 | — |

Paired test-RMSE difference vs `KAc_pm25_B1_lstm16_lr0.01` (same seed; negative = better):

| Candidate | Mean Δ RMSE | Std | Seeds better |
|---|---:|---:|---:|
| KAc_pm25_QB_quantum_lr0.001 | -1.70 | 1.80 | 3/3 |
