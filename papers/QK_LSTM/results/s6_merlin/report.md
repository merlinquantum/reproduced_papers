# s6_merlin_20261021-183822: test metrics (observed AQI targets), mean ± std over seeds

Coverage: **COMPLETE SWEEP**

| Candidate | Params | Seeds | RMSE | MAE | MAPE (%) | R² | Train RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| QB_photonic_1ph_ablation | 209 | 5/5 | 15.52 ± 3.98 | 12.38 ± 4.22 | 16.55 ± 6.27 | 0.339 ± 0.334 | 32.78 |
| QB_photonic_2ph_209 | 209 | 5/5 | 15.73 ± 3.72 | 12.53 ± 3.98 | 16.78 ± 6.00 | 0.327 ± 0.312 | 32.69 |
| QA_photonic_2ph_eq6_N4 | 449 | 5/5 | 15.74 ± 4.07 | 12.43 ± 4.15 | 16.63 ± 6.21 | 0.320 ± 0.341 | 32.70 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 | — |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 | — |

Paired test-RMSE difference vs `QB_photonic_2ph_209` (same seed; negative = better):

| Candidate | Mean Δ RMSE | Std | Seeds better |
|---|---:|---:|---:|
| QB_photonic_1ph_ablation | -0.20 | 0.76 | 3/5 |
| QA_photonic_2ph_eq6_N4 | +0.01 | 0.55 | 3/5 |
