# Table D - Phase 4 MerLin adaptation vs Phase 3 (paper hyperparameters, 80/20 split)

Scope: **MerLin adaptation (simulation)**, paper hyperparameters, 5 seeds 0-4 paired with S1 (same seeds -> same init and references). Photonic kernel: 2 photons, 4 modes, FOCK computation (threshold-equivalent detection, no postselection), fixed meshes 7/8, analytic (shots 0).

| Model | Params | Seeds | RMSE | MAE | MAPE (%) | R² |
|---|---:|---:|---:|---:|---:|---:|
| B0 persistence | 0 | 5 | 11.42 ± 0.00 | 8.00 ± 0.00 | 10.11 ± 0.00 | 0.661 ± 0.000 |
| B1 LSTM(16) | 1873 | 5 | 12.61 ± 2.66 | 10.06 ± 2.67 | 13.45 ± 4.05 | 0.571 ± 0.185 |
| B2 LSTM(3) | 196 | 5 | 13.35 ± 3.61 | 10.71 ± 3.67 | 14.28 ± 5.41 | 0.509 ± 0.261 |
| Q-B photonic 1ph (ablation) | 209 | 5 | 15.52 ± 3.98 | 12.38 ± 4.22 | 16.55 ± 6.27 | 0.339 ± 0.334 |
| Q-A qubit (N=4) | 449 | 5 | 15.66 ± 3.77 | 12.20 ± 3.80 | 16.24 ± 5.67 | 0.332 ± 0.315 |
| Q-B qubit | 209 | 5 | 15.68 ± 3.97 | 12.38 ± 4.01 | 16.53 ± 5.96 | 0.327 ± 0.328 |
| Q-B photonic 2ph | 209 | 5 | 15.73 ± 3.72 | 12.53 ± 3.98 | 16.78 ± 6.00 | 0.327 ± 0.312 |
| Q-A photonic 2ph (N=4) | 449 | 5 | 15.74 ± 4.07 | 12.43 ± 4.15 | 16.63 ± 6.21 | 0.320 ± 0.341 |
| B3-B RBF | 209 | 5 | 16.01 ± 3.80 | 12.67 ± 3.92 | 16.90 ± 5.89 | 0.303 ± 0.322 |
| QK-LSTM (paper) | 209 | 1 | 9.20 | 7.15 | 9.14 | 0.84 |
| LSTM (paper) | 1873 | 1 | 15.94 | 11.07 | 13.32 | 0.78 |

Paired same-seed differences (A − B), test RMSE and R²; 95 % t-interval:

| Claim | A vs B | ΔRMSE mean [95 % CI] | A better (RMSE) | ΔR² mean [95 % CI] |
|---|---|---:|---:|---:|
| preservation (tol ±1.0) | Q-B photonic 2ph vs Q-B qubit | +0.05 [-0.62, +0.71] | 3/5 | -0.000 [-0.051, +0.051] |
| preservation (tol ±1.0) | Q-A photonic 2ph (N=4) vs Q-A qubit (N=4) | +0.08 [-0.46, +0.61] | 2/5 | -0.011 [-0.054, +0.032] |
| multi-photon interference | Q-B photonic 2ph vs Q-B photonic 1ph (ablation) | +0.20 [-0.75, +1.15] | 2/5 | -0.012 [-0.089, +0.064] |
| C4 photonic vs classical kernel | Q-B photonic 2ph vs B3-B RBF | -0.28 [-0.72, +0.17] | 4/5 | +0.024 [-0.015, +0.063] |
| C1 photonic vs LSTM | Q-B photonic 2ph vs B1 LSTM(16) | +3.12 [+1.32, +4.91] | 0/5 | -0.244 [-0.423, -0.065] |
| C1/C2 photonic iso-param | Q-B photonic 2ph vs B2 LSTM(3) | +2.38 [+1.80, +2.95] | 0/5 | -0.182 [-0.255, -0.108] |
| naive floor | Q-B photonic 2ph vs B0 persistence | +4.31 [-0.31, +8.93] | 0/5 | -0.334 [-0.721, +0.053] |
