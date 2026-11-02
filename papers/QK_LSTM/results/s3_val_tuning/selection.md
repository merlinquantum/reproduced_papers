# s3_val_tuning_20261021-142942: validation-only selection (predeclared rule)

## QB_quantum: selected `QB_quantum_lr0.001` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| QB_quantum_lr0.001 | 209 | 3 | 15.897 ± 0.507 | yes |
| QB_quantum_lr0.01 | 209 | 3 | 18.042 ± 2.298 |  |
| QB_quantum_lr0.003 | 209 | 3 | 19.086 ± 1.121 |  |

## B3B_rbf: selected `B3B_rbf_lr0.001` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| B3B_rbf_lr0.001 | 209 | 3 | 15.819 ± 0.620 | yes |
| B3B_rbf_lr0.01 | 209 | 3 | 18.304 ± 3.006 |  |
| B3B_rbf_lr0.003 | 209 | 3 | 19.323 ± 1.779 |  |

## QA_quantum: selected `QA_quantum_N4_lr0.001` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| QA_quantum_N4_lr0.001 | 449 | 3 | 16.385 ± 1.084 | yes |
| QA_quantum_N16_lr0.001 | 1217 | 3 | 17.138 ± 0.191 |  |
| QA_quantum_N4_lr0.01 | 449 | 3 | 17.728 ± 1.224 |  |
| QA_quantum_N4_lr0.003 | 449 | 3 | 19.213 ± 2.379 |  |
| QA_quantum_N16_lr0.01 | 1217 | 3 | 19.676 ± 6.657 |  |
| QA_quantum_N16_lr0.003 | 1217 | 3 | 19.956 ± 1.441 |  |

## B3A_rbf: selected `B3A_rbf_N4_lr0.001` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| B3A_rbf_N4_lr0.001 | 449 | 3 | 16.437 ± 1.281 | yes |
| B3A_rbf_N16_lr0.001 | 1217 | 3 | 17.329 ± 0.618 |  |
| B3A_rbf_N4_lr0.003 | 449 | 3 | 19.016 ± 1.181 |  |
| B3A_rbf_N4_lr0.01 | 449 | 3 | 19.086 ± 3.260 |  |
| B3A_rbf_N16_lr0.003 | 1217 | 3 | 19.691 ± 0.670 |  |
| B3A_rbf_N16_lr0.01 | 1217 | 3 | 20.768 ± 2.840 |  |

## B1_lstm16: selected `B1_lstm16_lr0.01` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| B1_lstm16_lr0.01 | 1873 | 3 | 15.583 ± 1.533 | yes |
| B1_lstm16_lr0.003 | 1873 | 3 | 17.430 ± 0.157 |  |
| B1_lstm16_lr0.001 | 1873 | 3 | 17.915 ± 0.545 |  |

## B2_lstm3: selected `B2_lstm3_lr0.01` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| B2_lstm3_lr0.01 | 196 | 3 | 16.466 ± 2.389 | yes |
| B2_lstm3_lr0.003 | 196 | 3 | 17.467 ± 1.526 |  |
| B2_lstm3_lr0.001 | 196 | 3 | 17.518 ± 1.668 |  |

## B0b_ridge: selected `B0b_ridge_alpha1.0` (COMPLETE)

| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |
|---|---:|---:|---:|---|
| B0b_ridge_alpha0.001 | 34 | 3 | 14.318 ± 0.000 | yes |
| B0b_ridge_alpha0.01 | 34 | 3 | 14.319 ± 0.000 | yes |
| B0b_ridge_alpha0.1 | 34 | 3 | 14.329 ± 0.000 | yes |
| B0b_ridge_alpha1.0 | 34 | 3 | 14.425 ± 0.000 | yes |
| B0b_ridge_alpha10.0 | 34 | 3 | 15.266 ± 0.000 |  |

## Diagnostics (not selected; validation split, seeds 0-2)

| Candidate | Params | Val RMSE mean ± std |
|---|---:|---:|
| ANG_QB_arctan_lr0.001 | 209 | 15.901 ± 0.339 |
| D2_QB_figure_lr0.001 | 209 | 16.120 ± 0.694 |
| ANG_QB_tanhpi_lr0.001 | 209 | 16.159 ± 1.834 |
| D2_QA_figure_N4_lr0.001 | 449 | 16.463 ± 1.448 |
| B0_persistence_val | 0 | 17.133 ± 0.000 |
