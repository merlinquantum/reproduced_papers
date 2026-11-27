# Sweep `calibration_vqtn_e1_0v9`

Calibrate the inferred VQTN (depth, learning rate, init scale) on E1 task 0 vs 9 using validation accuracy of the non-federated model, to close the gap to the paper's 94.03% centralised row

- Declared rule: metric `validation_accuracy`, direction maximize, split validation, checkpoint last, aggregation mean over seeds [1, 2, 3], tie tolerance 0.005; tie handling: report all tied candidates; for downstream runs use the tied candidate with the fewest parameters (smallest model.layers), then the lowest training.lr.
- Coverage: **COMPLETE SWEEP**, 36 of 36 planned runs completed (2026-10-27T17:23:09Z to 2026-10-27T17:44:20Z).
- Highest mean under the declared rule: `layers2_lr0.03_init_std1.0`, `layers2_lr0.03_init_std0.1`, `layers3_lr0.03_init_std0.1` (tie within 0.005).

| Candidate | Parameters | Successful / expected runs | Mean validation_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `layers1_lr0.03_init_std0.1` | 21 | 3 / 3 | 83.46 | 3.68 | 82.48, 87.53, 80.37 | 84.21 |
| `layers1_lr0.03_init_std1.0` | 21 | 3 / 3 | 83.46 | 3.74 | 82.39, 87.62, 80.37 | 84.25 |
| `layers1_lr0.1_init_std0.1` | 21 | 3 / 3 | 73.52 | 6.03 | 79.44, 67.40, 73.72 | 73.76 |
| `layers1_lr0.1_init_std1.0` | 21 | 3 / 3 | 75.37 | 7.04 | 81.89, 67.90, 76.33 | 76.55 |
| `layers2_lr0.03_init_std0.1` | 42 | 3 / 3 | 87.08 | 5.51 | 91.07, 89.39, 80.79 | 87.83 |
| `layers2_lr0.03_init_std1.0` | 42 | 3 / 3 | 87.28 | 6.13 | 90.99, 90.65, 80.20 | 87.88 |
| `layers2_lr0.1_init_std0.1` | 42 | 3 / 3 | 85.48 | 6.19 | 84.33, 92.17, 79.95 | 85.49 |
| `layers2_lr0.1_init_std1.0` | 42 | 3 / 3 | 84.86 | 8.09 | 84.75, 93.01, 76.83 | 84.75 |
| `layers3_lr0.03_init_std0.1` | 63 | 3 / 3 | 86.97 | 5.62 | 90.90, 89.47, 80.54 | 87.48 |
| `layers3_lr0.03_init_std1.0` | 63 | 3 / 3 | 86.55 | 3.52 | 87.53, 89.47, 82.65 | 86.63 |
| `layers3_lr0.1_init_std0.1` | 63 | 3 / 3 | 84.72 | 10.11 | 88.37, 92.50, 73.29 | 84.58 |
| `layers3_lr0.1_init_std1.0` | 63 | 3 / 3 | 83.09 | 7.39 | 83.74, 90.14, 75.40 | 83.76 |
