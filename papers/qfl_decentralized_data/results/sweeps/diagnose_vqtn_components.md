# Sweep `diagnose_vqtn_components`

One-at-a-time diagnosis of the VQTN gap (about 87% validation accuracy vs the paper's 94%) on the non-federated model, task 0 vs 9, against the calibrated baseline

- Declared rule: metric `validation_accuracy`, direction maximize, split validation, checkpoint last, aggregation mean over seeds [1, 2, 3], tie tolerance 0.005; tie handling: diagnosis only: no candidate is selected; all means are reported against the baseline.
- Coverage: **COMPLETE SWEEP**, 24 of 24 planned runs completed (2026-10-28T11:27:29Z to 2026-10-28T11:44:38Z).
- Diagnosis sweep: no candidate is selected; highest observed mean: `readout_z_head`.

| Candidate | Parameters | Successful / expected runs | Mean validation_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `baseline` | 42 | 3 / 3 | 87.08 | 5.51 | 91.07, 89.39, 80.79 | 87.83 |
| `encoding_scale_0.5` | 42 | 3 / 3 | 75.79 | 3.75 | 78.85, 71.61, 76.92 | 76.05 |
| `encoding_scale_2.0` | 42 | 3 / 3 | 88.12 | 2.15 | 88.96, 89.72, 85.68 | 87.65 |
| `encoding_gate_rx` | 42 | 3 / 3 | 86.30 | 3.84 | 88.88, 88.12, 81.89 | 87.23 |
| `feature_minmax` | 42 | 3 / 3 | 90.59 | 1.94 | 90.48, 92.59, 88.71 | 90.20 |
| `readout_z_head` | 60 | 3 / 3 | 93.82 | 1.52 | 94.86, 94.52, 92.08 | 93.55 |
| `epochs_60` | 42 | 3 / 3 | 88.29 | 2.83 | 88.54, 85.34, 90.99 | 88.19 |
| `known_answer_0v1` | 42 | 3 / 3 | 86.33 | 3.72 | 90.52, 85.07, 83.41 | 84.96 |
