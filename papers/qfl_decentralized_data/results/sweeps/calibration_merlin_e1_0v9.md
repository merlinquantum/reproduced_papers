# Sweep `calibration_merlin_e1_0v9`

Phase 4 MerLin adaptation calibration: superposition placement, encoding scale and photon number on task 0 vs 9, non-federated, validation split (plan in LOG.md, Phase 4 Adaptation Plan)

- Declared rule: metric `validation_accuracy`, direction maximize, split validation, checkpoint last, aggregation mean over seeds [1, 2, 3], tie tolerance 0.005; tie handling: fewer trainable parameters first (replace before front), then 4 photons before 2, then the encoding scale closest to 1.0.
- Coverage: **COMPLETE SWEEP**, 36 of 36 planned runs completed (2026-10-28T14:59:59Z to 2026-10-28T15:17:43Z).
- Highest mean under the declared rule: `superpositionreplace_encoding_scale1.0_n_photons2` (no tie within 0.005).

| Candidate | Parameters | Successful / expected runs | Mean validation_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `superpositionfront_encoding_scale0.25_n_photons2` | 186 | 3 / 3 | 92.05 | 2.18 | 93.01, 93.60, 89.55 | 92.88 |
| `superpositionfront_encoding_scale0.25_n_photons4` | 186 | 3 / 3 | 88.65 | 5.31 | 83.49, 94.10, 88.37 | 89.79 |
| `superpositionfront_encoding_scale0.5_n_photons2` | 186 | 3 / 3 | 85.43 | 18.02 | 96.12, 95.53, 64.62 | 87.01 |
| `superpositionfront_encoding_scale0.5_n_photons4` | 186 | 3 / 3 | 94.41 | 1.28 | 93.26, 94.19, 95.79 | 95.17 |
| `superpositionfront_encoding_scale1.0_n_photons2` | 186 | 3 / 3 | 91.74 | 8.19 | 96.97, 95.96, 82.31 | 92.21 |
| `superpositionfront_encoding_scale1.0_n_photons4` | 186 | 3 / 3 | 95.79 | 1.17 | 94.44, 96.55, 96.38 | 96.01 |
| `superpositionreplace_encoding_scale0.25_n_photons2` | 130 | 3 / 3 | 87.19 | 10.08 | 75.57, 93.43, 92.59 | 88.59 |
| `superpositionreplace_encoding_scale0.25_n_photons4` | 130 | 3 / 3 | 91.13 | 2.00 | 89.89, 93.43, 90.06 | 91.94 |
| `superpositionreplace_encoding_scale0.5_n_photons2` | 130 | 3 / 3 | 88.88 | 12.34 | 74.64, 96.55, 95.45 | 90.13 |
| `superpositionreplace_encoding_scale0.5_n_photons4` | 130 | 3 / 3 | 94.95 | 3.42 | 91.07, 96.21, 97.56 | 94.94 |
| `superpositionreplace_encoding_scale1.0_n_photons2` | 130 | 3 / 3 | 96.91 | 0.18 | 97.05, 96.71, 96.97 | 97.08 |
| `superpositionreplace_encoding_scale1.0_n_photons4` | 130 | 3 / 3 | 95.87 | 0.15 | 96.04, 95.79, 95.79 | 96.26 |
