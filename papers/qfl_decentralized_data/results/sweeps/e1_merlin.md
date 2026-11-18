# Sweep `e1_merlin`

Phase 4 MerLin adaptation, E1: selected photonic classifier on three binary tasks, non-federated and 2, 4, 8, 16 clients, 5 seeds

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: configuration fixed by the calibration sweep, no candidate is selected.
- Coverage: **COMPLETE SWEEP**, 75 of 75 planned runs completed (2026-10-28T15:18:37Z to 2026-10-28T15:56:47Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `task0v9_centralized` | 130 | 5 / 5 | 96.38 | 0.93 | 94.82, 97.13, 96.88, 96.78, 96.28 | 96.38 |
| `task0v9_clients2` | 130 | 5 / 5 | 96.44 | 0.77 | 96.68, 96.93, 95.32, 97.23, 96.03 | 96.44 |
| `task0v9_clients4` | 130 | 5 / 5 | 96.34 | 0.36 | 96.73, 96.08, 96.73, 96.08, 96.08 | 96.34 |
| `task0v9_clients8` | 130 | 5 / 5 | 96.28 | 0.52 | 95.53, 96.68, 96.03, 96.33, 96.83 | 96.28 |
| `task0v9_clients16` | 130 | 5 / 5 | 96.25 | 0.23 | 96.23, 96.18, 96.03, 96.63, 96.18 | 96.25 |
| `task2v7_centralized` | 130 | 5 / 5 | 96.03 | 0.87 | 95.10, 96.41, 97.09, 96.41, 95.15 | 96.03 |
| `task2v7_clients2` | 130 | 5 / 5 | 96.55 | 0.66 | 97.18, 96.36, 95.58, 97.14, 96.50 | 96.55 |
| `task2v7_clients4` | 130 | 5 / 5 | 96.39 | 0.38 | 95.97, 96.36, 96.80, 96.07, 96.75 | 96.39 |
| `task2v7_clients8` | 130 | 5 / 5 | 96.13 | 0.23 | 95.83, 95.97, 96.17, 96.41, 96.26 | 96.13 |
| `task2v7_clients16` | 130 | 5 / 5 | 95.73 | 0.33 | 95.92, 95.87, 95.15, 95.78, 95.92 | 95.73 |
| `task3v6_centralized` | 130 | 5 / 5 | 98.39 | 0.38 | 98.42, 98.73, 98.78, 97.92, 98.12 | 98.39 |
| `task3v6_clients2` | 130 | 5 / 5 | 98.55 | 0.42 | 98.83, 98.73, 98.88, 97.87, 98.42 | 98.55 |
| `task3v6_clients4` | 130 | 5 / 5 | 98.33 | 0.66 | 98.93, 97.82, 98.42, 97.51, 98.98 | 98.33 |
| `task3v6_clients8` | 130 | 5 / 5 | 98.34 | 0.28 | 98.17, 98.27, 98.63, 98.63, 98.02 | 98.34 |
| `task3v6_clients16` | 130 | 5 / 5 | 97.31 | 1.13 | 97.51, 96.24, 96.04, 98.27, 98.48 | 97.31 |
