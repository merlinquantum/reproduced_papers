# Sweep `e1_vqtn_hybrid`

E1 primary: hybrid VQTN (2 layers, eight-<Z> head) on three binary MNIST tasks, non-federated plus 2, 4, 8, 16 clients, 5 seeds, paper settings otherwise (B=20, E=1, 20 rounds)

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: hyperparameters are fixed in advance, no candidate is selected; every mean +- std over 5 seeds is reported.
- Coverage: **COMPLETE SWEEP**, 75 of 75 planned runs completed (2026-10-28T12:13:36Z to 2026-10-28T13:03:29Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `task0v9_centralized` | 60 | 5 / 5 | 94.60 | 0.17 | 94.72, 94.57, 94.72, 94.32, 94.67 | 94.60 |
| `task0v9_clients2` | 60 | 5 / 5 | 94.13 | 0.80 | 94.92, 94.02, 92.86, 94.67, 94.17 | 94.13 |
| `task0v9_clients4` | 60 | 5 / 5 | 94.51 | 0.55 | 94.12, 95.02, 94.97, 93.77, 94.67 | 94.51 |
| `task0v9_clients8` | 60 | 5 / 5 | 94.40 | 0.24 | 94.42, 94.32, 94.37, 94.77, 94.12 | 94.40 |
| `task0v9_clients16` | 60 | 5 / 5 | 94.32 | 0.20 | 94.47, 94.42, 94.22, 94.47, 94.02 | 94.32 |
| `task2v7_centralized` | 60 | 5 / 5 | 95.61 | 0.30 | 95.24, 95.97, 95.49, 95.87, 95.49 | 95.61 |
| `task2v7_clients2` | 60 | 5 / 5 | 95.37 | 0.58 | 94.76, 96.12, 95.00, 95.83, 95.15 | 95.37 |
| `task2v7_clients4` | 60 | 5 / 5 | 95.45 | 0.60 | 95.34, 96.02, 95.44, 94.51, 95.92 | 95.45 |
| `task2v7_clients8` | 60 | 5 / 5 | 95.51 | 0.43 | 95.63, 94.76, 95.63, 95.78, 95.78 | 95.51 |
| `task2v7_clients16` | 60 | 5 / 5 | 95.48 | 0.25 | 95.44, 95.78, 95.19, 95.29, 95.68 | 95.48 |
| `task3v6_centralized` | 60 | 5 / 5 | 94.82 | 1.27 | 94.87, 92.73, 95.68, 95.99, 94.82 | 94.82 |
| `task3v6_clients2` | 60 | 5 / 5 | 94.68 | 1.20 | 93.55, 93.70, 96.29, 95.58, 94.31 | 94.68 |
| `task3v6_clients4` | 60 | 5 / 5 | 94.91 | 0.70 | 95.38, 93.95, 94.36, 95.43, 95.43 | 94.91 |
| `task3v6_clients8` | 60 | 5 / 5 | 95.02 | 0.32 | 95.02, 95.48, 95.02, 95.02, 94.56 | 95.02 |
| `task3v6_clients16` | 60 | 5 / 5 | 94.12 | 0.23 | 94.51, 94.05, 93.95, 94.05, 94.00 | 94.12 |
