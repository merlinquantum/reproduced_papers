# Sweep `control_classical_e1`

Fair baseline for E1: classical 16-3-2 tanh MLP (59 parameters vs 60 for the hybrid VQTN), same data, partitions, rounds, Adam lr 0.03, B=20

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: hyperparameters are fixed in advance, no candidate is selected; every mean +- std over 5 seeds is reported.
- Coverage: **COMPLETE SWEEP**, 75 of 75 planned runs completed (2026-10-28T13:31:48Z to 2026-10-28T13:35:29Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `task0v9_centralized` | 59 | 5 / 5 | 97.93 | 0.16 | 98.14, 97.89, 98.04, 97.74, 97.84 | 97.93 |
| `task0v9_clients2` | 59 | 5 / 5 | 97.66 | 0.38 | 97.79, 97.94, 97.84, 96.98, 97.74 | 97.66 |
| `task0v9_clients4` | 59 | 5 / 5 | 97.56 | 0.38 | 97.89, 97.64, 97.84, 96.93, 97.49 | 97.56 |
| `task0v9_clients8` | 59 | 5 / 5 | 97.14 | 0.07 | 97.13, 97.23, 97.03, 97.13, 97.18 | 97.14 |
| `task0v9_clients16` | 59 | 5 / 5 | 96.96 | 0.11 | 96.98, 97.13, 96.93, 96.93, 96.83 | 96.96 |
| `task2v7_centralized` | 59 | 5 / 5 | 97.70 | 0.27 | 97.52, 97.72, 98.16, 97.57, 97.52 | 97.70 |
| `task2v7_clients2` | 59 | 5 / 5 | 97.64 | 0.13 | 97.86, 97.52, 97.57, 97.62, 97.62 | 97.64 |
| `task2v7_clients4` | 59 | 5 / 5 | 97.06 | 0.30 | 96.84, 96.75, 97.38, 97.38, 96.94 | 97.06 |
| `task2v7_clients8` | 59 | 5 / 5 | 96.79 | 0.16 | 97.04, 96.70, 96.65, 96.70, 96.84 | 96.79 |
| `task2v7_clients16` | 59 | 5 / 5 | 96.57 | 0.16 | 96.41, 96.75, 96.46, 96.50, 96.75 | 96.57 |
| `task3v6_centralized` | 59 | 5 / 5 | 99.24 | 0.13 | 99.24, 99.29, 99.24, 99.03, 99.39 | 99.24 |
| `task3v6_clients2` | 59 | 5 / 5 | 99.34 | 0.16 | 99.19, 99.24, 99.24, 99.49, 99.54 | 99.34 |
| `task3v6_clients4` | 59 | 5 / 5 | 99.08 | 0.25 | 99.14, 99.19, 99.24, 98.63, 99.19 | 99.08 |
| `task3v6_clients8` | 59 | 5 / 5 | 99.01 | 0.22 | 99.09, 99.14, 99.09, 98.63, 99.14 | 99.01 |
| `task3v6_clients16` | 59 | 5 / 5 | 98.72 | 0.08 | 98.78, 98.58, 98.73, 98.73, 98.78 | 98.72 |
