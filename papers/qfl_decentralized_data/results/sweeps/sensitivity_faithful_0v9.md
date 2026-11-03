# Sweep `sensitivity_faithful_0v9`

Sensitivity: paper-faithful root-qubit read-out VQTN (2 layers) on task 0 vs 9, non-federated plus 2, 4, 8, 16 clients

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: hyperparameters are fixed in advance, no candidate is selected; every mean +- std over 5 seeds is reported.
- Coverage: **COMPLETE SWEEP**, 25 of 25 planned runs completed (2026-10-28T13:52:04Z to 2026-10-28T14:11:55Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `task0v9_centralized` | 42 | 5 / 5 | 89.79 | 2.13 | 86.63, 91.80, 88.94, 91.60, 89.99 | 89.79 |
| `task0v9_clients2` | 42 | 5 / 5 | 91.77 | 1.06 | 92.11, 92.21, 91.55, 90.10, 92.91 | 91.77 |
| `task0v9_clients4` | 42 | 5 / 5 | 90.96 | 2.64 | 92.26, 86.38, 92.31, 91.10, 92.76 | 90.96 |
| `task0v9_clients8` | 42 | 5 / 5 | 90.63 | 1.18 | 90.15, 88.94, 91.60, 91.86, 90.60 | 90.63 |
| `task0v9_clients16` | 42 | 5 / 5 | 89.80 | 1.98 | 89.79, 91.00, 90.20, 86.48, 91.55 | 89.80 |
