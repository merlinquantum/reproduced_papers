# Sweep `sensitivity_merlin_lexgroup_0v9`

Phase 4 sensitivity: selected photonic circuit with a fixed LexGrouping read-out (no classical layer), task 0 vs 9, non-federated and 2, 4, 8, 16 clients, 5 seeds

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: configuration fixed by the calibration sweep, no candidate is selected.
- Coverage: **COMPLETE SWEEP**, 25 of 25 planned runs completed (2026-10-28T16:19:29Z to 2026-10-28T16:43:09Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `task0v9_centralized` | 112 | 5 / 5 | 94.73 | 0.88 | 95.78, 93.41, 95.22, 94.62, 94.62 | 94.73 |
| `task0v9_clients2` | 112 | 5 / 5 | 93.48 | 1.76 | 90.80, 94.57, 92.91, 93.77, 95.37 | 93.48 |
| `task0v9_clients4` | 112 | 5 / 5 | 93.68 | 1.61 | 92.41, 94.97, 94.92, 91.50, 94.57 | 93.68 |
| `task0v9_clients8` | 112 | 5 / 5 | 94.09 | 0.82 | 93.61, 93.77, 93.36, 94.27, 95.42 | 94.09 |
| `task0v9_clients16` | 112 | 5 / 5 | 94.35 | 1.13 | 94.47, 96.03, 94.52, 92.96, 93.77 | 94.35 |
