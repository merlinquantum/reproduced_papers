# Sweep `e2_merlin`

Phase 4 MerLin adaptation, E2: selected photonic classifier on the 4-class task, non-federated and 4 clients, 5 seeds

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: configuration fixed by the calibration sweep, no candidate is selected.
- Coverage: **COMPLETE SWEEP**, 10 of 10 planned runs completed (2026-10-28T15:57:26Z to 2026-10-28T16:19:06Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `fourclass_centralized` | 148 | 5 / 5 | 86.53 | 1.99 | 83.32, 85.87, 87.71, 88.08, 87.69 | 86.53 |
| `fourclass_clients4` | 148 | 5 / 5 | 83.81 | 1.52 | 86.20, 83.81, 84.09, 82.47, 82.50 | 83.81 |
