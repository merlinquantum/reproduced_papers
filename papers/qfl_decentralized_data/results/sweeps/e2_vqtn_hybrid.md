# Sweep `e2_vqtn_hybrid`

E2 primary: hybrid VQTN on the 4-class task (2, 3, 6, 7), non-federated and 4 clients, 5 seeds

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: hyperparameters are fixed in advance, no candidate is selected; every mean +- std over 5 seeds is reported.
- Coverage: **COMPLETE SWEEP**, 10 of 10 planned runs completed (2026-10-28T13:04:44Z to 2026-10-28T13:31:01Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `fourclass_centralized` | 78 | 5 / 5 | 76.16 | 2.35 | 75.05, 78.50, 72.62, 77.16, 77.48 | 76.16 |
| `fourclass_clients4` | 78 | 5 / 5 | 73.73 | 0.49 | 73.54, 73.24, 73.58, 73.73, 74.55 | 73.73 |
