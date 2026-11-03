# Sweep `control_classical_e2`

Fair baseline for E2: classical 16-4-4 tanh MLP (88 parameters vs 78 for the hybrid VQTN), same data and settings

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: hyperparameters are fixed in advance, no candidate is selected; every mean +- std over 5 seeds is reported.
- Coverage: **COMPLETE SWEEP**, 10 of 10 planned runs completed (2026-10-28T13:35:46Z to 2026-10-28T13:36:33Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `fourclass_centralized` | 88 | 5 / 5 | 90.74 | 0.60 | 91.14, 90.96, 91.04, 89.67, 90.89 | 90.74 |
| `fourclass_clients4` | 88 | 5 / 5 | 89.85 | 1.04 | 90.76, 88.18, 89.52, 90.54, 90.22 | 89.85 |
