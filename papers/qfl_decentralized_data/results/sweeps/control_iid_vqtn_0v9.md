# Sweep `control_iid_vqtn_0v9`

Control: hybrid VQTN on task 0 vs 9 with an i.i.d. equal-size partition at 2 and 16 clients, to show what the non-i.i.d. split costs

- Declared rule: metric `test_accuracy`, direction maximize, split test, checkpoint last, aggregation mean over seeds [1, 2, 3, 4, 5], tie tolerance 0.005; tie handling: reporting runs: hyperparameters are fixed in advance, no candidate is selected; every mean +- std over 5 seeds is reported.
- Coverage: **COMPLETE SWEEP**, 10 of 10 planned runs completed (2026-10-28T13:36:49Z to 2026-10-28T13:51:45Z).
- Reporting sweep: no candidate is selected (configuration fixed in advance).

| Candidate | Parameters | Successful / expected runs | Mean test_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `task0v9_clients2_iid` | 60 | 5 / 5 | 94.74 | 0.39 | 95.12, 94.87, 94.12, 94.62, 94.97 | 94.74 |
| `task0v9_clients16_iid` | 60 | 5 / 5 | 94.25 | 0.08 | 94.32, 94.27, 94.12, 94.32, 94.22 | 94.25 |
