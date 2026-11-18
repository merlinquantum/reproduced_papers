# Sweep `diagnose_vqtn_known_answer_zhead`

Separate an implementation bug from a read-out capacity limit: repeat the easy 0 vs 1 known-answer task with the diagnostic read-out head (amendment written after the components sweep showed the known-answer failure)

- Declared rule: metric `validation_accuracy`, direction maximize, split validation, checkpoint last, aggregation mean over seeds [1, 2, 3], tie tolerance 0.005; tie handling: diagnosis only: no candidate is selected; all means are reported against the baseline.
- Coverage: **COMPLETE SWEEP**, 3 of 3 planned runs completed (2026-10-28T11:45:46Z to 2026-10-28T11:47:46Z).
- Diagnosis sweep: no candidate is selected; highest observed mean: `known_answer_0v1_z_head`.

| Candidate | Parameters | Successful / expected runs | Mean validation_accuracy (%) | Std (points) | Per-seed (%) | Mean test accuracy (%) |
|---|---:|---:|---:|---:|---|---:|
| `known_answer_0v1_z_head` | 60 | 3 / 3 | 94.52 | 0.09 | 94.63, 94.47, 94.47 | 94.58 |
