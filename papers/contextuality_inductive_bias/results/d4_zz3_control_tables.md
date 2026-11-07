### d4_zz3_control: candidates (final test KL is the primary metric)

| Protocol | Model | L2 | Steps | Train data | n | Final test KL (mean +- std) | Min test KL (mean) | Final train NLL (mean) |
|---|---|---:|---:|---|---:|---|---:|---:|
| code | quantum (biased) L=1 | 0 | 2000 | author | 5 | 4.726e-03 +- 1.6e-04 | 4.715e-03 | 0.6837 |
| code | quantum (biased) L=2 | 0 | 2000 | author | 5 | 4.660e-03 +- 1.3e-04 | 4.582e-03 | 0.6810 |
| paper | quantum (biased) L=1 | 0 | 2000 | author | 5 | 4.063e-03 +- 3.6e-04 | 3.957e-03 | 0.6837 |
| paper | quantum (biased) L=2 | 0 | 2000 | author | 5 | 5.144e-03 +- 3.5e-04 | 4.589e-03 | 0.6810 |

### d4_zz3_control: reference models vs other model classes (final test KL, two-sided)

| Protocol | Reference | Other | n | Diff (reference - other) | 95% CI (Welch) | Welch p | Mann-Whitney p |
|---|---|---|---|---:|---|---:|---:|
| code | quantum (biased) L=1 | quantum (biased) L=2 | 5 vs 5 | 6.53e-05 | [-1.49e-04, 2.79e-04] | 5.0e-01 | 5.5e-01 |
| code | quantum (biased) L=2 | quantum (biased) L=1 | 5 vs 5 | -6.53e-05 | [-2.79e-04, 1.49e-04] | 5.0e-01 | 5.5e-01 |
| paper | quantum (biased) L=1 | quantum (biased) L=2 | 5 vs 5 | -1.08e-03 | [-1.60e-03, -5.61e-04] | 1.4e-03 | 7.9e-03 |
| paper | quantum (biased) L=2 | quantum (biased) L=1 | 5 vs 5 | 1.08e-03 | [5.61e-04, 1.60e-03] | 1.4e-03 | 7.9e-03 |
