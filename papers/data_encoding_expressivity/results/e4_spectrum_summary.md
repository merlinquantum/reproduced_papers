Run `outdir/run_20261028-100025`: 19/19 checks passed; largest coefficient outside a predicted spectrum 8.38e-16; FFT tolerance 1e-09; 5 random draws per check.

| Check | Predicted spectrum | Observed support = prediction | Max outside | Passed |
|---|---|---|---:|---|
| pauli_serial_r1 | 3 frequencies, max 1 | True | 1.1e-16 | True |
| pauli_parallel_r1 | 3 frequencies, max 1 | True | 1.2e-16 | True |
| pauli_serial_r2 | 5 frequencies, max 2 | True | 9.6e-17 | True |
| pauli_parallel_r2 | 5 frequencies, max 2 | True | 1.3e-16 | True |
| pauli_serial_r3 | 7 frequencies, max 3 | True | 1.4e-16 | True |
| pauli_parallel_r3 | 7 frequencies, max 3 | True | 7.9e-17 | True |
| pauli_serial_r4 | 9 frequencies, max 4 | True | 2.0e-16 | True |
| pauli_parallel_r4 | 9 frequencies, max 4 | True | 8.4e-17 | True |
| pauli_serial_r5 | 11 frequencies, max 5 | True | 1.6e-16 | True |
| pauli_parallel_r5 | 11 frequencies, max 5 | True | 6.4e-17 | True |
| dense_qubit_L1 | 3 frequencies, max 1 | True | 1.7e-16 | True |
| dense_qubit_L4 | 9 frequencies, max 4 | True | 3.8e-16 | True |
| dense_d3_L1 | 7 frequencies, max 3 | True | 2.0e-16 | True |
| dense_d3_L2_rotated | 13 frequencies, max 6 | True | 7.7e-16 | True |
| dense_d4_L1_golomb | 13 frequencies, max 6 | True | 8.4e-16 | True |
| dense_d4_L2 | 25 frequencies, max 12 | True | 3.2e-16 | True |
| dense_d4_L1_degenerate | 3 frequencies, max 1 | True | 4.1e-16 | True |
| multivariate_r1x2 | 15 frequency pairs | True | 1.5e-16 | True |
| non_integer_w0_0.4 | 13 frequencies, max 6 | True | 2.5e-16 | True |

| Dense case | d | L | K | Paper bound d^(2L)/2 − 1 | Distinct-sum bound m(m−1)/2 |
|---|---:|---:|---:|---:|---:|
| qubit_L1 | 2 | 1 | 1 | 1 | 1 |
| qubit_L4 | 2 | 4 | 4 | 127 | 120 |
| d3_L1 | 3 | 1 | 3 | 3.5 | 3 |
| d3_L2_rotated | 3 | 2 | 6 | 39.5 | 36 |
| d4_L1_golomb | 4 | 1 | 6 | 7 | 6 |
| d4_L2 | 4 | 2 | 12 | 127 | 120 |
| d4_L1_degenerate | 4 | 1 | 1 | 7 | 6 |
