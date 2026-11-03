Run `outdir/run_20261028-110524` (diagnostic): largest |c_k| found by maximisation (Adam lr 0.1, 500 steps, 10 random inits). A lower bound on the reachable value, and only a necessary condition for fitting: the other coefficients must also match.

| Case | Kind | Model degree | k | Best \|c_k\| | Median \|c_k\| | Target \|c_k\| | Reached |
|---|---|---:|---:|---:|---:|---:|---|
| photonic_n5_L1_k5 | photonic | 5 | 5 | 0.0024 | 0.0024 | 0.0707 | False |
| photonic_n2_L3_k5 | photonic | 6 | 5 | 0.2009 | 0.2009 | 0.0707 | True |
| photonic_n2_L1_k2_fig3c | photonic | 2 | 2 | 0.2500 | 0.2500 | 0.2121 | True |
| qubit_serial_r5_k5 | quantum | 5 | 5 | 0.5000 | 0.5000 | 0.0707 | True |
| qubit_parallel_r5_k5 | quantum | 5 | 5 | 0.5000 | 0.5000 | 0.0707 | True |
