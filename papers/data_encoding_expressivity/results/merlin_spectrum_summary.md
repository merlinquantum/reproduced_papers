Run `outdir/run_20261028-110318` (MerLin adaptation): 9/9 checks behave as predicted; 5 random draws, 128-point grid, FFT tolerance 1e-09. FOCK cases must be band-limited to the photon-number prediction; UNBUNCHED (postselected, renormalised) is expected not to be.

| Check | Space | n | Modes | Encoded modes | L | Predicted degree | Observed degree | Max outside | Passed |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| photonic_fock_n1_single_mode | fock | 1 | 2 | [0] | 1 | 1 | 1 | 1.1e-16 | True |
| photonic_fock_n2_single_mode | fock | 2 | 3 | [0] | 1 | 2 | 2 | 9.3e-17 | True |
| photonic_fock_n3_single_mode | fock | 3 | 4 | [0] | 1 | 3 | 3 | 1.1e-16 | True |
| photonic_fock_n4_single_mode | fock | 4 | 5 | [0] | 1 | 4 | 4 | 1.1e-16 | True |
| photonic_fock_n2_reupload_L2 | fock | 2 | 3 | [0] | 2 | 4 | 4 | 9.1e-17 | True |
| photonic_fock_n2_reupload_L3 | fock | 2 | 3 | [0] | 3 | 6 | 6 | 1.0e-16 | True |
| photonic_fock_n2_two_of_four_modes | fock | 2 | 4 | [0, 1] | 1 | 2 | 2 | 1.4e-16 | True |
| photonic_fock_n2_all_modes_global_phase | fock | 2 | 3 | [0, 1, 2] | 1 | 0 | 0 | 7.3e-17 | True |
| photonic_unbunched_n2_single_mode | unbunched | 2 | 3 | [0] | 1 | 2 | 63 | 1.3e-01 | True |
