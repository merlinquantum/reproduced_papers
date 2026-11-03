### merlin_vs_phase3: candidates (final test KL is the primary metric)

| Protocol | Model | L2 | Steps | Train data | n | Final test KL (mean +- std) | Min test KL (mean) | Final train NLL (mean) |
|---|---|---:|---:|---|---:|---|---:|---:|
| code | surrogate L=1 | 0 | 2000 | author | 5 | 7.481e-03 +- 1.6e-05 | 6.231e-03 | 0.6765 |
| code | surrogate L=2 | 0 | 2000 | author | 5 | 1.271e-02 +- 3.0e-06 | 6.945e-03 | 0.6588 |
| fork | surrogate L=1 | 0 | 2000 | author | 5 | 4.276e-03 +- 5.5e-06 | 4.276e-03 | 0.6824 |
| fork | surrogate L=2 | 0 | 2000 | author | 5 | 6.965e-03 +- 1.9e-06 | 4.174e-03 | 0.6752 |
| paper | quantum (biased) L=1 | 0 | 2000 | author | 20 | 4.080e-03 +- 3.1e-04 | 4.026e-03 | 0.6839 |
| paper | quantum (biased) L=2 | 0 | 2000 | author | 20 | 5.144e-03 +- 5.7e-04 | 4.675e-03 | 0.6811 |
| paper | quantum (no bias, code CNOTs) L=1 | 0 | 2000 | author | 20 | 7.231e-03 +- 7.9e-04 | 7.195e-03 | 0.6865 |
| paper | quantum (no bias, Fig. 10 CNOTs) L=1 | 0 | 2000 | author | 20 | 5.666e-03 +- 5.1e-04 | 5.635e-03 | 0.6836 |
| paper | quantum (no bias, code CNOTs) L=2 | 0 | 2000 | author | 20 | 7.693e-03 +- 5.7e-04 | 7.619e-03 | 0.6847 |
| paper | quantum (no bias, Fig. 10 CNOTs) L=2 | 0 | 2000 | author | 20 | 7.023e-03 +- 6.5e-04 | 6.784e-03 | 0.6794 |
| paper | photonic MerLin (group readout, PNR) N=2 | 0 | 2000 | author | 10 | 4.064e-03 +- 2.1e-04 | 4.018e-03 | 0.6836 |
| paper | photonic MerLin (group readout, threshold) N=2 | 0 | 2000 | author | 10 | 4.442e-03 +- 4.2e-04 | 4.285e-03 | 0.6832 |
| paper | photonic MerLin (group readout, threshold) N=3 | 0 | 2000 | author | 10 | 5.069e-03 +- 3.1e-04 | 4.869e-03 | 0.6831 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | 0 | 2000 | author | 10 | 5.895e-03 +- 9.2e-04 | 5.024e-03 | 0.6813 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | 0 | 2000 | author | 10 | 7.064e-03 +- 7.5e-04 | 5.932e-03 | 0.6806 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | 0 | 2000 | author | 10 | 4.132e-03 +- 3.1e-04 | 4.074e-03 | 0.6834 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | 0 | 2000 | author | 10 | 4.436e-03 +- 4.3e-04 | 4.261e-03 | 0.6828 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | 0 | 2000 | author | 10 | 5.652e-03 +- 9.0e-04 | 4.672e-03 | 0.6817 |
| paper | surrogate L=1 | 0 | 2000 | author | 5 | 5.448e-03 +- 5.7e-06 | 4.473e-03 | 0.6765 |
| paper | surrogate L=2 | 0 | 2000 | author | 5 | 1.865e-02 +- 3.5e-06 | 4.991e-03 | 0.6588 |

### merlin_vs_phase3: reference models vs other model classes (final test KL, two-sided)

| Protocol | Reference | Other | n | Diff (reference - other) | 95% CI (Welch) | Welch p | Mann-Whitney p |
|---|---|---|---|---:|---|---:|---:|
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | -5.80e-06 | [-4.02e-04, 3.90e-04] | 9.8e-01 | 7.9e-01 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | -1.46e-03 | [-2.15e-03, -7.67e-04] | 5.6e-04 | 1.0e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | -1.22e-03 | [-1.89e-03, -5.36e-04] | 2.0e-03 | 5.8e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | -6.33e-04 | [-9.83e-04, -2.82e-04] | 1.5e-03 | 2.8e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -2.63e-03 | [-3.21e-03, -2.05e-03] | 1.2e-07 | 1.8e-04 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | 3.04e-04 | [-4.71e-05, 6.55e-04] | 8.5e-02 | 4.5e-02 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 3.73e-04 | [4.87e-05, 6.97e-04] | 2.7e-02 | 3.1e-02 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | quantum (biased) L=1 | 10 vs 20 | 3.57e-04 | [3.18e-05, 6.81e-04] | 3.4e-02 | 1.3e-02 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | quantum (biased) L=2 | 10 vs 20 | -7.08e-04 | [-1.09e-03, -3.26e-04] | 8.4e-04 | 1.7e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | -1.23e-03 | [-1.60e-03, -8.63e-04] | 6.3e-07 | 5.7e-05 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -2.59e-03 | [-2.99e-03, -2.18e-03] | 7.5e-13 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -2.79e-03 | [-3.25e-03, -2.34e-03] | 5.2e-13 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -3.26e-03 | [-3.64e-03, -2.87e-03] | 5.2e-15 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | surrogate L=1 | 10 vs 5 | -1.01e-03 | [-1.32e-03, -7.07e-04] | 3.6e-05 | 6.7e-04 |
| paper | photonic MerLin (vertex readout, threshold) N=2 | surrogate L=2 | 10 vs 5 | -1.42e-02 | [-1.45e-02, -1.39e-02] | 3.1e-15 | 6.7e-04 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | 5.80e-06 | [-3.90e-04, 4.02e-04] | 9.8e-01 | 7.9e-01 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | -1.45e-03 | [-2.14e-03, -7.63e-04] | 5.8e-04 | 3.6e-03 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | -1.21e-03 | [-1.89e-03, -5.32e-04] | 2.0e-03 | 3.6e-03 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | -6.27e-04 | [-9.73e-04, -2.81e-04] | 1.4e-03 | 1.7e-03 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -2.62e-03 | [-3.20e-03, -2.04e-03] | 1.3e-07 | 1.8e-04 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | 3.10e-04 | [-3.66e-05, 6.56e-04] | 7.6e-02 | 1.2e-01 |
| paper | photonic MerLin (group readout, threshold) N=2 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 3.79e-04 | [5.98e-05, 6.97e-04] | 2.3e-02 | 5.4e-02 |
| paper | photonic MerLin (group readout, threshold) N=2 | quantum (biased) L=1 | 10 vs 20 | 3.62e-04 | [4.30e-05, 6.82e-04] | 2.9e-02 | 3.3e-02 |
| paper | photonic MerLin (group readout, threshold) N=2 | quantum (biased) L=2 | 10 vs 20 | -7.02e-04 | [-1.08e-03, -3.24e-04] | 8.0e-04 | 2.6e-03 |
| paper | photonic MerLin (group readout, threshold) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | -1.22e-03 | [-1.59e-03, -8.61e-04] | 5.2e-07 | 1.5e-05 |
| paper | photonic MerLin (group readout, threshold) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -2.58e-03 | [-2.98e-03, -2.18e-03] | 5.5e-13 | 1.2e-05 |
| paper | photonic MerLin (group readout, threshold) N=2 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -2.79e-03 | [-3.24e-03, -2.34e-03] | 4.4e-13 | 1.2e-05 |
| paper | photonic MerLin (group readout, threshold) N=2 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -3.25e-03 | [-3.63e-03, -2.87e-03] | 3.2e-15 | 1.2e-05 |
| paper | photonic MerLin (group readout, threshold) N=2 | surrogate L=1 | 10 vs 5 | -1.01e-03 | [-1.30e-03, -7.07e-04] | 3.2e-05 | 6.7e-04 |
| paper | photonic MerLin (group readout, threshold) N=2 | surrogate L=2 | 10 vs 5 | -1.42e-02 | [-1.45e-02, -1.39e-02] | 2.6e-15 | 6.7e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | 1.46e-03 | [7.67e-04, 2.15e-03] | 5.6e-04 | 1.0e-03 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | 1.45e-03 | [7.63e-04, 2.14e-03] | 5.8e-04 | 3.6e-03 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | 2.44e-04 | [-6.09e-04, 1.10e-03] | 5.6e-01 | 5.7e-01 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | 8.26e-04 | [1.53e-04, 1.50e-03] | 2.1e-02 | 4.5e-02 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -1.17e-03 | [-1.96e-03, -3.81e-04] | 6.0e-03 | 7.3e-03 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | 1.76e-03 | [1.09e-03, 2.44e-03] | 1.3e-04 | 3.3e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 1.83e-03 | [1.17e-03, 2.50e-03] | 1.1e-04 | 1.8e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | quantum (biased) L=1 | 10 vs 20 | 1.82e-03 | [1.15e-03, 2.48e-03] | 1.1e-04 | 1.8e-05 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | quantum (biased) L=2 | 10 vs 20 | 7.51e-04 | [6.50e-05, 1.44e-03] | 3.4e-02 | 5.0e-02 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | 2.29e-04 | [-4.50e-04, 9.09e-04] | 4.8e-01 | 3.7e-01 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -1.13e-03 | [-1.83e-03, -4.31e-04] | 3.8e-03 | 1.9e-03 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -1.34e-03 | [-2.06e-03, -6.16e-04] | 1.2e-03 | 4.7e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -1.80e-03 | [-2.48e-03, -1.11e-03] | 8.6e-05 | 2.7e-05 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | surrogate L=1 | 10 vs 5 | 4.48e-04 | [-2.08e-04, 1.10e-03] | 1.6e-01 | 2.5e-01 |
| paper | photonic MerLin (unbiased readout, threshold) N=2 | surrogate L=2 | 10 vs 5 | -1.28e-02 | [-1.34e-02, -1.21e-02] | 8.1e-12 | 6.7e-04 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | 1.22e-03 | [5.36e-04, 1.89e-03] | 2.0e-03 | 5.8e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | 1.21e-03 | [5.32e-04, 1.89e-03] | 2.0e-03 | 3.6e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | -2.44e-04 | [-1.10e-03, 6.09e-04] | 5.6e-01 | 5.7e-01 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | 5.82e-04 | [-7.72e-05, 1.24e-03] | 7.8e-02 | 8.9e-02 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -1.41e-03 | [-2.19e-03, -6.35e-04] | 1.3e-03 | 1.7e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | 1.52e-03 | [8.59e-04, 2.18e-03] | 3.6e-04 | 1.7e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 1.59e-03 | [9.38e-04, 2.24e-03] | 2.8e-04 | 1.0e-03 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | quantum (biased) L=1 | 10 vs 20 | 1.57e-03 | [9.21e-04, 2.22e-03] | 3.0e-04 | 1.4e-04 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | quantum (biased) L=2 | 10 vs 20 | 5.07e-04 | [-1.66e-04, 1.18e-03] | 1.3e-01 | 8.2e-02 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | -1.45e-05 | [-6.81e-04, 6.52e-04] | 9.6e-01 | 9.8e-01 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -1.37e-03 | [-2.06e-03, -6.87e-04] | 7.4e-04 | 2.4e-04 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -1.58e-03 | [-2.29e-03, -8.72e-04] | 2.2e-04 | 8.2e-05 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -2.04e-03 | [-2.72e-03, -1.37e-03] | 2.0e-05 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | surrogate L=1 | 10 vs 5 | 2.04e-04 | [-4.39e-04, 8.46e-04] | 4.9e-01 | 2.5e-01 |
| paper | photonic MerLin (vertex readout, threshold) N=3 | surrogate L=2 | 10 vs 5 | -1.30e-02 | [-1.36e-02, -1.24e-02] | 5.6e-12 | 6.7e-04 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | 6.33e-04 | [2.82e-04, 9.83e-04] | 1.5e-03 | 2.8e-03 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | 6.27e-04 | [2.81e-04, 9.73e-04] | 1.4e-03 | 1.7e-03 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | -8.26e-04 | [-1.50e-03, -1.53e-04] | 2.1e-02 | 4.5e-02 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | -5.82e-04 | [-1.24e-03, 7.72e-05] | 7.8e-02 | 8.9e-02 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -2.00e-03 | [-2.55e-03, -1.44e-03] | 4.9e-06 | 2.5e-04 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | 9.37e-04 | [6.49e-04, 1.22e-03] | 2.1e-06 | 3.3e-04 |
| paper | photonic MerLin (group readout, threshold) N=3 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 1.01e-03 | [7.57e-04, 1.25e-03] | 2.3e-07 | 1.8e-04 |
| paper | photonic MerLin (group readout, threshold) N=3 | quantum (biased) L=1 | 10 vs 20 | 9.90e-04 | [7.40e-04, 1.24e-03] | 1.2e-07 | 3.9e-05 |
| paper | photonic MerLin (group readout, threshold) N=3 | quantum (biased) L=2 | 10 vs 20 | -7.52e-05 | [-4.02e-04, 2.51e-04] | 6.4e-01 | 8.4e-01 |
| paper | photonic MerLin (group readout, threshold) N=3 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | -5.97e-04 | [-9.04e-04, -2.89e-04] | 4.6e-04 | 2.2e-03 |
| paper | photonic MerLin (group readout, threshold) N=3 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -1.95e-03 | [-2.31e-03, -1.60e-03] | 7.6e-12 | 1.5e-05 |
| paper | photonic MerLin (group readout, threshold) N=3 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -2.16e-03 | [-2.57e-03, -1.75e-03] | 2.8e-11 | 1.2e-05 |
| paper | photonic MerLin (group readout, threshold) N=3 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -2.62e-03 | [-2.95e-03, -2.30e-03] | 8.4e-16 | 1.2e-05 |
| paper | photonic MerLin (group readout, threshold) N=3 | surrogate L=1 | 10 vs 5 | -3.79e-04 | [-5.97e-04, -1.60e-04] | 3.5e-03 | 7.5e-02 |
| paper | photonic MerLin (group readout, threshold) N=3 | surrogate L=2 | 10 vs 5 | -1.36e-02 | [-1.38e-02, -1.34e-02] | 2.3e-16 | 6.7e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | 2.63e-03 | [2.05e-03, 3.21e-03] | 1.2e-07 | 1.8e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | 2.62e-03 | [2.04e-03, 3.20e-03] | 1.3e-07 | 1.8e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | 1.17e-03 | [3.81e-04, 1.96e-03] | 6.0e-03 | 7.3e-03 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | 1.41e-03 | [6.35e-04, 2.19e-03] | 1.3e-03 | 1.7e-03 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | 2.00e-03 | [1.44e-03, 2.55e-03] | 4.9e-06 | 2.5e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | 2.93e-03 | [2.38e-03, 3.49e-03] | 8.0e-08 | 1.8e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 3.00e-03 | [2.46e-03, 3.54e-03] | 1.6e-07 | 1.8e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | quantum (biased) L=1 | 10 vs 20 | 2.98e-03 | [2.44e-03, 3.53e-03] | 1.5e-07 | 1.2e-05 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | quantum (biased) L=2 | 10 vs 20 | 1.92e-03 | [1.35e-03, 2.49e-03] | 4.1e-06 | 2.7e-05 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | 1.40e-03 | [8.33e-04, 1.96e-03] | 1.2e-04 | 1.4e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | 4.10e-05 | [-5.46e-04, 6.28e-04] | 8.8e-01 | 8.8e-01 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -1.67e-04 | [-7.83e-04, 4.49e-04] | 5.8e-01 | 9.5e-01 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -6.29e-04 | [-1.20e-03, -5.49e-05] | 3.4e-02 | 2.6e-02 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | surrogate L=1 | 10 vs 5 | 1.62e-03 | [1.08e-03, 2.15e-03] | 7.5e-05 | 6.7e-04 |
| paper | photonic MerLin (unbiased readout, threshold) N=3 | surrogate L=2 | 10 vs 5 | -1.16e-02 | [-1.21e-02, -1.11e-02] | 3.0e-12 | 6.7e-04 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | -3.04e-04 | [-6.55e-04, 4.71e-05] | 8.5e-02 | 4.5e-02 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | -3.10e-04 | [-6.56e-04, 3.66e-05] | 7.6e-02 | 1.2e-01 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | -1.76e-03 | [-2.44e-03, -1.09e-03] | 1.3e-04 | 3.3e-04 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | -1.52e-03 | [-2.18e-03, -8.59e-04] | 3.6e-04 | 1.7e-03 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | -9.37e-04 | [-1.22e-03, -6.49e-04] | 2.1e-06 | 3.3e-04 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -2.93e-03 | [-3.49e-03, -2.38e-03] | 8.0e-08 | 1.8e-04 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | 6.88e-05 | [-1.81e-04, 3.19e-04] | 5.7e-01 | 8.5e-01 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | quantum (biased) L=1 | 10 vs 20 | 5.27e-05 | [-1.98e-04, 3.03e-04] | 6.6e-01 | 6.8e-01 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | quantum (biased) L=2 | 10 vs 20 | -1.01e-03 | [-1.34e-03, -6.85e-04] | 7.7e-07 | 9.9e-05 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | -1.53e-03 | [-1.84e-03, -1.23e-03] | 9.6e-11 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -2.89e-03 | [-3.25e-03, -2.53e-03] | 5.6e-16 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -3.10e-03 | [-3.51e-03, -2.69e-03] | 6.3e-15 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -3.56e-03 | [-3.89e-03, -3.23e-03] | 3.4e-19 | 1.2e-05 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | surrogate L=1 | 10 vs 5 | -1.32e-03 | [-1.54e-03, -1.10e-03] | 2.7e-07 | 6.7e-04 |
| paper | photonic MerLin (vertex readout, PNR) N=2 | surrogate L=2 | 10 vs 5 | -1.45e-02 | [-1.47e-02, -1.43e-02] | 1.3e-16 | 6.7e-04 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (vertex readout, threshold) N=2 | 10 vs 10 | -3.73e-04 | [-6.97e-04, -4.87e-05] | 2.7e-02 | 3.1e-02 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (group readout, threshold) N=2 | 10 vs 10 | -3.79e-04 | [-6.97e-04, -5.98e-05] | 2.3e-02 | 5.4e-02 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (unbiased readout, threshold) N=2 | 10 vs 10 | -1.83e-03 | [-2.50e-03, -1.17e-03] | 1.1e-04 | 1.8e-04 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (vertex readout, threshold) N=3 | 10 vs 10 | -1.59e-03 | [-2.24e-03, -9.38e-04] | 2.8e-04 | 1.0e-03 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (group readout, threshold) N=3 | 10 vs 10 | -1.01e-03 | [-1.25e-03, -7.57e-04] | 2.3e-07 | 1.8e-04 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (unbiased readout, threshold) N=3 | 10 vs 10 | -3.00e-03 | [-3.54e-03, -2.46e-03] | 1.6e-07 | 1.8e-04 |
| paper | photonic MerLin (group readout, PNR) N=2 | photonic MerLin (vertex readout, PNR) N=2 | 10 vs 10 | -6.88e-05 | [-3.19e-04, 1.81e-04] | 5.7e-01 | 8.5e-01 |
| paper | photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=1 | 10 vs 20 | -1.61e-05 | [-2.15e-04, 1.82e-04] | 8.7e-01 | 8.1e-01 |
| paper | photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=2 | 10 vs 20 | -1.08e-03 | [-1.37e-03, -7.87e-04] | 4.4e-08 | 2.7e-05 |
| paper | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 20 | -1.60e-03 | [-1.87e-03, -1.33e-03] | 1.7e-12 | 1.2e-05 |
| paper | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 20 | -2.96e-03 | [-3.29e-03, -2.63e-03] | 2.8e-16 | 1.2e-05 |
| paper | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, code CNOTs) L=1 | 10 vs 20 | -3.17e-03 | [-3.56e-03, -2.78e-03] | 9.8e-15 | 1.2e-05 |
| paper | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, code CNOTs) L=2 | 10 vs 20 | -3.63e-03 | [-3.93e-03, -3.33e-03] | 4.4e-20 | 1.2e-05 |
| paper | photonic MerLin (group readout, PNR) N=2 | surrogate L=1 | 10 vs 5 | -1.38e-03 | [-1.54e-03, -1.23e-03] | 6.4e-09 | 6.7e-04 |
| paper | photonic MerLin (group readout, PNR) N=2 | surrogate L=2 | 10 vs 5 | -1.46e-02 | [-1.47e-02, -1.44e-02] | 4.4e-18 | 6.7e-04 |
