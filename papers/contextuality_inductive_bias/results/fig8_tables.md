### fig8: candidates (final test KL is the primary metric)

| Protocol | Model | L2 | Steps | Train data | n | Final test KL (mean +- std) | Min test KL (mean) | Final train NLL (mean) |
|---|---|---:|---:|---|---:|---|---:|---:|
| code | quantum (biased) L=1 | 0 | 2000 | author | 5 | 4.675e-03 +- 1.3e-04 | 4.660e-03 | 0.6839 |
| code | quantum (biased) L=2 | 0 | 2000 | author | 5 | 4.606e-03 +- 2.6e-04 | 4.508e-03 | 0.6815 |
| code | quantum (no bias, code CNOTs) L=1 | 0 | 2000 | author | 5 | 9.271e-03 +- 4.2e-04 | 8.796e-03 | 0.6867 |
| code | quantum (no bias, Fig. 10 CNOTs) L=1 | 0 | 2000 | author | 5 | 6.897e-03 +- 7.1e-04 | 6.860e-03 | 0.6839 |
| code | quantum (no bias, code CNOTs) L=2 | 0 | 2000 | author | 5 | 9.639e-03 +- 9.9e-04 | 9.003e-03 | 0.6854 |
| code | quantum (no bias, Fig. 10 CNOTs) L=2 | 0 | 2000 | author | 5 | 7.645e-03 +- 7.8e-04 | 7.181e-03 | 0.6799 |
| code | surrogate L=1 | 0 | 2000 | author | 5 | 7.481e-03 +- 1.6e-05 | 6.231e-03 | 0.6765 |
| code | surrogate L=2 | 0 | 2000 | author | 5 | 1.271e-02 +- 3.0e-06 | 6.945e-03 | 0.6588 |
| fork | quantum (biased) L=1 | 0 | 2000 | author | 5 | 3.797e-03 +- 1.6e-04 | 3.797e-03 | 0.6850 |
| fork | quantum (biased) L=2 | 0 | 2000 | author | 5 | 3.604e-03 +- 1.6e-04 | 3.511e-03 | 0.6833 |
| fork | quantum (no bias, code CNOTs) L=1 | 0 | 2000 | author | 5 | 6.628e-03 +- 1.1e-03 | 6.628e-03 | 0.6871 |
| fork | quantum (no bias, Fig. 10 CNOTs) L=1 | 0 | 2000 | author | 5 | 5.095e-03 +- 5.5e-04 | 5.095e-03 | 0.6851 |
| fork | quantum (no bias, code CNOTs) L=2 | 0 | 2000 | author | 5 | 6.199e-03 +- 2.0e-04 | 6.199e-03 | 0.6859 |
| fork | quantum (no bias, Fig. 10 CNOTs) L=2 | 0 | 2000 | author | 5 | 4.830e-03 +- 1.7e-04 | 4.830e-03 | 0.6825 |
| fork | surrogate L=1 | 0 | 2000 | author | 5 | 4.276e-03 +- 5.5e-06 | 4.276e-03 | 0.6824 |
| fork | surrogate L=2 | 0 | 2000 | author | 5 | 6.965e-03 +- 1.9e-06 | 4.174e-03 | 0.6752 |
| paper | quantum (biased) L=1 | 0 | 2000 | author | 20 | 4.080e-03 +- 3.1e-04 | 4.026e-03 | 0.6839 |
| paper | quantum (biased) L=2 | 0 | 2000 | author | 20 | 5.144e-03 +- 5.7e-04 | 4.675e-03 | 0.6811 |
| paper | quantum (no bias, code CNOTs) L=1 | 0 | 2000 | author | 20 | 7.231e-03 +- 7.9e-04 | 7.195e-03 | 0.6865 |
| paper | quantum (no bias, Fig. 10 CNOTs) L=1 | 0 | 2000 | author | 20 | 5.666e-03 +- 5.1e-04 | 5.635e-03 | 0.6836 |
| paper | quantum (no bias, code CNOTs) L=2 | 0 | 2000 | author | 20 | 7.693e-03 +- 5.7e-04 | 7.619e-03 | 0.6847 |
| paper | quantum (no bias, Fig. 10 CNOTs) L=2 | 0 | 2000 | author | 20 | 7.023e-03 +- 6.5e-04 | 6.784e-03 | 0.6794 |
| paper | surrogate L=1 | 0 | 2000 | author | 5 | 5.448e-03 +- 5.7e-06 | 4.473e-03 | 0.6765 |
| paper | surrogate L=2 | 0 | 2000 | author | 5 | 1.865e-02 +- 3.5e-06 | 4.991e-03 | 0.6588 |

### fig8: reference models vs other model classes (final test KL, two-sided)

| Protocol | Reference | Other | n | Diff (reference - other) | 95% CI (Welch) | Welch p | Mann-Whitney p |
|---|---|---|---|---:|---|---:|---:|
| paper | quantum (biased) L=1 | quantum (biased) L=2 | 20 vs 20 | -1.06e-03 | [-1.36e-03, -7.69e-04] | 3.7e-08 | 6.0e-07 |
| paper | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 20 vs 20 | -1.59e-03 | [-1.86e-03, -1.31e-03] | 4.2e-13 | 7.9e-08 |
| paper | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | 20 vs 20 | -2.94e-03 | [-3.27e-03, -2.61e-03] | 8.1e-17 | 6.8e-08 |
| paper | quantum (biased) L=1 | quantum (no bias, code CNOTs) L=1 | 20 vs 20 | -3.15e-03 | [-3.54e-03, -2.76e-03] | 5.6e-15 | 6.8e-08 |
| paper | quantum (biased) L=1 | quantum (no bias, code CNOTs) L=2 | 20 vs 20 | -3.61e-03 | [-3.91e-03, -3.32e-03] | 3.0e-21 | 6.8e-08 |
| paper | quantum (biased) L=1 | surrogate L=1 | 20 vs 5 | -1.37e-03 | [-1.51e-03, -1.22e-03] | 4.0e-14 | 3.8e-05 |
| paper | quantum (biased) L=1 | surrogate L=2 | 20 vs 5 | -1.46e-02 | [-1.47e-02, -1.44e-02] | 1.8e-33 | 3.8e-05 |
| paper | quantum (biased) L=2 | quantum (biased) L=1 | 20 vs 20 | 1.06e-03 | [7.69e-04, 1.36e-03] | 3.7e-08 | 6.0e-07 |
| paper | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 20 vs 20 | -5.22e-04 | [-8.68e-04, -1.76e-04] | 4.1e-03 | 4.7e-03 |
| paper | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 20 vs 20 | -1.88e-03 | [-2.27e-03, -1.49e-03] | 8.5e-12 | 1.9e-07 |
| paper | quantum (biased) L=2 | quantum (no bias, code CNOTs) L=1 | 20 vs 20 | -2.09e-03 | [-2.53e-03, -1.65e-03] | 2.6e-11 | 7.9e-08 |
| paper | quantum (biased) L=2 | quantum (no bias, code CNOTs) L=2 | 20 vs 20 | -2.55e-03 | [-2.91e-03, -2.18e-03] | 9.7e-17 | 6.8e-08 |
| paper | quantum (biased) L=2 | surrogate L=1 | 20 vs 5 | -3.04e-04 | [-5.69e-04, -3.83e-05] | 2.7e-02 | 1.1e-01 |
| paper | quantum (biased) L=2 | surrogate L=2 | 20 vs 5 | -1.35e-02 | [-1.38e-02, -1.32e-02] | 7.3e-28 | 3.8e-05 |
| code | quantum (biased) L=1 | quantum (biased) L=2 | 5 vs 5 | 6.91e-05 | [-2.48e-04, 3.87e-04] | 6.1e-01 | 8.4e-01 |
| code | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 5 vs 5 | -2.22e-03 | [-3.09e-03, -1.35e-03] | 1.8e-03 | 7.9e-03 |
| code | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | 5 vs 5 | -2.97e-03 | [-3.93e-03, -2.01e-03] | 8.4e-04 | 7.9e-03 |
| code | quantum (biased) L=1 | quantum (no bias, code CNOTs) L=1 | 5 vs 5 | -4.60e-03 | [-5.11e-03, -4.09e-03] | 3.9e-06 | 7.9e-03 |
| code | quantum (biased) L=1 | quantum (no bias, code CNOTs) L=2 | 5 vs 5 | -4.96e-03 | [-6.18e-03, -3.74e-03] | 3.0e-04 | 7.9e-03 |
| code | quantum (biased) L=1 | surrogate L=1 | 5 vs 5 | -2.81e-03 | [-2.97e-03, -2.64e-03] | 9.3e-07 | 7.9e-03 |
| code | quantum (biased) L=1 | surrogate L=2 | 5 vs 5 | -8.03e-03 | [-8.20e-03, -7.87e-03] | 1.8e-08 | 7.9e-03 |
| code | quantum (biased) L=2 | quantum (biased) L=1 | 5 vs 5 | -6.91e-05 | [-3.87e-04, 2.48e-04] | 6.1e-01 | 8.4e-01 |
| code | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 5 vs 5 | -2.29e-03 | [-3.15e-03, -1.43e-03] | 1.0e-03 | 7.9e-03 |
| code | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 5 vs 5 | -3.04e-03 | [-3.99e-03, -2.09e-03] | 4.7e-04 | 7.9e-03 |
| code | quantum (biased) L=2 | quantum (no bias, code CNOTs) L=1 | 5 vs 5 | -4.67e-03 | [-5.19e-03, -4.14e-03] | 2.3e-07 | 7.9e-03 |
| code | quantum (biased) L=2 | quantum (no bias, code CNOTs) L=2 | 5 vs 5 | -5.03e-03 | [-6.24e-03, -3.82e-03] | 1.9e-04 | 7.9e-03 |
| code | quantum (biased) L=2 | surrogate L=1 | 5 vs 5 | -2.88e-03 | [-3.20e-03, -2.56e-03] | 1.4e-05 | 7.9e-03 |
| code | quantum (biased) L=2 | surrogate L=2 | 5 vs 5 | -8.10e-03 | [-8.42e-03, -7.78e-03] | 2.4e-07 | 7.9e-03 |
| fork | quantum (biased) L=1 | quantum (biased) L=2 | 5 vs 5 | 1.94e-04 | [-4.01e-05, 4.27e-04] | 9.3e-02 | 9.5e-02 |
| fork | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 5 vs 5 | -1.30e-03 | [-1.97e-03, -6.22e-04] | 4.8e-03 | 7.9e-03 |
| fork | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | 5 vs 5 | -1.03e-03 | [-1.27e-03, -7.92e-04] | 9.1e-06 | 7.9e-03 |
| fork | quantum (biased) L=1 | quantum (no bias, code CNOTs) L=1 | 5 vs 5 | -2.83e-03 | [-4.16e-03, -1.50e-03] | 3.8e-03 | 7.9e-03 |
| fork | quantum (biased) L=1 | quantum (no bias, code CNOTs) L=2 | 5 vs 5 | -2.40e-03 | [-2.67e-03, -2.13e-03] | 6.2e-08 | 7.9e-03 |
| fork | quantum (biased) L=1 | surrogate L=1 | 5 vs 5 | -4.79e-04 | [-6.78e-04, -2.79e-04] | 2.6e-03 | 7.9e-03 |
| fork | quantum (biased) L=1 | surrogate L=2 | 5 vs 5 | -3.17e-03 | [-3.37e-03, -2.97e-03] | 1.6e-06 | 7.9e-03 |
| fork | quantum (biased) L=2 | quantum (biased) L=1 | 5 vs 5 | -1.94e-04 | [-4.27e-04, 4.01e-05] | 9.3e-02 | 9.5e-02 |
| fork | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 5 vs 5 | -1.49e-03 | [-2.17e-03, -8.15e-04] | 2.7e-03 | 7.9e-03 |
| fork | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 5 vs 5 | -1.23e-03 | [-1.47e-03, -9.87e-04] | 2.5e-06 | 7.9e-03 |
| fork | quantum (biased) L=2 | quantum (no bias, code CNOTs) L=1 | 5 vs 5 | -3.02e-03 | [-4.36e-03, -1.69e-03] | 3.0e-03 | 7.9e-03 |
| fork | quantum (biased) L=2 | quantum (no bias, code CNOTs) L=2 | 5 vs 5 | -2.59e-03 | [-2.87e-03, -2.32e-03] | 3.5e-08 | 7.9e-03 |
| fork | quantum (biased) L=2 | surrogate L=1 | 5 vs 5 | -6.72e-04 | [-8.70e-04, -4.74e-04] | 7.0e-04 | 7.9e-03 |
| fork | quantum (biased) L=2 | surrogate L=2 | 5 vs 5 | -3.36e-03 | [-3.56e-03, -3.16e-03] | 1.2e-06 | 7.9e-03 |
