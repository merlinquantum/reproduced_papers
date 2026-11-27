### Phase 5 fair comparison (protocol paper, author training data, seeds 10-19, 2000 Adam steps)

| System | Role | Provenance | Scope | Successful / expected | Final test KL (mean +- std) | Early-stopped test KL (mean +- std) | Selected lr, L2 | Parameters | Wall-clock per run (s) | Excluded |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| B2 bias-encoded Fourier L=1 | additional fair baseline B2 | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 3.606e-03 +- 2.8e-08 | 3.690e-03 +- 1.4e-04 | 0.001, 0.1 | 8748 | 528 | none |
| photonic MerLin (group readout, PNR) N=2 | MerLin ADAPTATION (Phase 4) | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 4.009e-03 +- 9.6e-05 | 3.965e-03 +- 1.9e-04 | 0.003, 0 | 90 | 550 | none |
| quantum (biased) L=1 | paper method (fair-tuned) | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 4.106e-03 +- 3.7e-04 | 4.053e-03 +- 4.4e-04 | 0.001, 0 | 55 | 154 | none |
| surrogate L=1 | paper baseline B0 (fair-tuned) | locally executed (fair_final) | fair-tuned, paper-accurate data; PARTIAL | 9/10 | 4.415e-03 +- 2.1e-08 | 4.353e-03 +- 1.1e-04 | 0.001, 1 | 4374 | 231 | fair_final__c009_r01 FAILED |
| B5 quadratic (ceiling) | ceiling B5 (class contains the truth) | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 4.521e-03 +- 7.3e-05 | 4.521e-03 +- 9.8e-05 | 0.001, 0.01 | 84 | 9 | none |
| quantum (biased) L=2 | paper method (fair-tuned) | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 5.124e-03 +- 4.3e-04 | 4.733e-03 +- 2.5e-04 | 0.001, 0 | 91 | 242 | none |
| quantum (no bias, Fig. 10 CNOTs) L=1 | paper baseline (fair-tuned) | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 5.382e-03 +- 3.1e-04 | 5.330e-03 +- 2.8e-04 | 0.003, 0 | 54 | 144 | none |
| B3 bias-encoded MLP h=7 | additional fair baseline B3 | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 5.762e-03 +- 1.3e-05 | 5.757e-03 +- 2.2e-05 | 0.001, 0.001 | 97 | 9 | none |
| B4 unbiased MLP h=9 | additional fair baseline B4 | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 5.779e-03 +- 1.3e-05 | 5.820e-03 +- 5.6e-05 | 0.001, 0.001 | 93 | 8 | none |
| quantum (no bias, Fig. 10 CNOTs) L=2 | paper baseline (fair-tuned) | locally executed (fair_final) | fair-tuned, paper-accurate data | 10/10 | 7.127e-03 +- 6.8e-04 | 7.076e-03 +- 7.1e-04 | 0.001, 0 | 90 | 236 | none |

### Untuned reference (not in the statistics)

| System | Provenance | Successful / expected | Seeds | Final test KL (mean +- std) | Wall-clock per run (s) | Status |
|---|---|---:|---|---:|---:|---|
| surrogate L=2 (not tuned, paper App. F settings) | locally executed (Phase 3, fig8_surrogates) | 5/5 | 0, 1, 2, 3, 4 | 1.865e-02 +- 3.5e-06 | 2941 | PARTIALLY_COMMENSURABLE: same data, protocol paper, test set and 2000 steps, but plain SGD at the fixed paper lr, no validation tuning, Phase 3 seeds; excluded from the statistics |

### Commensurability

| System | Status | Reason |
|---|---|---|
| B2 bias-encoded Fourier L=1 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| photonic MerLin (group readout, PNR) N=2 | COMMENSURABLE | same data, protocol, test set, steps and tuning budget; a simulated MerLin ADAPTATION (exact probabilities, PNR detection), not the reproduction |
| quantum (biased) L=1 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| surrogate L=1 | COMMENSURABLE | PARTIAL coverage (9/10): one seed failed on both attempts (HardTanh saturation under Adam) |
| B5 quadratic (ceiling) | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| quantum (biased) L=2 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| quantum (no bias, Fig. 10 CNOTs) L=1 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| B3 bias-encoded MLP h=7 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| B4 unbiased MLP h=9 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| quantum (no bias, Fig. 10 CNOTs) L=2 | COMMENSURABLE | same data, protocol, test set, steps, seeds and tuning budget |
| surrogate L=2 (not tuned, paper App. F settings) | PARTIALLY_COMMENSURABLE | same data, protocol paper, test set and 2000 steps, but plain SGD at the fixed paper lr, no validation tuning, Phase 3 seeds; excluded from the statistics |
| Paper-reported Fig. 8 values (read off) | INCOMPARABLE | author-code evaluation (unscaled test inputs, D12), untuned optimisers, test-selected curves; compared separately in `fig8_vs_paper.md` |

### Pairwise comparisons (Welch 95% CI on reference - other; resolved = Welch and Mann-Whitney p < 0.05)

| Metric | Reference | Other | n | Diff | 95% CI | Welch p | MW p | Resolved |
|---|---|---|---|---:|---|---:|---:|---|
| final_test_kl | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -2.58e-04 | [-6.13e-04, +9.70e-05] | 1.4e-01 | 1.4e-01 | no |
| early_stop_test_kl | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -5.98e-04 | [-8.45e-04, -3.50e-04] | 8.2e-05 | 3.3e-04 | yes |
| final_test_kl | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -2.00e-03 | [-2.54e-03, -1.46e-03] | 9.6e-07 | 2.5e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -2.34e-03 | [-2.87e-03, -1.82e-03] | 8.0e-07 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=2 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | +1.11e-03 | [+8.05e-04, +1.42e-03] | 1.2e-05 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | +7.67e-04 | [+5.61e-04, +9.74e-04] | 5.1e-07 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=2 | B3 bias-encoded MLP h=7 | 10 vs 10 | -6.38e-04 | [-9.44e-04, -3.31e-04] | 1.1e-03 | 2.8e-03 | yes |
| early_stop_test_kl | quantum (biased) L=2 | B3 bias-encoded MLP h=7 | 10 vs 10 | -1.02e-03 | [-1.20e-03, -8.49e-04] | 3.0e-07 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=2 | B4 unbiased MLP h=9 | 10 vs 10 | -6.55e-04 | [-9.61e-04, -3.49e-04] | 9.2e-04 | 2.8e-03 | yes |
| early_stop_test_kl | quantum (biased) L=2 | B4 unbiased MLP h=9 | 10 vs 10 | -1.09e-03 | [-1.26e-03, -9.09e-04] | 9.3e-08 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=2 | quantum (biased) L=1 | 10 vs 10 | +1.02e-03 | [+6.42e-04, +1.39e-03] | 2.3e-05 | 5.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | quantum (biased) L=1 | 10 vs 10 | +6.80e-04 | [+3.35e-04, +1.02e-03] | 8.3e-04 | 3.6e-03 | yes |
| final_test_kl | quantum (biased) L=2 | B2 bias-encoded Fourier L=1 | 10 vs 10 | +1.52e-03 | [+1.21e-03, +1.82e-03] | 1.4e-06 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | B2 bias-encoded Fourier L=1 | 10 vs 10 | +1.04e-03 | [+8.52e-04, +1.23e-03] | 1.2e-08 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=2 | B5 quadratic (ceiling) | 10 vs 10 | +6.03e-04 | [+2.95e-04, +9.10e-04] | 1.5e-03 | 3.3e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | B5 quadratic (ceiling) | 10 vs 10 | +2.11e-04 | [+2.87e-05, +3.94e-04] | 2.7e-02 | 3.8e-02 | yes |
| final_test_kl | quantum (biased) L=2 | surrogate L=1 | 10 vs 9 | +7.09e-04 | [+4.03e-04, +1.01e-03] | 5.4e-04 | 2.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | surrogate L=1 | 10 vs 9 | +3.80e-04 | [+1.94e-04, +5.65e-04] | 7.2e-04 | 1.7e-03 | yes |
| final_test_kl | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -1.28e-03 | [-1.60e-03, -9.54e-04] | 1.6e-07 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -1.28e-03 | [-1.63e-03, -9.24e-04] | 1.3e-06 | 3.3e-04 | yes |
| final_test_kl | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -3.02e-03 | [-3.55e-03, -2.50e-03] | 7.2e-09 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -3.02e-03 | [-3.59e-03, -2.46e-03] | 8.2e-09 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=1 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | +9.65e-05 | [-1.71e-04, +3.64e-04] | 4.4e-01 | 5.7e-01 | no |
| early_stop_test_kl | quantum (biased) L=1 | photonic MerLin (group readout, PNR) N=2 | 10 vs 10 | +8.74e-05 | [-2.45e-04, +4.20e-04] | 5.8e-01 | 9.7e-01 | no |
| final_test_kl | quantum (biased) L=1 | B3 bias-encoded MLP h=7 | 10 vs 10 | -1.66e-03 | [-1.92e-03, -1.39e-03] | 1.8e-07 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | B3 bias-encoded MLP h=7 | 10 vs 10 | -1.70e-03 | [-2.02e-03, -1.39e-03] | 6.8e-07 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=1 | B4 unbiased MLP h=9 | 10 vs 10 | -1.67e-03 | [-1.94e-03, -1.41e-03] | 1.6e-07 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | B4 unbiased MLP h=9 | 10 vs 10 | -1.77e-03 | [-2.09e-03, -1.45e-03] | 4.2e-07 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=1 | quantum (biased) L=2 | 10 vs 10 | -1.02e-03 | [-1.39e-03, -6.42e-04] | 2.3e-05 | 5.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | quantum (biased) L=2 | 10 vs 10 | -6.80e-04 | [-1.02e-03, -3.35e-04] | 8.3e-04 | 3.6e-03 | yes |
| final_test_kl | quantum (biased) L=1 | B2 bias-encoded Fourier L=1 | 10 vs 10 | +5.00e-04 | [+2.36e-04, +7.63e-04] | 2.0e-03 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | B2 bias-encoded Fourier L=1 | 10 vs 10 | +3.63e-04 | [+3.79e-05, +6.88e-04] | 3.2e-02 | 1.4e-02 | yes |
| final_test_kl | quantum (biased) L=1 | B5 quadratic (ceiling) | 10 vs 10 | -4.16e-04 | [-6.81e-04, -1.50e-04] | 6.0e-03 | 2.8e-03 | yes |
| early_stop_test_kl | quantum (biased) L=1 | B5 quadratic (ceiling) | 10 vs 10 | -4.69e-04 | [-7.90e-04, -1.47e-04] | 8.8e-03 | 3.6e-03 | yes |
| final_test_kl | quantum (biased) L=1 | surrogate L=1 | 10 vs 9 | -3.09e-04 | [-5.73e-04, -4.58e-05] | 2.6e-02 | 3.7e-03 | yes |
| early_stop_test_kl | quantum (biased) L=1 | surrogate L=1 | 10 vs 9 | -3.00e-04 | [-6.23e-04, +2.24e-05] | 6.5e-02 | 1.6e-02 | no |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -1.37e-03 | [-1.60e-03, -1.14e-03] | 5.9e-08 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -1.36e-03 | [-1.59e-03, -1.14e-03] | 9.4e-10 | 1.8e-04 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -3.12e-03 | [-3.61e-03, -2.63e-03] | 1.1e-07 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -3.11e-03 | [-3.63e-03, -2.59e-03] | 8.2e-08 | 1.8e-04 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | B3 bias-encoded MLP h=7 | 10 vs 10 | -1.75e-03 | [-1.82e-03, -1.68e-03] | 3.2e-13 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | B3 bias-encoded MLP h=7 | 10 vs 10 | -1.79e-03 | [-1.93e-03, -1.66e-03] | 1.6e-10 | 1.8e-04 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | B4 unbiased MLP h=9 | 10 vs 10 | -1.77e-03 | [-1.84e-03, -1.70e-03] | 3.0e-13 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | B4 unbiased MLP h=9 | 10 vs 10 | -1.85e-03 | [-1.99e-03, -1.72e-03] | 1.4e-11 | 1.8e-04 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=1 | 10 vs 10 | -9.65e-05 | [-3.64e-04, +1.71e-04] | 4.4e-01 | 5.7e-01 | no |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=1 | 10 vs 10 | -8.74e-05 | [-4.20e-04, +2.45e-04] | 5.8e-01 | 9.7e-01 | no |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=2 | 10 vs 10 | -1.11e-03 | [-1.42e-03, -8.05e-04] | 1.2e-05 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=2 | 10 vs 10 | -7.67e-04 | [-9.74e-04, -5.61e-04] | 5.1e-07 | 1.8e-04 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | B2 bias-encoded Fourier L=1 | 10 vs 10 | +4.03e-04 | [+3.35e-04, +4.72e-04] | 3.2e-07 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | B2 bias-encoded Fourier L=1 | 10 vs 10 | +2.76e-04 | [+1.20e-04, +4.31e-04] | 1.7e-03 | 3.6e-03 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | B5 quadratic (ceiling) | 10 vs 10 | -5.12e-04 | [-5.93e-04, -4.32e-04] | 2.0e-10 | 1.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | B5 quadratic (ceiling) | 10 vs 10 | -5.56e-04 | [-7.00e-04, -4.12e-04] | 1.1e-06 | 1.8e-04 | yes |
| final_test_kl | photonic MerLin (group readout, PNR) N=2 | surrogate L=1 | 10 vs 9 | -4.06e-04 | [-4.74e-04, -3.37e-04] | 3.0e-07 | 2.8e-04 | yes |
| early_stop_test_kl | photonic MerLin (group readout, PNR) N=2 | surrogate L=1 | 10 vs 9 | -3.88e-04 | [-5.36e-04, -2.39e-04] | 5.9e-05 | 5.2e-04 | yes |
| final_test_kl | B2 bias-encoded Fourier L=1 | surrogate L=1 | 10 vs 9 | -8.09e-04 | [-8.09e-04, -8.09e-04] | 1.0e-71 | 2.8e-04 | yes |
| early_stop_test_kl | B2 bias-encoded Fourier L=1 | surrogate L=1 | 10 vs 9 | -6.63e-04 | [-7.82e-04, -5.45e-04] | 1.6e-09 | 2.8e-04 | yes |
| final_test_kl | B3 bias-encoded MLP h=7 | B4 unbiased MLP h=9 | 10 vs 10 | -1.76e-05 | [-3.00e-05, -5.27e-06] | 7.7e-03 | 1.1e-02 | yes |
| early_stop_test_kl | B3 bias-encoded MLP h=7 | B4 unbiased MLP h=9 | 10 vs 10 | -6.23e-05 | [-1.04e-04, -2.06e-05] | 7.0e-03 | 2.2e-03 | yes |
| final_test_kl | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -2.00e-03 | [-2.54e-03, -1.46e-03] | 9.6e-07 | 2.5e-04 | yes |
| early_stop_test_kl | quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | 10 vs 10 | -2.34e-03 | [-2.87e-03, -1.82e-03] | 8.0e-07 | 1.8e-04 | yes |
| final_test_kl | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -1.28e-03 | [-1.60e-03, -9.54e-04] | 1.6e-07 | 1.8e-04 | yes |
| early_stop_test_kl | quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | 10 vs 10 | -1.28e-03 | [-1.63e-03, -9.24e-04] | 1.3e-06 | 3.3e-04 | yes |

### Data robustness (fair_robustness: init seed 10, data seeds 101-103, final test KL)

| System | data seed 101 | 102 | 103 | mean |
|---|---:|---:|---:|---:|
| B2 bias-encoded Fourier L=1 | 3.737e-03 | 4.485e-03 | 4.776e-03 | 4.332e-03 |
| photonic MerLin (group readout, PNR) N=2 | 4.362e-03 | 5.115e-03 | 5.525e-03 | 5.001e-03 |
| surrogate L=1 | 4.802e-03 | 4.884e-03 | 5.670e-03 | 5.118e-03 |
| B5 quadratic (ceiling) | 5.587e-03 | 4.829e-03 | 6.340e-03 | 5.585e-03 |
| quantum (biased) L=1 | 6.011e-03 | 5.921e-03 | 5.842e-03 | 5.925e-03 |
| quantum (no bias, Fig. 10 CNOTs) L=1 | 5.602e-03 | 5.834e-03 | 6.439e-03 | 5.958e-03 |
| quantum (biased) L=2 | 5.034e-03 | 6.890e-03 | 6.199e-03 | 6.041e-03 |
| B3 bias-encoded MLP h=7 | 5.779e-03 | 6.748e-03 | 7.448e-03 | 6.658e-03 |
| B4 unbiased MLP h=9 | 6.421e-03 | 6.485e-03 | 7.172e-03 | 6.693e-03 |
| quantum (no bias, Fig. 10 CNOTs) L=2 | 6.315e-03 | 7.317e-03 | 6.915e-03 | 6.849e-03 |

| Reference | Other | Mean paired diff | Reference lower | Paired t p |
|---|---|---:|---:|---:|
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | +8.29e-05 | 2/3 | 0.88 |
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | -8.08e-04 | 3/3 | 0.084 |
| quantum (biased) L=2 | photonic MerLin (group readout, PNR) N=2 | +1.04e-03 | 0/3 | 0.11 |
| quantum (biased) L=2 | B3 bias-encoded MLP h=7 | -6.17e-04 | 2/3 | 0.27 |
| quantum (biased) L=2 | B4 unbiased MLP h=9 | -6.51e-04 | 2/3 | 0.35 |
| quantum (biased) L=2 | quantum (biased) L=1 | +1.16e-04 | 1/3 | 0.86 |
| quantum (biased) L=2 | B2 bias-encoded Fourier L=1 | +1.71e-03 | 0/3 | 0.039 |
| quantum (biased) L=2 | B5 quadratic (ceiling) | +4.56e-04 | 2/3 | 0.63 |
| quantum (biased) L=2 | surrogate L=1 | +9.23e-04 | 0/3 | 0.23 |
| quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | -3.33e-05 | 1/3 | 0.92 |
| quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | -9.24e-04 | 3/3 | 0.1 |
| quantum (biased) L=1 | photonic MerLin (group readout, PNR) N=2 | +9.24e-04 | 0/3 | 0.14 |
| quantum (biased) L=1 | B3 bias-encoded MLP h=7 | -7.33e-04 | 2/3 | 0.3 |
| quantum (biased) L=1 | B4 unbiased MLP h=9 | -7.68e-04 | 3/3 | 0.11 |
| quantum (biased) L=1 | quantum (biased) L=2 | -1.16e-04 | 2/3 | 0.86 |
| quantum (biased) L=1 | B2 bias-encoded Fourier L=1 | +1.59e-03 | 0/3 | 0.047 |
| quantum (biased) L=1 | B5 quadratic (ceiling) | +3.40e-04 | 1/3 | 0.54 |
| quantum (biased) L=1 | surrogate L=1 | +8.06e-04 | 0/3 | 0.13 |
| photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | -9.57e-04 | 3/3 | 0.024 |
| photonic MerLin (group readout, PNR) N=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | -1.85e-03 | 3/3 | 0.016 |
| photonic MerLin (group readout, PNR) N=2 | B3 bias-encoded MLP h=7 | -1.66e-03 | 3/3 | 0.0077 |
| photonic MerLin (group readout, PNR) N=2 | B4 unbiased MLP h=9 | -1.69e-03 | 3/3 | 0.014 |
| photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=1 | -9.24e-04 | 3/3 | 0.14 |
| photonic MerLin (group readout, PNR) N=2 | quantum (biased) L=2 | -1.04e-03 | 3/3 | 0.11 |
| photonic MerLin (group readout, PNR) N=2 | B2 bias-encoded Fourier L=1 | +6.69e-04 | 0/3 | 0.0037 |
| photonic MerLin (group readout, PNR) N=2 | B5 quadratic (ceiling) | -5.84e-04 | 2/3 | 0.32 |
| photonic MerLin (group readout, PNR) N=2 | surrogate L=1 | -1.18e-04 | 2/3 | 0.61 |
| B2 bias-encoded Fourier L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | -1.63e-03 | 3/3 | 0.0084 |
| B2 bias-encoded Fourier L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | -2.52e-03 | 3/3 | 0.0064 |
| B2 bias-encoded Fourier L=1 | photonic MerLin (group readout, PNR) N=2 | -6.69e-04 | 3/3 | 0.0037 |
| B2 bias-encoded Fourier L=1 | B3 bias-encoded MLP h=7 | -2.33e-03 | 3/3 | 0.0062 |
| B2 bias-encoded Fourier L=1 | B4 unbiased MLP h=9 | -2.36e-03 | 3/3 | 0.007 |
| B2 bias-encoded Fourier L=1 | quantum (biased) L=1 | -1.59e-03 | 3/3 | 0.047 |
| B2 bias-encoded Fourier L=1 | quantum (biased) L=2 | -1.71e-03 | 3/3 | 0.039 |
| B2 bias-encoded Fourier L=1 | B5 quadratic (ceiling) | -1.25e-03 | 3/3 | 0.11 |
| B2 bias-encoded Fourier L=1 | surrogate L=1 | -7.86e-04 | 3/3 | 0.059 |
