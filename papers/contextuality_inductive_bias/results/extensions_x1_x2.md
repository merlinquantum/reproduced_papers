### Extension X1 (characterization): contextuality witness of trained fair-final models

Inputs: test strategies + corner strategies. Ground truth on the same inputs: eta = 1.000. Contextual iff eta > 2/3.

| Model | n | eta mean | eta min | eta max | contextual runs |
|---|---:|---:|---:|---:|---:|
| B2 bias-encoded Fourier L=1 | 10 | 0.252 | 0.252 | 0.252 | 0/10 |
| B3 bias-encoded MLP h=7 | 10 | 0.181 | 0.178 | 0.183 | 0/10 |
| B4 unbiased MLP h=9 | | | | | n/a (no bias: Theorem 4.1 premise fails) |
| B5 quadratic (ceiling) | | | | | n/a (no bias: Theorem 4.1 premise fails) |
| photonic MerLin (group readout, PNR) N=2 | 10 | 0.335 | 0.300 | 0.389 | 0/10 |
| quantum (biased) L=1 | 10 | 0.386 | 0.339 | 0.480 | 0/10 |
| quantum (biased) L=2 | 10 | 0.518 | 0.372 | 0.639 | 0/10 |
| quantum (no bias, Fig. 10 CNOTs) L=1 | | | | | n/a (no bias: Theorem 4.1 premise fails) |
| quantum (no bias, Fig. 10 CNOTs) L=2 | | | | | n/a (no bias: Theorem 4.1 premise fails) |
| surrogate L=1 | | | | | n/a (no bias: Theorem 4.1 premise fails) |

### Extension X2 (advantage evidence): test KL by distance of the true behaviour from the centre

Test points per stratum: [0.0, 0.1): 7032, [0.1, 0.2): 2588, [0.2, 0.5]: 380

| Model | [0.0, 0.1) | [0.1, 0.2) | [0.2, 0.5] |
|---|---:|---:|---:|
| B5 quadratic (ceiling) | 2.54e-03 +- 2.7e-05 | 7.41e-03 +- 1.5e-04 | 2.15e-02 +- 4.0e-04 |
| B2 bias-encoded Fourier L=1 | 1.97e-03 +- 8.6e-07 | 5.30e-03 +- 1.6e-06 | 2.23e-02 +- 4.6e-06 |
| photonic MerLin (group readout, PNR) N=2 | 2.04e-03 +- 9.5e-05 | 6.29e-03 +- 2.1e-04 | 2.48e-02 +- 9.8e-04 |
| quantum (biased) L=1 | 2.20e-03 +- 1.5e-04 | 6.23e-03 +- 6.5e-04 | 2.49e-02 +- 3.4e-03 |
| surrogate L=1 | 2.26e-03 +- 3.1e-08 | 6.99e-03 +- 6.0e-09 | 2.68e-02 +- 4.8e-08 |
| quantum (no bias, Fig. 10 CNOTs) L=1 | 3.22e-03 +- 1.4e-04 | 8.02e-03 +- 6.7e-04 | 2.74e-02 +- 2.4e-03 |
| quantum (biased) L=2 | 2.81e-03 +- 2.2e-04 | 7.85e-03 +- 7.4e-04 | 2.95e-02 +- 3.7e-03 |
| quantum (no bias, Fig. 10 CNOTs) L=2 | 4.61e-03 +- 5.6e-04 | 1.03e-02 +- 8.6e-04 | 3.21e-02 +- 2.9e-03 |
| B4 unbiased MLP h=9 | 2.65e-03 +- 4.8e-06 | 1.02e-02 +- 2.9e-05 | 3.35e-02 +- 1.2e-04 |
| B3 bias-encoded MLP h=7 | 2.48e-03 +- 3.3e-06 | 1.04e-02 +- 3.1e-05 | 3.47e-02 +- 1.3e-04 |

| Reference | Other | Stratum | Diff | Welch p | MW p | Resolved |
|---|---|---|---:|---:|---:|---|
| quantum (biased) L=2 | surrogate L=1 | [0.0, 0.1) | +5.47e-04 | 2.5e-05 | 2.8e-04 | yes |
| quantum (biased) L=2 | surrogate L=1 | [0.1, 0.2) | +8.58e-04 | 5.3e-03 | 3.7e-03 | yes |
| quantum (biased) L=2 | surrogate L=1 | [0.2, 0.5] | +2.69e-03 | 4.7e-02 | 3.0e-02 | yes |
| quantum (biased) L=1 | surrogate L=1 | [0.0, 0.1) | -5.84e-05 | 2.5e-01 | 1.5e-01 | no |
| quantum (biased) L=1 | surrogate L=1 | [0.1, 0.2) | -7.62e-04 | 5.0e-03 | 3.0e-02 | yes |
| quantum (biased) L=1 | surrogate L=1 | [0.2, 0.5] | -1.87e-03 | 1.2e-01 | 3.0e-02 | no |
| quantum (biased) L=1 | B2 bias-encoded Fourier L=1 | [0.0, 0.1) | +2.25e-04 | 1.1e-03 | 1.8e-04 | yes |
| quantum (biased) L=1 | B2 bias-encoded Fourier L=1 | [0.1, 0.2) | +9.33e-04 | 1.5e-03 | 1.8e-04 | yes |
| quantum (biased) L=1 | B2 bias-encoded Fourier L=1 | [0.2, 0.5] | +2.62e-03 | 3.9e-02 | 2.6e-02 | yes |
| photonic MerLin (group readout, PNR) N=2 | B2 bias-encoded Fourier L=1 | [0.0, 0.1) | +7.00e-05 | 4.4e-02 | 2.8e-03 | yes |
| photonic MerLin (group readout, PNR) N=2 | B2 bias-encoded Fourier L=1 | [0.1, 0.2) | +9.96e-04 | 1.1e-07 | 1.8e-04 | yes |
| photonic MerLin (group readout, PNR) N=2 | B2 bias-encoded Fourier L=1 | [0.2, 0.5] | +2.53e-03 | 1.9e-05 | 1.8e-04 | yes |
| B2 bias-encoded Fourier L=1 | surrogate L=1 | [0.0, 0.1) | -2.84e-04 | 2.9e-24 | 2.7e-04 | yes |
| B2 bias-encoded Fourier L=1 | surrogate L=1 | [0.1, 0.2) | -1.70e-03 | 1.1e-28 | 2.8e-04 | yes |
| B2 bias-encoded Fourier L=1 | surrogate L=1 | [0.2, 0.5] | -4.49e-03 | 2.1e-28 | 2.8e-04 | yes |
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | [0.0, 0.1) | -1.80e-03 | 7.6e-07 | 1.8e-04 | yes |
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | [0.1, 0.2) | -2.45e-03 | 2.4e-06 | 4.4e-04 | yes |
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | [0.2, 0.5] | -2.67e-03 | 9.2e-02 | 8.9e-02 | no |
