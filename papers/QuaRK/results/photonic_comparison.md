# Photonic adaptation: comparison with QuaRK

Simulated photonic reservoir (MerLin, 3 photons, 14 modes, threshold detection), same data, seeds and readout as QuaRK. Metric: test MSE at the validation-selected ridge; normalized = MSE / variance of the test labels; mean ± std over 5 seeds.

**Tolerance (frozen plan): PASS**

| Photonic | QuaRK | Task | Ratio of normalized MSE | Interpolation | Decay | Within ×2 | Pass |
|---|---|---|---:|:-:|:-:|:-:|:-:|
| Photonic, exact | QuaRK, exact | Current-point readout (F1 as run) | 1.237 | yes | yes | yes | yes |
| Photonic, exact | QuaRK, exact | Exponential fading (F2) | 0.936 | yes | yes | yes | yes |
| Photonic, exact | QuaRK, exact | Volterra (F3) | 0.992 | yes | yes | yes | yes |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | Current-point readout (F1 as run) | 0.641 | yes | yes | yes | yes |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | Exponential fading (F2) | 0.793 | yes | yes | yes | yes |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | Volterra (F3) | 0.929 | yes | yes | yes | yes |

## Setup of every photonic configuration

Simulated, not hardware-executed; no photonic parameter is trained, so no gradient is used. Common to all: computation space: FOCK, occupancy_readout=True; detector: ideal threshold detectors (click / no click), no dark counts, no loss; postselection: none; encoding: phases pi (1 - 1e-6) tanh(Pi x_t) on all modes at each step, between copies of one fixed random MZI mesh; measurement: click patterns, reduced to 1- and 2-mode click marginals.

| Configuration | Photons | Modes | Sub-reservoirs | Input state | Click patterns | Marginals | Shots | Simulator | Seeds | Wall-clock per run (min / mean / max) |
|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| Photonic, 100 shots | 3 | 14 | 3 | `[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | 469 | 105 per sub-reservoir | 100 | MerLin 0.4.1, Perceval 1.3.1, cpu, float64 | 1, 2, 3, 4, 12345 | 615 / 627 / 646 s |
| Photonic, 1000 shots | 3 | 14 | 3 | `[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | 469 | 105 per sub-reservoir | 1000 | MerLin 0.4.1, Perceval 1.3.1, cpu, float64 | 1, 2, 3, 4, 12345 | 614 / 633 / 651 s |
| Photonic, 10000 shots | 3 | 14 | 3 | `[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | 469 | 105 per sub-reservoir | 10000 | MerLin 0.4.1, Perceval 1.3.1, cpu, float64 | 1, 2, 3, 4, 12345 | 592 / 608 / 621 s |
| Photonic, exact | 3 | 14 | 3 | `[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | 469 | 105 per sub-reservoir | exact | MerLin 0.4.1, Perceval 1.3.1, cpu, float64 | 1, 2, 3, 4, 12345 | 629 / 645 / 675 s |
| Photonic, exact, distinguishable photons | 3, distinguishable | 14 | 3 | `[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | 469 | 105 per sub-reservoir | exact | MerLin 0.4.1, Perceval 1.3.1, cpu, float64 | 1, 2, 3, 4, 12345 | 629 / 666 / 697 s |

The version string is the one MerLin prints; the installed distribution is merlinquantum 0.4.2 (`requirements.txt`).

## Measurement penalty

Each seed's test MSE divided by the same seed's test MSE with exact photonic features; mean ± std over 5 seeds.

| Configuration | Task | Ratio to exact |
|---|---|---:|
| Photonic, 100 shots | Current-point readout (F1 as run) | 9.35e+03 ± 1.02e+04 |
| Photonic, 1000 shots | Current-point readout (F1 as run) | 1.19e+03 ± 1.24e+03 |
| Photonic, 10000 shots | Current-point readout (F1 as run) | 133 ± 122 |
| Photonic, 100 shots | Exponential fading (F2) | 2.48 ± 0.424 |
| Photonic, 1000 shots | Exponential fading (F2) | 1.61 ± 0.165 |
| Photonic, 10000 shots | Exponential fading (F2) | 1.22 ± 0.105 |
| Photonic, 100 shots | Volterra (F3) | 2.12 ± 0.294 |
| Photonic, 1000 shots | Volterra (F3) | 1.53 ± 0.149 |
| Photonic, 10000 shots | Volterra (F3) | 1.2 ± 0.0708 |
| Photonic, 100 shots | One-step forecast (F1 as written) | 1.08 ± 0.0622 |
| Photonic, 1000 shots | One-step forecast (F1 as written) | 1.01 ± 0.0245 |
| Photonic, 10000 shots | One-step forecast (F1 as written) | 0.991 ± 0.0108 |

## Interpolation and decay, every photonic configuration

Seeds (of 5) with training MSE ≤ 1e-10 at ridge 1e-12, and with a lower test MSE at N = 8000 than at N = 100.

| Configuration | Task | Interpolating | Decaying | Max training MSE |
|---|---|---:|---:|---:|
| Photonic, 100 shots | Current-point readout (F1 as run) | 5 | 5 | 5.60e-24 |
| Photonic, 100 shots | Exponential fading (F2) | 5 | 5 | 3.10e-24 |
| Photonic, 100 shots | Volterra (F3) | 5 | 5 | 3.17e-24 |
| Photonic, 100 shots | One-step forecast (F1 as written) | 5 | 5 | 5.24e-25 |
| Photonic, 1000 shots | Current-point readout (F1 as run) | 5 | 5 | 3.56e-25 |
| Photonic, 1000 shots | Exponential fading (F2) | 5 | 5 | 2.30e-24 |
| Photonic, 1000 shots | Volterra (F3) | 5 | 5 | 2.92e-24 |
| Photonic, 1000 shots | One-step forecast (F1 as written) | 5 | 5 | 4.73e-25 |
| Photonic, 10000 shots | Current-point readout (F1 as run) | 5 | 5 | 3.64e-25 |
| Photonic, 10000 shots | Exponential fading (F2) | 5 | 5 | 1.77e-24 |
| Photonic, 10000 shots | Volterra (F3) | 5 | 5 | 1.89e-23 |
| Photonic, 10000 shots | One-step forecast (F1 as written) | 5 | 5 | 4.78e-25 |
| Photonic, exact | Current-point readout (F1 as run) | 5 | 5 | 4.61e-20 |
| Photonic, exact | Exponential fading (F2) | 5 | 5 | 1.98e-20 |
| Photonic, exact | Volterra (F3) | 5 | 5 | 4.41e-20 |
| Photonic, exact | One-step forecast (F1 as written) | 5 | 4 | 5.05e-25 |
| Photonic, exact, distinguishable photons | Current-point readout (F1 as run) | 5 | 5 | 3.71e-17 |
| Photonic, exact, distinguishable photons | Exponential fading (F2) | 5 | 5 | 2.06e-20 |
| Photonic, exact, distinguishable photons | Volterra (F3) | 5 | 5 | 4.92e-20 |
| Photonic, exact, distinguishable photons | One-step forecast (F1 as written) | 5 | 4 | 5.17e-25 |

## Paired comparisons

Delta = MSE(photonic) − MSE(other), 2 SE rule; "lower" names the model with the lower test MSE; seeds = seeds where the photonic model is lower.

| Photonic | Other | Task | Normalized (photonic) | Normalized (other) | Mean delta | SE | Seeds | Label |
|---|---|---|---:|---:|---:|---:|---:|---|
| Photonic, exact | QuaRK, exact | Current-point readout (F1 as run) | 1.43e-04 ± 1.71e-04 | 1.16e-04 ± 1.81e-04 | 1.00e-05 | 1.20e-05 | 2/5 | no clear separation |
| Photonic, exact | QuaRK, exact | Exponential fading (F2) | 0.307 ± 0.0622 | 0.328 ± 0.0438 | -0.0321 | 0.0394 | 3/5 | no clear separation |
| Photonic, exact | QuaRK, exact | Volterra (F3) | 0.388 ± 0.065 | 0.391 ± 0.108 | -0.00387 | 0.0694 | 3/5 | no clear separation |
| Photonic, exact | QuaRK, exact | One-step forecast (F1 as written) | 0.89 ± 0.0736 | 0.891 ± 0.0761 | -8.56e-04 | 0.00236 | 2/5 | no clear separation |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | Current-point readout (F1 as run) | 0.0565 ± 0.031 | 0.0881 ± 0.0424 | -0.0137 | 0.00617 | 4/5 | photonic lower (interval contains 0) |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | Exponential fading (F2) | 0.492 ± 0.0876 | 0.62 ± 0.026 | -0.148 | 0.0692 | 5/5 | photonic lower (interval contains 0) |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | Volterra (F3) | 0.589 ± 0.0796 | 0.634 ± 0.0985 | -0.0744 | 0.0535 | 4/5 | no clear separation |
| Photonic, 1000 shots | QuaRK, shadows, 1000 shots | One-step forecast (F1 as written) | 0.901 ± 0.053 | 0.914 ± 0.0456 | -0.00567 | 0.0038 | 3/5 | no clear separation |
| Photonic, 100 shots | QuaRK, shadows, 100 shots | Current-point readout (F1 as run) | 0.398 ± 0.154 | 0.435 ± 0.114 | -0.0152 | 0.0235 | 4/5 | no clear separation |
| Photonic, 100 shots | QuaRK, shadows, 100 shots | Exponential fading (F2) | 0.744 ± 0.0564 | 0.832 ± 0.0479 | -0.103 | 0.0508 | 4/5 | photonic lower (interval contains 0) |
| Photonic, 100 shots | QuaRK, shadows, 100 shots | Volterra (F3) | 0.809 ± 0.0462 | 0.803 ± 0.071 | 0.00966 | 0.0256 | 2/5 | no clear separation |
| Photonic, 100 shots | QuaRK, shadows, 100 shots | One-step forecast (F1 as written) | 0.953 ± 0.0286 | 0.952 ± 0.0336 | 3.94e-04 | 0.00242 | 3/5 | no clear separation |
| Photonic, 10000 shots | QuaRK, shadows, 10000 shots | Current-point readout (F1 as run) | 0.0083 ± 0.00635 | 0.015 ± 0.0104 | -0.00289 | 0.00103 | 5/5 | photonic lower |
| Photonic, 10000 shots | QuaRK, shadows, 10000 shots | Exponential fading (F2) | 0.377 ± 0.094 | 0.454 ± 0.0311 | -0.0897 | 0.0397 | 4/5 | photonic lower (interval contains 0) |
| Photonic, 10000 shots | QuaRK, shadows, 10000 shots | Volterra (F3) | 0.466 ± 0.079 | 0.492 ± 0.114 | -0.0386 | 0.0627 | 3/5 | no clear separation |
| Photonic, 10000 shots | QuaRK, shadows, 10000 shots | One-step forecast (F1 as written) | 0.881 ± 0.0676 | 0.884 ± 0.064 | -0.00153 | 0.00142 | 4/5 | no clear separation |
| Photonic, 100 shots | QuaRK, per-observable noise, 100 shots | Current-point readout (F1 as run) | 0.398 ± 0.154 | 0.126 ± 0.0543 | 0.119 | 0.0241 | 0/5 | other lower |
| Photonic, 100 shots | QuaRK, per-observable noise, 100 shots | Exponential fading (F2) | 0.744 ± 0.0564 | 0.656 ± 0.0378 | 0.0748 | 0.0179 | 0/5 | other lower |
| Photonic, 100 shots | QuaRK, per-observable noise, 100 shots | Volterra (F3) | 0.809 ± 0.0462 | 0.662 ± 0.0913 | 0.229 | 0.049 | 0/5 | other lower |
| Photonic, 100 shots | QuaRK, per-observable noise, 100 shots | One-step forecast (F1 as written) | 0.953 ± 0.0286 | 0.914 ± 0.0493 | 0.0177 | 0.00471 | 0/5 | other lower |
| Photonic, 1000 shots | QuaRK, per-observable noise, 1000 shots (paper) | Current-point readout (F1 as run) | 0.0565 ± 0.031 | 0.019 ± 0.0133 | 0.0166 | 0.00492 | 0/5 | other lower |
| Photonic, 1000 shots | QuaRK, per-observable noise, 1000 shots (paper) | Exponential fading (F2) | 0.492 ± 0.0876 | 0.48 ± 0.0347 | 8.25e-04 | 0.0255 | 3/5 | no clear separation |
| Photonic, 1000 shots | QuaRK, per-observable noise, 1000 shots (paper) | Volterra (F3) | 0.589 ± 0.0796 | 0.508 ± 0.103 | 0.119 | 0.0411 | 0/5 | other lower |
| Photonic, 1000 shots | QuaRK, per-observable noise, 1000 shots (paper) | One-step forecast (F1 as written) | 0.901 ± 0.053 | 0.882 ± 0.0654 | 0.0088 | 0.003 | 0/5 | other lower |
| Photonic, 10000 shots | QuaRK, per-observable noise, 10000 shots | Current-point readout (F1 as run) | 0.0083 ± 0.00635 | 0.00381 ± 0.00373 | 0.00199 | 6.77e-04 | 0/5 | other lower |
| Photonic, 10000 shots | QuaRK, per-observable noise, 10000 shots | Exponential fading (F2) | 0.377 ± 0.094 | 0.391 ± 0.0316 | -0.028 | 0.0338 | 3/5 | no clear separation |
| Photonic, 10000 shots | QuaRK, per-observable noise, 10000 shots | Volterra (F3) | 0.466 ± 0.079 | 0.435 ± 0.104 | 0.0436 | 0.0621 | 2/5 | no clear separation |
| Photonic, 10000 shots | QuaRK, per-observable noise, 10000 shots | One-step forecast (F1 as written) | 0.881 ± 0.0676 | 0.881 ± 0.0726 | 1.44e-04 | 0.00212 | 2/5 | no clear separation |
| Photonic, exact | Photonic, exact, distinguishable photons | Current-point readout (F1 as run) | 1.43e-04 ± 1.71e-04 | 1.40e-04 ± 1.65e-04 | 1.78e-06 | 2.06e-06 | 1/5 | no clear separation |
| Photonic, exact | Photonic, exact, distinguishable photons | Exponential fading (F2) | 0.307 ± 0.0622 | 0.307 ± 0.0618 | 6.82e-04 | 4.83e-04 | 2/5 | no clear separation |
| Photonic, exact | Photonic, exact, distinguishable photons | Volterra (F3) | 0.388 ± 0.065 | 0.387 ± 0.0656 | 3.60e-04 | 0.00356 | 3/5 | no clear separation |
| Photonic, exact | Photonic, exact, distinguishable photons | One-step forecast (F1 as written) | 0.89 ± 0.0736 | 0.89 ± 0.0728 | -1.96e-04 | 3.44e-04 | 3/5 | no clear separation |
| Photonic, 100 shots | Matérn kernel on the raw window | Current-point readout (F1 as run) | 0.398 ± 0.154 | 7.40e-10 ± 2.45e-10 | 0.175 | 0.0308 | 0/5 | other lower |
| Photonic, 100 shots | Matérn kernel on the raw window | Exponential fading (F2) | 0.744 ± 0.0564 | 1.30e-09 ± 5.83e-10 | 0.725 | 0.112 | 0/5 | other lower |
| Photonic, 100 shots | Matérn kernel on the raw window | Volterra (F3) | 0.809 ± 0.0462 | 7.37e-05 ± 3.83e-05 | 1.22 | 0.131 | 0/5 | other lower |
| Photonic, 100 shots | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.953 ± 0.0286 | 0.843 ± 0.0903 | 0.0497 | 0.0155 | 0/5 | other lower |
| Photonic, 100 shots | Linear ridge on the raw window | Current-point readout (F1 as run) | 0.398 ± 0.154 | 5.03e-13 ± 1.12e-12 | 0.175 | 0.0308 | 0/5 | other lower |
| Photonic, 100 shots | Linear ridge on the raw window | Exponential fading (F2) | 0.744 ± 0.0564 | 5.50e-13 ± 1.23e-12 | 0.725 | 0.112 | 0/5 | other lower |
| Photonic, 100 shots | Linear ridge on the raw window | Volterra (F3) | 0.809 ± 0.0462 | 0.281 ± 0.133 | 0.798 | 0.108 | 0/5 | other lower |
| Photonic, 100 shots | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.953 ± 0.0286 | 0.826 ± 0.0934 | 0.057 | 0.0162 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, default | Current-point readout (F1 as run) | 0.398 ± 0.154 | 1.58e-04 ± 1.60e-05 | 0.175 | 0.0308 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, default | Exponential fading (F2) | 0.744 ± 0.0564 | 0.0634 ± 0.011 | 0.665 | 0.106 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, default | Volterra (F3) | 0.809 ± 0.0462 | 0.0895 ± 0.0146 | 1.08 | 0.12 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, default | One-step forecast (F1 as written) | 0.953 ± 0.0286 | 0.848 ± 0.0919 | 0.0477 | 0.0155 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, tuned | Current-point readout (F1 as run) | 0.398 ± 0.154 | 2.07e-07 ± 4.25e-08 | 0.175 | 0.0308 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, tuned | Exponential fading (F2) | 0.744 ± 0.0564 | 4.20e-06 ± 3.02e-06 | 0.725 | 0.112 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, tuned | Volterra (F3) | 0.809 ± 0.0462 | 1.80e-05 ± 1.60e-05 | 1.22 | 0.131 | 0/5 | other lower |
| Photonic, 100 shots | Echo state network, tuned | One-step forecast (F1 as written) | 0.953 ± 0.0286 | 0.854 ± 0.0818 | 0.0445 | 0.0137 | 0/5 | other lower |
| Photonic, 1000 shots | Matérn kernel on the raw window | Current-point readout (F1 as run) | 0.0565 ± 0.031 | 7.40e-10 ± 2.45e-10 | 0.0249 | 0.00638 | 0/5 | other lower |
| Photonic, 1000 shots | Matérn kernel on the raw window | Exponential fading (F2) | 0.492 ± 0.0876 | 1.30e-09 ± 5.83e-10 | 0.464 | 0.0499 | 0/5 | other lower |
| Photonic, 1000 shots | Matérn kernel on the raw window | Volterra (F3) | 0.589 ± 0.0796 | 7.37e-05 ± 3.83e-05 | 0.87 | 0.0562 | 0/5 | other lower |
| Photonic, 1000 shots | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.901 ± 0.053 | 0.843 ± 0.0903 | 0.0269 | 0.0131 | 0/5 | other lower (interval contains 0) |
| Photonic, 1000 shots | Linear ridge on the raw window | Current-point readout (F1 as run) | 0.0565 ± 0.031 | 5.03e-13 ± 1.12e-12 | 0.0249 | 0.00638 | 0/5 | other lower |
| Photonic, 1000 shots | Linear ridge on the raw window | Exponential fading (F2) | 0.492 ± 0.0876 | 5.50e-13 ± 1.23e-12 | 0.464 | 0.0499 | 0/5 | other lower |
| Photonic, 1000 shots | Linear ridge on the raw window | Volterra (F3) | 0.589 ± 0.0796 | 0.281 ± 0.133 | 0.45 | 0.0784 | 0/5 | other lower |
| Photonic, 1000 shots | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.901 ± 0.053 | 0.826 ± 0.0934 | 0.0342 | 0.0137 | 0/5 | other lower (interval contains 0) |
| Photonic, 1000 shots | Echo state network, default | Current-point readout (F1 as run) | 0.0565 ± 0.031 | 1.58e-04 ± 1.60e-05 | 0.0249 | 0.00638 | 0/5 | other lower |
| Photonic, 1000 shots | Echo state network, default | Exponential fading (F2) | 0.492 ± 0.0876 | 0.0634 ± 0.011 | 0.404 | 0.0449 | 0/5 | other lower |
| Photonic, 1000 shots | Echo state network, default | Volterra (F3) | 0.589 ± 0.0796 | 0.0895 ± 0.0146 | 0.737 | 0.0441 | 0/5 | other lower |
| Photonic, 1000 shots | Echo state network, default | One-step forecast (F1 as written) | 0.901 ± 0.053 | 0.848 ± 0.0919 | 0.0248 | 0.0127 | 0/5 | no clear separation |
| Photonic, 1000 shots | Echo state network, tuned | Current-point readout (F1 as run) | 0.0565 ± 0.031 | 2.07e-07 ± 4.25e-08 | 0.0249 | 0.00638 | 0/5 | other lower |
| Photonic, 1000 shots | Echo state network, tuned | Exponential fading (F2) | 0.492 ± 0.0876 | 4.20e-06 ± 3.02e-06 | 0.464 | 0.0499 | 0/5 | other lower |
| Photonic, 1000 shots | Echo state network, tuned | Volterra (F3) | 0.589 ± 0.0796 | 1.80e-05 ± 1.60e-05 | 0.87 | 0.0562 | 0/5 | other lower |
| Photonic, 1000 shots | Echo state network, tuned | One-step forecast (F1 as written) | 0.901 ± 0.053 | 0.854 ± 0.0818 | 0.0216 | 0.0115 | 0/5 | no clear separation |
| Photonic, 10000 shots | Matérn kernel on the raw window | Current-point readout (F1 as run) | 0.0083 ± 0.00635 | 7.40e-10 ± 2.45e-10 | 0.00368 | 0.00135 | 0/5 | other lower (interval contains 0) |
| Photonic, 10000 shots | Matérn kernel on the raw window | Exponential fading (F2) | 0.377 ± 0.094 | 1.30e-09 ± 5.83e-10 | 0.349 | 0.0284 | 0/5 | other lower |
| Photonic, 10000 shots | Matérn kernel on the raw window | Volterra (F3) | 0.466 ± 0.079 | 7.37e-05 ± 3.83e-05 | 0.684 | 0.0377 | 0/5 | other lower |
| Photonic, 10000 shots | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.881 ± 0.0676 | 0.843 ± 0.0903 | 0.0177 | 0.0115 | 0/5 | no clear separation |
| Photonic, 10000 shots | Linear ridge on the raw window | Current-point readout (F1 as run) | 0.0083 ± 0.00635 | 5.03e-13 ± 1.12e-12 | 0.00368 | 0.00135 | 0/5 | other lower (interval contains 0) |
| Photonic, 10000 shots | Linear ridge on the raw window | Exponential fading (F2) | 0.377 ± 0.094 | 5.50e-13 ± 1.23e-12 | 0.349 | 0.0284 | 0/5 | other lower |
| Photonic, 10000 shots | Linear ridge on the raw window | Volterra (F3) | 0.466 ± 0.079 | 0.281 ± 0.133 | 0.265 | 0.0935 | 1/5 | other lower |
| Photonic, 10000 shots | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.881 ± 0.0676 | 0.826 ± 0.0934 | 0.025 | 0.012 | 0/5 | other lower (interval contains 0) |
| Photonic, 10000 shots | Echo state network, default | Current-point readout (F1 as run) | 0.0083 ± 0.00635 | 1.58e-04 ± 1.60e-05 | 0.00361 | 0.00135 | 0/5 | other lower (interval contains 0) |
| Photonic, 10000 shots | Echo state network, default | Exponential fading (F2) | 0.377 ± 0.094 | 0.0634 ± 0.011 | 0.289 | 0.0241 | 0/5 | other lower |
| Photonic, 10000 shots | Echo state network, default | Volterra (F3) | 0.466 ± 0.079 | 0.0895 ± 0.0146 | 0.552 | 0.0286 | 0/5 | other lower |
| Photonic, 10000 shots | Echo state network, default | One-step forecast (F1 as written) | 0.881 ± 0.0676 | 0.848 ± 0.0919 | 0.0156 | 0.011 | 0/5 | no clear separation |
| Photonic, 10000 shots | Echo state network, tuned | Current-point readout (F1 as run) | 0.0083 ± 0.00635 | 2.07e-07 ± 4.25e-08 | 0.00368 | 0.00135 | 0/5 | other lower (interval contains 0) |
| Photonic, 10000 shots | Echo state network, tuned | Exponential fading (F2) | 0.377 ± 0.094 | 4.20e-06 ± 3.02e-06 | 0.349 | 0.0284 | 0/5 | other lower |
| Photonic, 10000 shots | Echo state network, tuned | Volterra (F3) | 0.466 ± 0.079 | 1.80e-05 ± 1.60e-05 | 0.684 | 0.0377 | 0/5 | other lower |
| Photonic, 10000 shots | Echo state network, tuned | One-step forecast (F1 as written) | 0.881 ± 0.0676 | 0.854 ± 0.0818 | 0.0124 | 0.0102 | 1/5 | no clear separation |
| Photonic, exact | Matérn kernel on the raw window | Current-point readout (F1 as run) | 1.43e-04 ± 1.71e-04 | 7.40e-10 ± 2.45e-10 | 6.34e-05 | 3.58e-05 | 0/5 | no clear separation |
| Photonic, exact | Matérn kernel on the raw window | Exponential fading (F2) | 0.307 ± 0.0622 | 1.30e-09 ± 5.83e-10 | 0.287 | 0.0265 | 0/5 | other lower |
| Photonic, exact | Matérn kernel on the raw window | Volterra (F3) | 0.388 ± 0.065 | 7.37e-05 ± 3.83e-05 | 0.572 | 0.0418 | 0/5 | other lower |
| Photonic, exact | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.89 ± 0.0736 | 0.843 ± 0.0903 | 0.0211 | 0.0116 | 0/5 | no clear separation |
| Photonic, exact | Linear ridge on the raw window | Current-point readout (F1 as run) | 1.43e-04 ± 1.71e-04 | 5.03e-13 ± 1.12e-12 | 6.34e-05 | 3.58e-05 | 0/5 | no clear separation |
| Photonic, exact | Linear ridge on the raw window | Exponential fading (F2) | 0.307 ± 0.0622 | 5.50e-13 ± 1.23e-12 | 0.287 | 0.0265 | 0/5 | other lower |
| Photonic, exact | Linear ridge on the raw window | Volterra (F3) | 0.388 ± 0.065 | 0.281 ± 0.133 | 0.153 | 0.095 | 1/5 | no clear separation |
| Photonic, exact | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.89 ± 0.0736 | 0.826 ± 0.0934 | 0.0284 | 0.012 | 0/5 | other lower (interval contains 0) |
| Photonic, exact | Echo state network, default | Current-point readout (F1 as run) | 1.43e-04 ± 1.71e-04 | 1.58e-04 ± 1.60e-05 | -5.56e-06 | 3.69e-05 | 4/5 | no clear separation |
| Photonic, exact | Echo state network, default | Exponential fading (F2) | 0.307 ± 0.0622 | 0.0634 ± 0.011 | 0.227 | 0.02 | 0/5 | other lower |
| Photonic, exact | Echo state network, default | Volterra (F3) | 0.388 ± 0.065 | 0.0895 ± 0.0146 | 0.44 | 0.0335 | 0/5 | other lower |
| Photonic, exact | Echo state network, default | One-step forecast (F1 as written) | 0.89 ± 0.0736 | 0.848 ± 0.0919 | 0.0191 | 0.0109 | 0/5 | no clear separation |
| Photonic, exact | Echo state network, tuned | Current-point readout (F1 as run) | 1.43e-04 ± 1.71e-04 | 2.07e-07 ± 4.25e-08 | 6.33e-05 | 3.58e-05 | 0/5 | no clear separation |
| Photonic, exact | Echo state network, tuned | Exponential fading (F2) | 0.307 ± 0.0622 | 4.20e-06 ± 3.02e-06 | 0.287 | 0.0265 | 0/5 | other lower |
| Photonic, exact | Echo state network, tuned | Volterra (F3) | 0.388 ± 0.065 | 1.80e-05 ± 1.60e-05 | 0.572 | 0.0418 | 0/5 | other lower |
| Photonic, exact | Echo state network, tuned | One-step forecast (F1 as written) | 0.89 ± 0.0736 | 0.854 ± 0.0818 | 0.0159 | 0.0104 | 0/5 | no clear separation |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | Current-point readout (F1 as run) | 1.40e-04 ± 1.65e-04 | 7.40e-10 ± 2.45e-10 | 6.16e-05 | 3.44e-05 | 0/5 | no clear separation |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | Exponential fading (F2) | 0.307 ± 0.0618 | 1.30e-09 ± 5.83e-10 | 0.287 | 0.0264 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | Volterra (F3) | 0.387 ± 0.0656 | 7.37e-05 ± 3.83e-05 | 0.572 | 0.0454 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.89 ± 0.0728 | 0.843 ± 0.0903 | 0.0213 | 0.0113 | 0/5 | no clear separation |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | Current-point readout (F1 as run) | 1.40e-04 ± 1.65e-04 | 5.03e-13 ± 1.12e-12 | 6.16e-05 | 3.44e-05 | 0/5 | no clear separation |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | Exponential fading (F2) | 0.307 ± 0.0618 | 5.50e-13 ± 1.23e-12 | 0.287 | 0.0264 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | Volterra (F3) | 0.387 ± 0.0656 | 0.281 ± 0.133 | 0.153 | 0.0942 | 1/5 | no clear separation |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.89 ± 0.0728 | 0.826 ± 0.0934 | 0.0286 | 0.0118 | 0/5 | other lower (interval contains 0) |
| Photonic, exact, distinguishable photons | Echo state network, default | Current-point readout (F1 as run) | 1.40e-04 ± 1.65e-04 | 1.58e-04 ± 1.60e-05 | -7.35e-06 | 3.56e-05 | 4/5 | no clear separation |
| Photonic, exact, distinguishable photons | Echo state network, default | Exponential fading (F2) | 0.307 ± 0.0618 | 0.0634 ± 0.011 | 0.227 | 0.02 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Echo state network, default | Volterra (F3) | 0.387 ± 0.0656 | 0.0895 ± 0.0146 | 0.439 | 0.0369 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Echo state network, default | One-step forecast (F1 as written) | 0.89 ± 0.0728 | 0.848 ± 0.0919 | 0.0193 | 0.0106 | 0/5 | no clear separation |
| Photonic, exact, distinguishable photons | Echo state network, tuned | Current-point readout (F1 as run) | 1.40e-04 ± 1.65e-04 | 2.07e-07 ± 4.25e-08 | 6.15e-05 | 3.44e-05 | 0/5 | no clear separation |
| Photonic, exact, distinguishable photons | Echo state network, tuned | Exponential fading (F2) | 0.307 ± 0.0618 | 4.20e-06 ± 3.02e-06 | 0.287 | 0.0264 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Echo state network, tuned | Volterra (F3) | 0.387 ± 0.0656 | 1.80e-05 ± 1.60e-05 | 0.572 | 0.0454 | 0/5 | other lower |
| Photonic, exact, distinguishable photons | Echo state network, tuned | One-step forecast (F1 as written) | 0.89 ± 0.0728 | 0.854 ± 0.0818 | 0.0161 | 0.0101 | 0/5 | no clear separation |
