# Fair comparison: QuaRK, its photonic adaptation and classical baselines

All values are local runs on the same windows, split and seeds (1, 2, 3, 4, 12345); the paper reports no baseline. Metric: test MSE on 2,000 held-out windows at the ridge chosen on validation data; normalized = MSE / variance of the test labels; mean ± std over 5 seeds. Matching: the same Matérn kernel ridge readout and tuning for every kernel model; 315 features for QuaRK, its photonic version and the echo state networks; 75 raw inputs for the reservoir-free baselines.

**Coverage:** complete: 50/50 planned runs of the compared systems completed and validated. **Commensurability:** COMMENSURABLE (same seeds: True; same dataset per seed: True; identical data and readout settings: True; finite metrics: True).

## Systems

| System | Role | Scope | Runs | F1 | F2 | F3 | Forecast | Features | Measurement | Settings searched | Wall-clock per run |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| QuaRK, paper settings | method, reproduced | paper-accurate settings, 5 seeds | 5/5 | 0.019 ± 0.0133 | 0.48 ± 0.0347 | 0.508 ± 0.103 | 0.882 ± 0.0654 | 315 | 1000 per observable (authors' model) | 0 | 606 s |
| QuaRK, exact expectations | method variant: no shot noise | paper-accurate settings, 5 seeds | 5/5 | 1.16e-04 ± 1.81e-04 | 0.328 ± 0.0438 | 0.391 ± 0.108 | 0.891 ± 0.0761 | 315 | exact | 0 | 639 s |
| QuaRK, random-Pauli shadows | method variant: the paper text's estimator | paper-accurate settings, 5 seeds | 5/5 | 0.0881 ± 0.0424 | 0.62 ± 0.026 | 0.634 ± 0.0985 | 0.914 ± 0.0456 | 315 | 1000 snapshots, shared (random-Pauli shadows) | 0 | 641 s |
| Photonic, 1000 shots | MerLin adaptation | adaptation, simulated, 5 seeds | 5/5 | 0.0565 ± 0.031 | 0.492 ± 0.0876 | 0.589 ± 0.0796 | 0.901 ± 0.053 | 315 | 1000 click patterns, shared | 0 | 633 s |
| Photonic, exact | MerLin adaptation, no shot noise | adaptation, simulated, 5 seeds | 5/5 | 1.43e-04 ± 1.71e-04 | 0.307 ± 0.0622 | 0.388 ± 0.065 | 0.89 ± 0.0736 | 315 | exact | 0 | 645 s |
| Photonic, exact, distinguishable photons | control: no interference | adaptation, simulated, 5 seeds | 5/5 | 1.40e-04 ± 1.65e-04 | 0.307 ± 0.0618 | 0.387 ± 0.0656 | 0.89 ± 0.0728 | 315 | exact | 0 | 666 s |
| Matérn kernel on the raw window | fair baseline (ours) | 5 seeds | 5/5 | 7.40e-10 ± 2.45e-10 | 1.30e-09 ± 5.83e-10 | 7.37e-05 ± 3.83e-05 | 0.843 ± 0.0903 | 75 | none (classical) | 0 | 667 s |
| Linear ridge on the raw window | fair baseline (ours) | 5 seeds | 5/5 | 5.03e-13 ± 1.12e-12 | 5.50e-13 ± 1.23e-12 | 0.281 ± 0.133 | 0.826 ± 0.0934 | 75 | none (classical) | 0 | 4 s |
| Echo state network, default | fair baseline (ours) | 5 seeds | 5/5 | 1.58e-04 ± 1.60e-05 | 0.0634 ± 0.011 | 0.0895 ± 0.0146 | 0.848 ± 0.0919 | 315 | none (classical) | 0 | 644 s |
| Echo state network, tuned | fair baseline (ours) | 5 seeds | 5/5 | 2.07e-07 ± 4.25e-08 | 4.20e-06 ± 3.02e-06 | 1.80e-05 ± 1.60e-05 | 0.854 ± 0.0818 | 315 | none (classical) | 18 | 747 s |

F1–F3: the paper's tasks (F1 as run: current-point readout); Forecast: F1 as the text defines it. Values: normalized test MSE.

## Paper-reported values

The paper's only numbers are QuaRK's, from one run (the authors' seed 12345, ridge chosen on the test set). Test MSE at N = 8000, from the authors' release and from our run at the same seed and rule. They are PARTIALLY_COMMENSURABLE with the tables above: one seed, a ridge chosen on the test set, raw MSE; the paper has no value for any other system (N/A).

| Task | Paper (authors' release) | This reproduction |
|---|---:|---:|
| Current-point readout (F1 as run) | 0.004306 | 0.004306 |
| Exponential fading (F2) | 0.4491 | 0.4491 |
| Volterra (F3) | 0.6508 | 0.6508 |

## Paired comparisons

Delta = MSE(A) − MSE(B) over the 5 seeds; frozen 2 SE rule; seeds = seeds where A is lower; 95 % interval of the mean delta (descriptive).

| A | B | Task | Mean delta | SE | Seeds | 95 % interval | Label | Origin |
|---|---|---|---:|---:|---:|---|---|---|
| QuaRK, paper settings | Matérn kernel on the raw window | Current-point readout (F1 as run) | 0.00834 | 0.00276 | 0/5 | [6.77e-04, 0.016] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Matérn kernel on the raw window | Exponential fading (F2) | 0.464 | 0.0639 | 0/5 | [0.286, 0.641] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Matérn kernel on the raw window | Volterra (F3) | 0.751 | 0.0805 | 0/5 | [0.527, 0.974] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.0181 | 0.0121 | 0/5 | [-0.0154, 0.0515] | no clear separation | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Linear ridge on the raw window | Current-point readout (F1 as run) | 0.00834 | 0.00276 | 0/5 | [6.77e-04, 0.016] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Linear ridge on the raw window | Exponential fading (F2) | 0.464 | 0.0639 | 0/5 | [0.286, 0.641] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Linear ridge on the raw window | Volterra (F3) | 0.332 | 0.0379 | 0/5 | [0.227, 0.437] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.0254 | 0.0126 | 0/5 | [-0.00957, 0.0603] | B lower (interval contains 0) | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, default | Current-point readout (F1 as run) | 0.00827 | 0.00276 | 0/5 | [6.04e-04, 0.0159] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, default | Exponential fading (F2) | 0.404 | 0.0575 | 0/5 | [0.244, 0.563] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, default | Volterra (F3) | 0.618 | 0.0677 | 0/5 | [0.43, 0.806] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, default | One-step forecast (F1 as written) | 0.016 | 0.0116 | 1/5 | [-0.0161, 0.0481] | no clear separation | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, tuned | Current-point readout (F1 as run) | 0.00834 | 0.00276 | 0/5 | [6.77e-04, 0.016] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, tuned | Exponential fading (F2) | 0.464 | 0.0639 | 0/5 | [0.286, 0.641] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, tuned | Volterra (F3) | 0.751 | 0.0805 | 0/5 | [0.527, 0.974] | B lower | frozen in the Fair Baseline Plan |
| QuaRK, paper settings | Echo state network, tuned | One-step forecast (F1 as written) | 0.0128 | 0.0107 | 1/5 | [-0.017, 0.0426] | no clear separation | frozen in the Fair Baseline Plan |
| QuaRK, exact expectations | Matérn kernel on the raw window | Current-point readout (F1 as run) | 5.33e-05 | 3.80e-05 | 0/5 | [-5.22e-05, 1.59e-04] | no clear separation | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Matérn kernel on the raw window | Exponential fading (F2) | 0.319 | 0.0531 | 0/5 | [0.172, 0.467] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Matérn kernel on the raw window | Volterra (F3) | 0.576 | 0.0805 | 0/5 | [0.353, 0.8] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.022 | 0.0137 | 0/5 | [-0.0161, 0.06] | no clear separation | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Linear ridge on the raw window | Current-point readout (F1 as run) | 5.33e-05 | 3.80e-05 | 0/5 | [-5.22e-05, 1.59e-04] | no clear separation | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Linear ridge on the raw window | Exponential fading (F2) | 0.319 | 0.0531 | 0/5 | [0.172, 0.467] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Linear ridge on the raw window | Volterra (F3) | 0.157 | 0.0342 | 0/5 | [0.062, 0.252] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.0293 | 0.0142 | 0/5 | [-0.0101, 0.0687] | B lower (interval contains 0) | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, default | Current-point readout (F1 as run) | -1.56e-05 | 3.91e-05 | 4/5 | [-1.24e-04, 9.30e-05] | no clear separation | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, default | Exponential fading (F2) | 0.259 | 0.0488 | 0/5 | [0.124, 0.395] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, default | Volterra (F3) | 0.444 | 0.0684 | 0/5 | [0.254, 0.634] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, default | One-step forecast (F1 as written) | 0.0199 | 0.013 | 0/5 | [-0.0163, 0.0561] | no clear separation | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, tuned | Current-point readout (F1 as run) | 5.33e-05 | 3.80e-05 | 0/5 | [-5.23e-05, 1.59e-04] | no clear separation | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, tuned | Exponential fading (F2) | 0.319 | 0.0531 | 0/5 | [0.172, 0.467] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, tuned | Volterra (F3) | 0.576 | 0.0804 | 0/5 | [0.353, 0.8] | B lower | added after the main sweep (phase 03) |
| QuaRK, exact expectations | Echo state network, tuned | One-step forecast (F1 as written) | 0.0167 | 0.0126 | 1/5 | [-0.0182, 0.0517] | no clear separation | added after the main sweep (phase 03) |
| Photonic, 1000 shots | Matérn kernel on the raw window | Current-point readout (F1 as run) | 0.0249 | 0.00638 | 0/5 | [0.00721, 0.0426] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Matérn kernel on the raw window | Exponential fading (F2) | 0.464 | 0.0499 | 0/5 | [0.326, 0.603] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Matérn kernel on the raw window | Volterra (F3) | 0.87 | 0.0562 | 0/5 | [0.713, 1.03] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.0269 | 0.0131 | 0/5 | [-0.00941, 0.0631] | B lower (interval contains 0) | Phase 4 plan |
| Photonic, 1000 shots | Linear ridge on the raw window | Current-point readout (F1 as run) | 0.0249 | 0.00638 | 0/5 | [0.00721, 0.0426] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Linear ridge on the raw window | Exponential fading (F2) | 0.464 | 0.0499 | 0/5 | [0.326, 0.603] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Linear ridge on the raw window | Volterra (F3) | 0.45 | 0.0784 | 0/5 | [0.233, 0.668] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.0342 | 0.0137 | 0/5 | [-0.00384, 0.0722] | B lower (interval contains 0) | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, default | Current-point readout (F1 as run) | 0.0249 | 0.00638 | 0/5 | [0.00714, 0.0426] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, default | Exponential fading (F2) | 0.404 | 0.0449 | 0/5 | [0.28, 0.529] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, default | Volterra (F3) | 0.737 | 0.0441 | 0/5 | [0.614, 0.859] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, default | One-step forecast (F1 as written) | 0.0248 | 0.0127 | 0/5 | [-0.0103, 0.0599] | no clear separation | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, tuned | Current-point readout (F1 as run) | 0.0249 | 0.00638 | 0/5 | [0.00721, 0.0426] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, tuned | Exponential fading (F2) | 0.464 | 0.0499 | 0/5 | [0.326, 0.603] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, tuned | Volterra (F3) | 0.87 | 0.0562 | 0/5 | [0.714, 1.03] | B lower | Phase 4 plan |
| Photonic, 1000 shots | Echo state network, tuned | One-step forecast (F1 as written) | 0.0216 | 0.0115 | 0/5 | [-0.0102, 0.0534] | no clear separation | Phase 4 plan |
| Photonic, exact | Matérn kernel on the raw window | Current-point readout (F1 as run) | 6.34e-05 | 3.58e-05 | 0/5 | [-3.60e-05, 1.63e-04] | no clear separation | Phase 4 plan |
| Photonic, exact | Matérn kernel on the raw window | Exponential fading (F2) | 0.287 | 0.0265 | 0/5 | [0.214, 0.361] | B lower | Phase 4 plan |
| Photonic, exact | Matérn kernel on the raw window | Volterra (F3) | 0.572 | 0.0418 | 0/5 | [0.456, 0.688] | B lower | Phase 4 plan |
| Photonic, exact | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.0211 | 0.0116 | 0/5 | [-0.0109, 0.0532] | no clear separation | Phase 4 plan |
| Photonic, exact | Linear ridge on the raw window | Current-point readout (F1 as run) | 6.34e-05 | 3.58e-05 | 0/5 | [-3.60e-05, 1.63e-04] | no clear separation | Phase 4 plan |
| Photonic, exact | Linear ridge on the raw window | Exponential fading (F2) | 0.287 | 0.0265 | 0/5 | [0.214, 0.361] | B lower | Phase 4 plan |
| Photonic, exact | Linear ridge on the raw window | Volterra (F3) | 0.153 | 0.095 | 1/5 | [-0.11, 0.417] | no clear separation | Phase 4 plan |
| Photonic, exact | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.0284 | 0.012 | 0/5 | [-0.00493, 0.0618] | B lower (interval contains 0) | Phase 4 plan |
| Photonic, exact | Echo state network, default | Current-point readout (F1 as run) | -5.56e-06 | 3.69e-05 | 4/5 | [-1.08e-04, 9.70e-05] | no clear separation | Phase 4 plan |
| Photonic, exact | Echo state network, default | Exponential fading (F2) | 0.227 | 0.02 | 0/5 | [0.172, 0.283] | B lower | Phase 4 plan |
| Photonic, exact | Echo state network, default | Volterra (F3) | 0.44 | 0.0335 | 0/5 | [0.347, 0.533] | B lower | Phase 4 plan |
| Photonic, exact | Echo state network, default | One-step forecast (F1 as written) | 0.0191 | 0.0109 | 0/5 | [-0.0112, 0.0493] | no clear separation | Phase 4 plan |
| Photonic, exact | Echo state network, tuned | Current-point readout (F1 as run) | 6.33e-05 | 3.58e-05 | 0/5 | [-3.61e-05, 1.63e-04] | no clear separation | Phase 4 plan |
| Photonic, exact | Echo state network, tuned | Exponential fading (F2) | 0.287 | 0.0265 | 0/5 | [0.214, 0.361] | B lower | Phase 4 plan |
| Photonic, exact | Echo state network, tuned | Volterra (F3) | 0.572 | 0.0418 | 0/5 | [0.456, 0.689] | B lower | Phase 4 plan |
| Photonic, exact | Echo state network, tuned | One-step forecast (F1 as written) | 0.0159 | 0.0104 | 0/5 | [-0.0129, 0.0446] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | Current-point readout (F1 as run) | 6.16e-05 | 3.44e-05 | 0/5 | [-3.39e-05, 1.57e-04] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | Exponential fading (F2) | 0.287 | 0.0264 | 0/5 | [0.213, 0.36] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | Volterra (F3) | 0.572 | 0.0454 | 0/5 | [0.446, 0.698] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Matérn kernel on the raw window | One-step forecast (F1 as written) | 0.0213 | 0.0113 | 0/5 | [-0.00998, 0.0526] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | Current-point readout (F1 as run) | 6.16e-05 | 3.44e-05 | 0/5 | [-3.39e-05, 1.57e-04] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | Exponential fading (F2) | 0.287 | 0.0264 | 0/5 | [0.213, 0.36] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | Volterra (F3) | 0.153 | 0.0942 | 1/5 | [-0.109, 0.414] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Linear ridge on the raw window | One-step forecast (F1 as written) | 0.0286 | 0.0118 | 0/5 | [-0.004, 0.0613] | B lower (interval contains 0) | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, default | Current-point readout (F1 as run) | -7.35e-06 | 3.56e-05 | 4/5 | [-1.06e-04, 9.14e-05] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, default | Exponential fading (F2) | 0.227 | 0.02 | 0/5 | [0.171, 0.282] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, default | Volterra (F3) | 0.439 | 0.0369 | 0/5 | [0.337, 0.542] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, default | One-step forecast (F1 as written) | 0.0193 | 0.0106 | 0/5 | [-0.0103, 0.0488] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, tuned | Current-point readout (F1 as run) | 6.15e-05 | 3.44e-05 | 0/5 | [-3.40e-05, 1.57e-04] | no clear separation | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, tuned | Exponential fading (F2) | 0.287 | 0.0264 | 0/5 | [0.213, 0.36] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, tuned | Volterra (F3) | 0.572 | 0.0454 | 0/5 | [0.446, 0.698] | B lower | Phase 4 plan |
| Photonic, exact, distinguishable photons | Echo state network, tuned | One-step forecast (F1 as written) | 0.0161 | 0.0101 | 0/5 | [-0.0118, 0.044] | no clear separation | Phase 4 plan |
| Photonic, exact | QuaRK, exact expectations | Current-point readout (F1 as run) | 1.00e-05 | 1.20e-05 | 2/5 | [-2.34e-05, 4.34e-05] | no clear separation | Phase 4 plan |
| Photonic, exact | QuaRK, exact expectations | Exponential fading (F2) | -0.0321 | 0.0394 | 3/5 | [-0.141, 0.0774] | no clear separation | Phase 4 plan |
| Photonic, exact | QuaRK, exact expectations | Volterra (F3) | -0.00387 | 0.0694 | 3/5 | [-0.197, 0.189] | no clear separation | Phase 4 plan |
| Photonic, exact | QuaRK, exact expectations | One-step forecast (F1 as written) | -8.56e-04 | 0.00236 | 2/5 | [-0.00742, 0.00571] | no clear separation | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, random-Pauli shadows | Current-point readout (F1 as run) | -0.0137 | 0.00617 | 4/5 | [-0.0308, 0.00344] | A lower (interval contains 0) | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, random-Pauli shadows | Exponential fading (F2) | -0.148 | 0.0692 | 5/5 | [-0.34, 0.0441] | A lower (interval contains 0) | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, random-Pauli shadows | Volterra (F3) | -0.0744 | 0.0535 | 4/5 | [-0.223, 0.074] | no clear separation | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, random-Pauli shadows | One-step forecast (F1 as written) | -0.00567 | 0.0038 | 3/5 | [-0.0162, 0.00488] | no clear separation | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, paper settings | Current-point readout (F1 as run) | 0.0166 | 0.00492 | 0/5 | [0.00292, 0.0303] | B lower | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, paper settings | Exponential fading (F2) | 8.25e-04 | 0.0255 | 3/5 | [-0.0701, 0.0717] | no clear separation | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, paper settings | Volterra (F3) | 0.119 | 0.0411 | 0/5 | [0.00445, 0.233] | B lower | Phase 4 plan |
| Photonic, 1000 shots | QuaRK, paper settings | One-step forecast (F1 as written) | 0.0088 | 0.003 | 0/5 | [4.71e-04, 0.0171] | B lower | Phase 4 plan |

## Evidence for the paper's claims, every system

Interpolation: largest training MSE at ridge 10⁻¹² over the three paper tasks and 5 seeds. Decay: log-log slope of the test MSE over N = 500–8000 (−0.5 is 1/√N). Lower at 8000: seeds with a lower test MSE at N = 8000 than at N = 100. Order: seeds where F1 < F2 < F3, on raw and on normalized test MSE. The linear ridge cannot interpolate (76 parameters for 8,000 windows).

| System | Interpolation | Slope F1 | Slope F2 | Slope F3 | Lower at 8000 (F1/F2/F3) | Order, raw | Order, normalized |
|---|---:|---:|---:|---:|---|---:|---:|
| QuaRK, paper settings | 5.63e-21 | -0.458 ± 0.0747 | -0.0918 ± 0.0208 | -0.0855 ± 0.0235 | 5/5/5 | 5/5 | 3/5 |
| QuaRK, exact expectations | 1.09e-11 | -1.36 ± 0.289 | -0.162 ± 0.0703 | -0.16 ± 0.0685 | 5/5/5 | 5/5 | 4/5 |
| QuaRK, random-Pauli shadows | 5.59e-21 | -0.287 ± 0.0418 | -0.0681 ± 0.0095 | -0.0652 ± 0.0297 | 5/5/5 | 5/5 | 3/5 |
| Photonic, 1000 shots | 2.92e-24 | -0.306 ± 0.0634 | -0.0711 ± 0.0169 | -0.0589 ± 0.0154 | 5/5/5 | 5/5 | 5/5 |
| Photonic, exact | 4.61e-20 | -1.18 ± 0.201 | -0.128 ± 0.0177 | -0.121 ± 0.0234 | 5/5/5 | 5/5 | 5/5 |
| Photonic, exact, distinguishable photons | 3.71e-17 | -1.21 ± 0.237 | -0.122 ± 0.0162 | -0.118 ± 0.0183 | 5/5/5 | 5/5 | 5/5 |
| Matérn kernel on the raw window | 4.33e-13 | -0.526 ± 0.213 | -0.76 ± 0.348 | -3.26 ± 0.374 | 5/5/5 | 5/5 | 3/5 |
| Linear ridge on the raw window | 0.796 | -11.7 ± 3.36 | -10 ± 3.91 | -0.0491 ± 0.0111 | 5/5/5 | 5/5 | 5/5 |
| Echo state network, default | 4.01e-13 | -0.887 ± 0.0596 | -0.236 ± 0.0105 | -0.386 ± 0.0889 | 5/5/5 | 5/5 | 5/5 |
| Echo state network, tuned | 8.89e-13 | -1.85 ± 0.0259 | -0.961 ± 0.211 | -1.34 ± 0.11 | 5/5/5 | 5/5 | 4/5 |

## Wall-clock

One machine (Apple M5, 10 cores, local CPU, recorded in the reproduction log), one run at a time. Workload per run: 10,000 windows, features once, then for each of the 4 tasks the Matérn tuning (4 ν × up to 80 ξ evaluations), the 26-ridge sweep and the 10-size training sweep. Quantum features are simulated; these times say nothing about hardware execution.

| System | Per run (min / mean / max) | Feature construction (mean) | Share of the run |
|---|---|---:|---:|
| QuaRK, paper settings | 594 / 606 / 615 s | 12 s | 2% |
| QuaRK, exact expectations | 618 / 639 / 680 s | 8 s | 1% |
| QuaRK, random-Pauli shadows | 619 / 641 / 659 s | 33 s | 5% |
| Photonic, 1000 shots | 614 / 633 / 651 s | 40 s | 6% |
| Photonic, exact | 629 / 645 / 675 s | 40 s | 6% |
| Photonic, exact, distinguishable photons | 629 / 666 / 697 s | 50 s | 7% |
| Matérn kernel on the raw window | 656 / 667 / 686 s | 0 s | 0% |
| Linear ridge on the raw window | 4 / 4 / 4 s | 0 s | 0% |
| Echo state network, default | 623 / 644 / 662 s | 1 s | 0% |
| Echo state network, tuned | 710 / 747 / 793 s | 39 s | 5% |

## Extensions

Conditions the paper makes no claim about. Only QuaRK or its photonic version ran under them, so they characterize those models and make no advantage statement over the classical baselines.

| Extension | What ran | Runs | Label | Table |
|---|---|---:|---|---|
| shot noise | QuaRK at 100 and 10000 shots, both estimators | 20/20 | characterization | `results/statistics.md (shot budget)` |
| scaling | QuaRK at n = 3, 7 (R = 3) and R = 1, 6 (n = 5) | 20/20 | characterization | `results/statistics.md (n and R)` |
| photonic shots | Photonic at 100 and 10000 shots | 10/10 | characterization; compared only with QuaRK at the same budget | `results/photonic_comparison.md` |
