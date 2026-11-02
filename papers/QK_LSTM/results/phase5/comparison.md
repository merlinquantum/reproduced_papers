# Phase 5 fair comparison (generated after the frozen Phase 5 Comparison Plan)

Metric: test RMSE in AQI units on observed targets; lower is better. Paper-reported values are reference only (INCOMPARABLE for ranking). Wall-clock is from run_status timestamps on a local Apple M4 CPU, 1 torch thread per run; S1-S3 ran under CPU contention, so it is descriptive only.

## Primary: 80/20 split, paper hyperparameters (S1)

Scope: paper-accurate (paper hp, 5 seeds); test 2019-05-27..2020-07-01 (402 days).

| System | Role | Provenance | Scope | Successful / expected runs | Test RMSE (mean) | Std | Matching basis | Trainable params | Wall-clock / run (min) | Status |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| B0 persistence | naive floor (uses past AQI) | local run (s1_paper_hp_20261021-134301:B0_persistence) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 11.415 | 0.000 | 0 params; PARTIALLY_COMMENSURABLE | 0 | 0.0 | COMPLETE |
| B0b ridge | linear baseline | local run (s1_paper_hp_20261021-134301:B0b_ridge) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.425 | 0.000 | 34 params | 34 | 0.0 | COMPLETE |
| B1 LSTM(16) | paper baseline (reproduced) | local run (s1_paper_hp_20261021-134301:B1_lstm_1873) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.611 | 2.661 | paper architecture (1,873 params) | 1873 | 0.6 | COMPLETE |
| B2 LSTM(3) | fair baseline: iso-parameter | local run (s1_paper_hp_20261021-134301:B2_lstm_matched_196) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 13.351 | 3.615 | params within ±10% of Q-B (196/209) | 196 | 0.6 | COMPLETE |
| Q-A QK-LSTM (qubit) | reproduced method (Eq. 6 reading) | local run (s1_paper_hp_20261021-134301:QA_quantum_eq6_N4) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 15.661 | 3.771 | reference (449 params) | 449 | 9.4 | COMPLETE |
| Q-B QK-LSTM (qubit) | reproduced method | local run (s1_paper_hp_20261021-134301:QB_quantum_209) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 15.679 | 3.965 | reference (209 params) | 209 | 10.6 | COMPLETE |
| B3-A RBF-kernel LSTM | fair baseline: kernel ablation | local run (s1_paper_hp_20261021-134301:B3A_rbf_eq6_N4) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 15.927 | 3.855 | identical to Q-A except kernel | 449 | 0.7 | COMPLETE |
| B3-B RBF-kernel LSTM | fair baseline: kernel ablation | local run (s1_paper_hp_20261021-134301:B3B_rbf_209) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 16.006 | 3.801 | identical to Q-B except kernel | 209 | 0.7 | COMPLETE |
| QK-LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 9.20 | — | — | 209 | — | reference only |
| LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 15.94 | — | — | 1873 | — | reference only |

Paired decisions (frozen rule: 95 % t-CI of mean ΔRMSE, A − B):

| Claim | A | B | Commensurability | ΔRMSE mean | 95 % CI | A better (seeds) | Decision |
|---|---|---|---|---:|---|---:|---|
| C1 accuracy vs paper baseline | Q-B QK-LSTM (qubit) | B1 LSTM(16) | COMMENSURABLE | +3.068 | [+0.757, +5.379] | 0/5 | **A worse** |
| C1 accuracy vs paper baseline | Q-A QK-LSTM (qubit) | B1 LSTM(16) | COMMENSURABLE | +3.050 | [+1.150, +4.950] | 0/5 | **A worse** |
| C2 iso-parameter accuracy | Q-B QK-LSTM (qubit) | B2 LSTM(3) | COMMENSURABLE | +2.329 | [+1.595, +3.063] | 0/5 | **A worse** |
| C4 quantum vs classical kernel | Q-B QK-LSTM (qubit) | B3-B RBF-kernel LSTM | COMMENSURABLE | -0.326 | [-0.591, -0.062] | 5/5 | **A better** |
| C4 quantum vs classical kernel | Q-A QK-LSTM (qubit) | B3-A RBF-kernel LSTM | COMMENSURABLE | -0.266 | [-0.452, -0.080] | 5/5 | **A better** |
| naive floor | Q-B QK-LSTM (qubit) | B0 persistence | PARTIALLY_COMMENSURABLE | +4.264 | [-0.659, +9.188] | 0/5 | **not separated** |

Code commits across included runs: ['e7beb6a4f0a7b25298e44b37083fcdcb8ebeade9']

## Primary: 80/20 split, validation-tuned (S1 kernel/naive runs + S4 LSTM runs)

Scope: validation-tuned (S3 rule), 5 seeds; kernel models and ridge selected the S1 configuration; LSTMs lr 0.01 (S4).

| System | Role | Provenance | Scope | Successful / expected runs | Test RMSE (mean) | Std | Matching basis | Trainable params | Wall-clock / run (min) | Status |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| B0 persistence | naive floor (uses past AQI) | local run (s1_paper_hp_20261021-134301:B0_persistence) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 11.415 | 0.000 | 0 params; PARTIALLY_COMMENSURABLE | 0 | 0.0 | COMPLETE |
| B2 LSTM(3) | fair baseline: iso-parameter | local run (s4_tuned_test_20261021-152814:B2_lstm3_lr0.01) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 11.717 | 2.139 | params within ±10% of Q-B (196/209) | 196 | 0.2 | COMPLETE |
| B0b ridge | linear baseline | local run (s1_paper_hp_20261021-134301:B0b_ridge) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 12.425 | 0.000 | 34 params | 34 | 0.0 | COMPLETE |
| B1 LSTM(16) | paper baseline (reproduced) | local run (s4_tuned_test_20261021-152814:B1_lstm16_lr0.01) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 12.882 | 2.778 | paper architecture (1,873 params) | 1873 | 0.2 | COMPLETE |
| Q-A QK-LSTM (qubit) | reproduced method (Eq. 6 reading) | local run (s1_paper_hp_20261021-134301:QA_quantum_eq6_N4) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 15.661 | 3.771 | reference (449 params) | 449 | 9.4 | COMPLETE |
| Q-B QK-LSTM (qubit) | reproduced method | local run (s1_paper_hp_20261021-134301:QB_quantum_209) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 15.679 | 3.965 | reference (209 params) | 209 | 10.6 | COMPLETE |
| B3-A RBF-kernel LSTM | fair baseline: kernel ablation | local run (s1_paper_hp_20261021-134301:B3A_rbf_eq6_N4) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 15.927 | 3.855 | identical to Q-A except kernel | 449 | 0.7 | COMPLETE |
| B3-B RBF-kernel LSTM | fair baseline: kernel ablation | local run (s1_paper_hp_20261021-134301:B3B_rbf_209) | validation-tuned (S3 rule), 5 seeds | 5 / 5 | 16.006 | 3.801 | identical to Q-B except kernel | 209 | 0.7 | COMPLETE |
| QK-LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 9.20 | — | — | 209 | — | reference only |
| LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 15.94 | — | — | 1873 | — | reference only |

Paired decisions (frozen rule: 95 % t-CI of mean ΔRMSE, A − B):

| Claim | A | B | Commensurability | ΔRMSE mean | 95 % CI | A better (seeds) | Decision |
|---|---|---|---|---:|---|---:|---|
| C1 accuracy vs paper baseline | Q-B QK-LSTM (qubit) | B1 LSTM(16) | COMMENSURABLE | +2.797 | [+0.496, +5.098] | 0/5 | **A worse** |
| C1 accuracy vs paper baseline | Q-A QK-LSTM (qubit) | B1 LSTM(16) | COMMENSURABLE | +2.779 | [+0.681, +4.877] | 0/5 | **A worse** |
| C2 iso-parameter accuracy | Q-B QK-LSTM (qubit) | B2 LSTM(3) | COMMENSURABLE | +3.963 | [+1.372, +6.554] | 0/5 | **A worse** |
| C4 quantum vs classical kernel | Q-B QK-LSTM (qubit) | B3-B RBF-kernel LSTM | COMMENSURABLE | -0.326 | [-0.591, -0.062] | 5/5 | **A better** |
| C4 quantum vs classical kernel | Q-A QK-LSTM (qubit) | B3-A RBF-kernel LSTM | COMMENSURABLE | -0.266 | [-0.452, -0.080] | 5/5 | **A better** |
| naive floor | Q-B QK-LSTM (qubit) | B0 persistence | PARTIALLY_COMMENSURABLE | +4.264 | [-0.659, +9.188] | 0/5 | **not separated** |

Code commits across included runs: ['e7beb6a4f0a7b25298e44b37083fcdcb8ebeade9']

## Sensitivity: calendar-2020 test, paper hyperparameters (S2)

Scope: paper-accurate (paper hp, 5 seeds); test 2020-01-01..2020-07-01 (183 days).

| System | Role | Provenance | Scope | Successful / expected runs | Test RMSE (mean) | Std | Matching basis | Trainable params | Wall-clock / run (min) | Status |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| B0b ridge | linear baseline | local run (s2_split2020_20261021-142942:B0b_ridge) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 9.943 | 0.000 | 34 params | 34 | 0.0 | COMPLETE |
| B0 persistence | naive floor (uses past AQI) | local run (s2_split2020_20261021-142942:B0_persistence) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 10.455 | 0.000 | 0 params; PARTIALLY_COMMENSURABLE | 0 | 0.0 | COMPLETE |
| B1 LSTM(16) | paper baseline (reproduced) | local run (s2_split2020_20261021-142942:B1_lstm_1873) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 11.325 | 1.656 | paper architecture (1,873 params) | 1873 | 0.3 | COMPLETE |
| B2 LSTM(3) | fair baseline: iso-parameter | local run (s2_split2020_20261021-142942:B2_lstm_matched_196) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 11.619 | 1.777 | params within ±10% of Q-B (196/209) | 196 | 0.3 | COMPLETE |
| Q-B QK-LSTM (qubit) | reproduced method | local run (s2_split2020_20261021-142942:QB_quantum_209) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.318 | 2.614 | reference (209 params) | 209 | 5.5 | COMPLETE |
| B3-B RBF-kernel LSTM | fair baseline: kernel ablation | local run (s2_split2020_20261021-142942:B3B_rbf_209) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.327 | 2.467 | identical to Q-B except kernel | 209 | 0.3 | COMPLETE |
| B3-A RBF-kernel LSTM | fair baseline: kernel ablation | local run (s2_split2020_20261021-142942:B3A_rbf_eq6_N4) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.434 | 2.728 | identical to Q-A except kernel | 449 | 0.5 | COMPLETE |
| Q-A QK-LSTM (qubit) | reproduced method (Eq. 6 reading) | local run (s2_split2020_20261021-142942:QA_quantum_eq6_N4) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.619 | 2.901 | reference (449 params) | 449 | 5.4 | COMPLETE |
| QK-LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 9.20 | — | — | 209 | — | reference only |
| LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 15.94 | — | — | 1873 | — | reference only |

Paired decisions (frozen rule: 95 % t-CI of mean ΔRMSE, A − B):

| Claim | A | B | Commensurability | ΔRMSE mean | 95 % CI | A better (seeds) | Decision |
|---|---|---|---|---:|---|---:|---|
| C1 accuracy vs paper baseline | Q-B QK-LSTM (qubit) | B1 LSTM(16) | COMMENSURABLE | +0.993 | [-1.251, +3.238] | 2/5 | **not separated** |
| C1 accuracy vs paper baseline | Q-A QK-LSTM (qubit) | B1 LSTM(16) | COMMENSURABLE | +1.294 | [-1.303, +3.891] | 1/5 | **not separated** |
| C2 iso-parameter accuracy | Q-B QK-LSTM (qubit) | B2 LSTM(3) | COMMENSURABLE | +0.699 | [-0.417, +1.816] | 2/5 | **not separated** |
| C4 quantum vs classical kernel | Q-B QK-LSTM (qubit) | B3-B RBF-kernel LSTM | COMMENSURABLE | -0.009 | [-0.375, +0.357] | 2/5 | **not separated** |
| C4 quantum vs classical kernel | Q-A QK-LSTM (qubit) | B3-A RBF-kernel LSTM | COMMENSURABLE | +0.185 | [-0.255, +0.625] | 2/5 | **not separated** |
| naive floor | Q-B QK-LSTM (qubit) | B0 persistence | PARTIALLY_COMMENSURABLE | +1.864 | [-1.382, +5.109] | 2/5 | **not separated** |

Code commits across included runs: ['e7beb6a4f0a7b25298e44b37083fcdcb8ebeade9']

## MerLin adaptation (simulation) vs Phase 3, 80/20, paper hyperparameters (S6 + S1)

Scope: MerLin adaptation (ideal simulation: FOCK computation, threshold-equivalent detection, 2 photons / 4 modes, shots 0), paired by seed with S1. Wall-clock PARTIALLY_COMMENSURABLE (S1 contended CPU, S6 not).

| System | Role | Provenance | Scope | Successful / expected runs | Test RMSE (mean) | Std | Matching basis | Trainable params | Wall-clock / run (min) | Status |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| B1 LSTM(16) | paper baseline (reproduced) | local run (s1_paper_hp_20261021-134301:B1_lstm_1873) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 12.611 | 2.661 | paper architecture | 1873 | 0.6 | COMPLETE |
| B2 LSTM(3) | fair baseline: iso-parameter | local run (s1_paper_hp_20261021-134301:B2_lstm_matched_196) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 13.351 | 3.615 | 196/209 params | 196 | 0.6 | COMPLETE |
| Q-B photonic 1-photon | ablation (not the photonic implementation) | local run (s6_merlin_20261021-183822:QB_photonic_1ph_ablation) | MerLin simulation, 5 seeds | 5 / 5 | 15.524 | 3.980 | same as Q-B; 1 photon | 209 | 5.2 | COMPLETE |
| Q-B QK-LSTM (qubit) | reproduced method | local run (s1_paper_hp_20261021-134301:QB_quantum_209) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 15.679 | 3.965 | reference | 209 | 10.6 | COMPLETE |
| Q-B QK-LSTM photonic (2 ph) | MerLin adaptation | local run (s6_merlin_20261021-183822:QB_photonic_2ph_209) | MerLin simulation, 5 seeds | 5 / 5 | 15.728 | 3.722 | same as Q-B; kernel photonic (0 circuit params) | 209 | 6.2 | COMPLETE |
| Q-A QK-LSTM photonic (2 ph) | MerLin adaptation | local run (s6_merlin_20261021-183822:QA_photonic_2ph_eq6_N4) | MerLin simulation, 5 seeds | 5 / 5 | 15.737 | 4.073 | same as Q-A; kernel photonic | 449 | 5.7 | COMPLETE |
| B3-B RBF-kernel LSTM | fair baseline: kernel ablation | local run (s1_paper_hp_20261021-134301:B3B_rbf_209) | paper-accurate (paper hp, 5 seeds) | 5 / 5 | 16.006 | 3.801 | identical except kernel | 209 | 0.7 | COMPLETE |
| QK-LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 9.20 | — | — | 209 | — | reference only |
| LSTM (paper) | paper-reported | paper Table I/II (single run) | INCOMPARABLE for ranking (protocol unknown) | 1 / 1 | 15.94 | — | — | 1873 | — | reference only |

Paired decisions (frozen rule: 95 % t-CI of mean ΔRMSE, A − B):

| Claim | A | B | Commensurability | ΔRMSE mean | 95 % CI | A better (seeds) | Decision |
|---|---|---|---|---:|---|---:|---|
| MerLin preserves behaviour (tol ±1.0) | Q-B QK-LSTM photonic (2 ph) | Q-B QK-LSTM (qubit) | COMMENSURABLE | +0.049 | [-0.616, +0.713] | 3/5 | **not separated** |
| C1 (photonic) vs paper baseline | Q-B QK-LSTM photonic (2 ph) | B1 LSTM(16) | COMMENSURABLE | +3.117 | [+1.321, +4.912] | 0/5 | **A worse** |
| C2 (photonic) iso-parameter | Q-B QK-LSTM photonic (2 ph) | B2 LSTM(3) | COMMENSURABLE | +2.377 | [+1.801, +2.954] | 0/5 | **A worse** |
| C4 photonic vs classical kernel | Q-B QK-LSTM photonic (2 ph) | B3-B RBF-kernel LSTM | COMMENSURABLE | -0.278 | [-0.720, +0.165] | 4/5 | **not separated** |
| multi-photon interference | Q-B QK-LSTM photonic (2 ph) | Q-B photonic 1-photon | COMMENSURABLE | +0.204 | [-0.746, +1.154] | 2/5 | **not separated** |

Code commits across included runs: ['e7beb6a4f0a7b25298e44b37083fcdcb8ebeade9']

