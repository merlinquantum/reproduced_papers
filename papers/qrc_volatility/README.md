# Quantum Reservoir Computing for Realized Volatility Forecasting — Reproduction

## Reference and Attribution

- **Paper**: *Quantum Reservoir Computing for Realized Volatility Forecasting*
- **Authors**: Qingyu Li, Chiranjib Mukhopadhyay, Abolfazl Bayat, Ali Habibnia
- **Preprint**: [arXiv:2505.13933](https://arxiv.org/abs/2505.13933) (v2, 9 Apr 2026, quant-ph)
- **Original repository**: <https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting> (commit `d2e9b0a`; Julia + Jupyter)
- **Attribution**: this folder is an independent Python reimplementation. The
  authors' `Data.CSV` feature panel, their 100 saved reservoir coupling matrices
  (`coeff_10.jld2`) and their published QR1/QR2 forecasts (`predict_result.csv`)
  are used as inputs and as regression-test ground truth. No licence file is
  present in the upstream repository; cite the preprint when using these results.

## Original Paper

The paper forecasts the monthly realized volatility (RV) of the S&P 500 index
from February 1950 to December 2017 with a **quantum reservoir computer**.

- **Reservoir**: 10 qubits under a fixed, fully connected transverse-field Ising
  Hamiltonian `H = sum_{i<j} J_ij X_i X_j + v sum_i Z_i` with `v = 1`, split into
  `n1 = 7` *input* qubits and `n2 = 3` *hidden* (memory) qubits.
- **Encoding**: at each of three lags (`t-3`, `t-2`, `t-1`) the seven selected
  macro-financial features are encoded on fresh input qubits with `RY(pi * x)`,
  the whole register evolves for `tau = 1/v`, and the input qubits are then traced
  out so the hidden qubits carry memory forward.
- **Readout**: after the final lag all ten qubits are measured in the Pauli-Z
  basis and the expectation vector feeds a **ridge regression** (`delta = 1e-8`)
  predicting `log RV_t`. Only the readout is trained. **QR1** uses one reservoir
  (10 readout features); **QR2** ensembles two readouts taken at `tau/2` and
  `tau` (20 features).
- **Protocol**: rolling one-step-ahead re-estimation with a 571-month window,
  245 out-of-sample forecasts from August 1997 to December 2017, at horizons
  `S = 1` (open loop) and `S = 5` (closed loop, with the model's own predictions
  fed back while exogenous variables stay at ground truth).
- **Benchmarks**: HAR, HARX, AR1, AR3, ARMAX, LSTM, LSTMX, RC, RCX, scored by MSE
  and QLIKE with Model Confidence Set and Diebold-Mariano tests.
- **Main claim**: the quantum reservoir "consistently outperforms benchmark
  models across various metrics" (abstract), with QR2 best on every measure at
  `S = 1` (Table II). The authors explicitly do **not** claim a proven quantum
  advantage. Two supporting analyses select the input features by wrapper forward
  selection (Fig. 6) and interpret them with Shapley values (Fig. 8). Reported
  results are **the best of 100 random reservoir draws** (Sec. IV.B).

## Reproduction Scope (including Updates and Deviations)

### What is reproduced

| Item | Status |
|---|---|
| QR1 and QR2 at the paper's settings, instances and feature sets, `S = 1` | **Quantitatively reproduced** (see table below) |
| The authors' own saved 245 QR1/QR2 forecasts | **Reproduced to float32 precision** (`tests/test_qrc_matches_reference.py`, `atol = 1e-3`) |
| HAR, HARX, AR1, AR3, ARMAX classical baselines | Reproduced (HAR to four decimals) |
| RC, RCX classical reservoir baselines | Reproduced qualitatively (own NumPy echo state network, deviation D4) |
| Model Confidence Set and Diebold-Mariano tests | Reproduced (bootstrap seed unstated upstream) |
| `S = 5` closed-loop horizon | **Not quantitatively reproduced** — our losses are uniformly lower; protocol under-specified (F5/F8) |
| Distribution over all 100 reservoir instances | **New** (the paper reports only the best) |
| Correctly indexed HAR/HARX | **New** (fixes an indexing defect in the released code) |
| Iso-readout-dimension classical controls under the paper's own selection protocol | **New** |
| Wrapper forward feature selection (Fig. 6a) | **Reproduced exactly** (same set, same order, same optimum at `n1 = 7`); reduced scope (QR1 only) |
| MerLin photonic adaptation | **New**, `PARTIAL_MERLIN_TRANSLATION` |
| LSTM / LSTMX baselines | **Not run** (deviation D6) |
| Shapley feature importance (Fig. 8) | **Not run** (deviation D7) |

### Updates and deviations

1. **Data.** The authors publish only the *normalised* `Data.CSV`; their
   `Data_raw.csv` and `dff.csv` are missing upstream. Raw `log RV` is recovered
   exactly by inverting the `Min_RV`/`Max_RV` constants hard-coded in their
   `Time_series.jl`, and their ADF-based differencing rule is replayed (the ADF
   statistic is scale-invariant, so the decision is recovered exactly and
   reproduces their `diff_DP` / `diff_TB` naming). Validated by HAR reproducing
   to four decimals. Consequence: the neural and classical-reservoir baselines see
   a differently scaled input than the authors' (harmless for the OLS models,
   which are affine-invariant).
2. **Coupling draw.** The paper's Eq. 12 says `J_ij/v ~ U[0,1]`; the released
   `coeff_matrix` additionally rescales the matrix so its largest eigenvalue is 1.
   This reproduction uses the authors' saved matrices, and
   `lib.data.sample_coupling_instances` replicates the released rule for fresh
   draws.
3. **QLIKE.** The paper prints a Patton-style QLIKE formula but the released code
   computes `sum(r - log r - 1)` with `r = |RV| / |RV_hat|` on *log* RV values.
   Only the latter reproduces Table II, so it is what this folder reports, clearly
   labelled.
4. **RC/RCX** are a NumPy leaky-integrator echo state network rather than
   `reservoirpy`, to keep the rolling protocol byte-identical to the quantum path.
5. **`S = 5`** uses 241 forecast paths (1997-12..2017-12) rather than the paper's
   240; the paper does not state which origin starts the closed-loop chain.
6. **LSTM/LSTMX not run**: 245 rolling refits x 100 epochs x 2 models is the
   single most expensive item and neither model is near the decision boundary in
   the paper's own table. They are reported as not-run, not as reproduced.
7. **Shapley analysis not reproduced** (interpretability only).

Full source-evidence table, deviation IDs and open questions: `LOG.md`.

## Install and How to Run

```bash
pip install -r requirements.txt
```

Place the authors' data (gitignored) under `data/qrc_volatility/`:

```bash
git clone https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting /tmp/qrc_ref
mkdir -p data/qrc_volatility
cp /tmp/qrc_ref/Data.CSV /tmp/qrc_ref/coeff_10.jld2 data/qrc_volatility/
cp /tmp/qrc_ref/predict_result.csv data/qrc_volatility/authors_qr_predictions.csv
```

Quick smoke check (~1 min):

```bash
python implementation.py --paper qrc_volatility --config configs/smoke.json
python implementation.py --paper qrc_volatility --config configs/photonic_smoke.json
```

Full reproduction (from the repository root):

```bash
# Paper Table II / Table III, S = 1 and S = 5, all baselines and fair controls (~6 min)
python implementation.py --paper qrc_volatility --config configs/table2_no_lstm.json

# Distribution over all 100 published reservoir instances, QR1 and QR2 (~45 min)
python implementation.py --paper qrc_volatility --config configs/instance_sweep.json

# MerLin photonic adaptation, 2 variants x 4 encoding scales x 25 instances (~18 min)
python implementation.py --paper qrc_volatility --config configs/photonic.json

# Wrapper forward feature selection for QR1, paper protocol (~15 min)
python implementation.py --paper qrc_volatility --config configs/feature_selection_qr1.json

# Regenerate every README table from the run artefacts
python papers/qrc_volatility/utils/make_tables.py \
  --table2 papers/qrc_volatility/outdir/run_<TABLE2> \
  --instance-sweep papers/qrc_volatility/outdir/run_<SWEEP> \
  --photonic papers/qrc_volatility/outdir/run_<PHOTONIC> \
  --feature-selection papers/qrc_volatility/outdir/run_<FSEL> \
  --out papers/qrc_volatility/results
```

Add `--config configs/defaults.json` (the default) to include the two LSTM
baselines; expect roughly an extra hour of CPU time.

## Configuration

`cli.json` is the authoritative CLI schema. Key flags:

| Flag | Meaning |
|---|---|
| `--experiment` | `paper_table2`, `reservoir_instance_sweep`, `feature_selection_sweep`, `photonic` |
| `--selection-split` | `validation` (default, leakage-free) or `test` (reproduces the paper's protocol) |
| `--n-instances` | Reservoir draws in the instance sweep (paper: 100) |
| `--coupling-source` | `authors` (saved `coeff_10.jld2`) or `fresh` (re-drawn) |
| `--n-validation` | Length of the leakage-free validation window carved out of the training sample (default 120 months) |
| `--max-features` | Cap for forward selection (paper: 10) |
| `--mcs-reps` | MCS bootstrap replications (authors' code: 10000) |
| `--tau`, `--ridge-delta`, `--n-lags`, `--n-qubits`, `--horizons`, `--n-out-of-sample` | Reservoir and protocol overrides |

Named configs live in `configs/`; each carries a `description` stating whether it
is paper-accurate or reduced.

## Data

Repository data root: `data/qrc_volatility/` (gitignored).

- `Data.CSV` — the authors' monthly panel, 816 rows (1950-01..2017-12), 15
  min-max normalised columns: `RV`, `RV_q`, `RV_a`, `RV1`, `RV2`, `DP`, `EP`,
  `MKT`, `SMB`, `HML`, `TB`, `DEF`, `IP`, `INF`, `STR`. `RV` is normalised to
  `[-1, 0]`; `log RV` is recovered with `lib.data.denormalise_log_rv`.
- `coeff_10.jld2` — 100 saved 10x10 symmetric zero-diagonal coupling matrices,
  each rescaled to unit spectral radius.
- `authors_qr_predictions.csv` — the authors' 245 QR1/QR2 forecasts, used by
  `tests/test_qrc_matches_reference.py`.

## Results Obtained and Comparison with the Paper

All numbers below are produced by `utils/make_tables.py` reading `metrics.json`
and `sweep_summary.json`; the curated copies are in `results/`. MSE and QLIKE are
on raw `log RV`; QLIKE is the authors' code definition (deviation D3).

### Table II, one-step-ahead (`S = 1`, 245 forecasts, 1997-08..2017-12)

Rows marked **new** are additions by this reproduction. `MCS p` is the Model
Confidence Set p-value under squared-error loss.

| Model | Reproduced MSE | Paper MSE | Reproduced QLIKE | Paper QLIKE | MCS p | Paper MCS p |
|---|---:|---:|---:|---:|---:|---:|
| ESN-iso-20, best of 100 on test **(new)** | **0.0974** | – | **1.3353** | – | 1.0000 | – |
| ESN-iso-10, best of 100 on test **(new)** | 0.1009 | – | 1.3899 | – | 0.4905 | – |
| HARX, corrected indexing **(new)** | 0.1016 | – | 1.3709 | – | 0.4905 | – |
| ESN-iso-20, validation-selected **(new)** | 0.1025 | – | 1.4102 | – | 0.4010 | – |
| Linear-lag ridge, no reservoir **(new)** | 0.1031 | – | 1.4202 | – | 0.0474 | – |
| ESN-iso-10, validation-selected **(new)** | 0.1033 | – | 1.4362 | – | 0.2437 | – |
| RCX | 0.1034 | 0.1089 | 1.4615 | 1.6480 | 0.4335 | 0.6086 |
| **QR2** (authors' published instance) | 0.1038 | 0.1030 | 1.4004 | 1.4004 | 0.4905 | 1.0000 |
| **QR1** (authors' published instance) | 0.1051 | 0.1050 | 1.4427 | 1.4427 | 0.4010 | 0.7603 |
| ARMAX | 0.1073 | 0.1145 | 1.4623 | 1.6196 | 0.2482 | 0.4406 |
| HAR, corrected indexing **(new)** | 0.1157 | – | 1.5688 | – | 0.0014 | – |
| AR3 | 0.1179 | 0.1178 | 1.5867 | 1.5893 | 0.0014 | 0.0936 |
| RC | 0.1180 | 0.1441 | 1.5995 | 2.1011 | 0.0000 | 0.0084 |
| AR1 | 0.1301 | 0.1304 | 1.7169 | 1.7279 | 0.0012 | 0.0065 |
| HARX, as published | 0.1448 | 0.1508 | 2.0245 | 2.2436 | 0.0001 | 0.0004 |
| HAR, as published | 0.1477 | 0.1476 | 2.0432 | 2.0431 | 0.0000 | 0.0004 |
| LSTM / LSTMX | not run | 0.1295 / 0.1185 | not run | 1.7909 / 1.7571 | – | 0.0221 / 0.4406 |

The quantum rows reproduce the paper essentially exactly (QR1 0.1051 vs 0.105,
QLIKE 1.4427 vs 1.4427; QR2 0.1038 vs 0.103, QLIKE 1.4004 vs 1.4004), and so do
HAR, AR1 and AR3. **But six classical models now beat both quantum reservoirs.**

### Why the ranking changes: two baseline defects

**1. The published HAR and HARX losses are inflated by a one-month indexing
error.** In `Reservoir_Learning.ipynb` the rolling loop predicts from
`dff.iloc[end:end+1]` while scoring against `dff.iloc[-245:]`, so every HAR-family
forecast is compared with the wrong month. The AR, ARMAX, LSTM, classical-RC and
quantum paths in the same repository are correctly aligned, so only the HAR family
is affected. Correcting the indexing without changing the window or introducing
look-ahead:

| Model | As published | Corrected | Paper value |
|---|---:|---:|---:|
| HAR MSE | 0.1477 | **0.1157** | 0.1476 |
| HAR QLIKE | 2.0432 | **1.5688** | 2.0431 |
| HARX MSE | 0.1448 | **0.1016** | 0.1508 |
| HARX QLIKE | 2.0245 | **1.3709** | 2.2436 |

A correctly indexed HARX — ordinary least squares on three HAR terms and four
lagged exogenous regressors — beats both quantum reservoirs on both metrics.

**2. The reported quantum results are ~5th-percentile draws.** The paper selects
the best of 100 reservoirs. Running all 100 published coupling matrices:

| Variant | Runs | Mean MSE | SD | Median | Min | Max | 5th pct | Paper's value | Validation-selected (test MSE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QR1 | 100/100 | 0.1095 | 0.0026 | 0.1094 | 0.1032 | 0.1142 | 0.1051 | **0.1050** | 0.1063 (inst. 50) |
| QR2 | 100/100 | 0.1098 | 0.0036 | 0.1095 | 0.1018 | 0.1168 | 0.1037 | **0.1030** | 0.1073 (inst. 96) |

A *typical* reservoir draw scores 0.1095-0.1098, worse than the paper's own best
classical baseline (RCX, 0.1089). Applying the identical best-of-100 protocol to a
classical control — an echo state network with exactly 10 or 20 readout units, the
same seven features, the same three-step window and the same rolling ridge —
gives 0.1009 and **0.0974**, better than the quantum best-of-100. Under
leakage-free validation selection the classical controls still win
(0.1025 / 0.1033 versus 0.1063 / 0.1073).

The QR2-minus-QR1 gap in mean MSE is 0.0003, about 0.1 across-instance standard
deviations, so QR2 is not measurably better than QR1.

### The task is close to linear

A plain rolling ridge on the 21 raw lagged features the reservoir is given
(`Linear-lag`, 22 parameters) scores 0.1031 — better than both quantum variants.
In the photonic adaptation, accuracy improves monotonically as the encoding phase
scale shrinks toward the linear limit. Monthly log realized volatility is well
described by a linear function of its recent lags plus a few macro-financial
regressors, which leaves a fixed nonlinear feature map little to add.

### Five-step closed loop (`S = 5`, 241 forecasts)

| Model | Reproduced MSE | Paper MSE | MCS p |
|---|---:|---:|---:|
| QR1 | 0.1181 | 0.1556 | 1.0000 |
| ARMAX | 0.1218 | 0.2134 | 0.7343 |
| ESN-iso-10, best on test **(new)** | 0.1237 | – | 0.7343 |
| QR2 | 0.1286 | 0.1663 | 0.6973 |
| RCX | 0.1447 | 0.1667 | 0.3086 |
| HARX, corrected **(new)** | 0.1453 | – | 0.0514 |
| HAR, as published | 0.2061 | 0.2143 | 0.0052 |
| AR3 | 0.2139 | 0.2134 | 0.0127 |
| RC | 0.2197 | 0.1528 | 0.0080 |
| AR1 | 0.2633 | 0.2642 | 0.0127 |

**`S = 5` is not quantitatively reproduced.** Every model whose forecast depends
on feedback scores materially better here than in the paper (QR1 0.1181 vs
0.1556, ARMAX 0.1218 vs 0.2134), while the purely autoregressive models match
(AR1 0.2633 vs 0.2642, AR3 0.2139 vs 0.2134). That pattern points at an
unstated difference in the closed-loop protocol, not at an implementation error
in the autoregressive path. The *trend* also disagrees: the paper ranks classical
RC first at `S = 5`, this reproduction ranks QR1 first — but with no MCS
separation from ARMAX or the iso-readout classical control. Treat `S = 5` as
**unresolved** (failure classes F5, F8).

### MerLin photonic adaptation

Photonic status: **`PARTIAL_MERLIN_TRANSLATION`**. The frozen-reservoir
semantics, the `RY(pi x)` -> phase-shifter encoding, the lag-ordered structure,
the 7-encoded / 3-hidden mode split, the `<Z_j>` -> `<n_j>` readout, the QR2
two-readout ensemble, the rolling ridge and the whole evaluation protocol are
translated. The **partial trace that discards the input qubits is not**:
`merlin.QuantumLayer` accepts only a pure Fock `input_state`, so a mixed hidden
state cannot be fed back, and `MeasurementStrategy.partial` exposes the branch
decomposition but no way to re-enter it. The photonic register therefore retains
information the qubit reservoir throws away, which makes this an upper bound on
the qubit architecture rather than a copy of it. Details and the evidence for the
status: `lib/photonic.py` module docstring and `LOG.md`.

| Field | PQR1 | PQR2 |
|---|---|---|
| Computation space | UNBUNCHED | UNBUNCHED |
| Detector model | threshold (unbunched subspace) | threshold (unbunched subspace) |
| Photon number | 3 | 3 |
| Number of modes | 10 | 10 |
| Input state | `[1,0,0,1,0,0,1,0,0,0]` | `[1,0,0,1,0,0,1,0,0,0]` |
| Encoding | angle, modes 0-6, 3 sequential lag blocks, scale selected on validation | same |
| Measurement strategy | `MeasurementStrategy.mode_expectations` | `MeasurementStrategy.mode_expectations` |
| Readout width / trainable parameters | 10 | 20 |
| Frozen circuit parameters | 360 | 720 |
| Postselection | none | none |
| Simulator / QPU | MerLin 0.4.0 CPU simulator (analytic statevector) | same |
| Shot count | n/a (`shots = 0`) | n/a (`shots = 0`) |
| Wall clock, 813-month pass | ~0.09 s | ~0.18 s |
| Seeds | 25 mesh seeds x 4 encoding scales | 25 mesh seeds x 4 encoding scales |

Wall-clock figures are from an otherwise idle 8-core container; the same sweep
took roughly 10x longer in an earlier session that ran it concurrently with the
feature-selection sweep. Treat them as an order of magnitude, not a benchmark.

Results (see `results/photonic_summary.csv` and `results/photonic_by_scale.csv`
for the machine-generated versions, and `results/photonic_hardware.json` for the
selected candidate's full hardware block):

| Variant | Readout width | Mean MSE over 100 candidates | SD | Best on test | Validation-selected test MSE | Validation-selected QLIKE | `S=5` at that candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| PQR1 | 10 | 0.1715 | 0.0425 | 0.1067 | 0.1067 | 1.4830 | 0.1598 |
| PQR2 | 20 | 0.1372 | 0.0283 | **0.1004** | 0.1209 | 1.6519 | 0.1695 |

Encoding-scale dependence (mean test MSE over 25 mesh seeds each):

| Encoding scale | PQR1 | PQR2 |
|---|---:|---:|
| `pi` | 0.2070 | 0.1765 |
| `pi/2` | 0.1743 | 0.1370 |
| `pi/4` | 0.1565 | 0.1217 |
| `pi/8` | **0.1482** | **0.1136** |

**Photonic findings.**

1. **A 10-mode, 3-photon linear-optical reservoir is as good a feature map for
   this task as the 10-qubit Ising reservoir.** Under the paper's own best-of-N
   selection protocol the photonic ensemble PQR2 reaches 0.1004, slightly better
   than the qubit QR2's best of 100 (0.1018) and than its published value
   (0.1030). Under leakage-free validation selection PQR1 reaches 0.1067,
   statistically indistinguishable from the qubit QR1's 0.1063.
2. **Photonic instance-to-instance spread is an order of magnitude larger**
   (SD 0.028-0.043 versus 0.003-0.004 for the qubit reservoir), so the photonic
   reservoir is far more sensitive to the draw and the best-of-N selection
   protocol flatters it correspondingly more. This is the clearest illustration in
   this reproduction of why the paper's selection protocol matters.
3. **Accuracy improves monotonically as the encoding phase scale is reduced
   toward the linear limit**, at both variants and every scale step
   (PQR1 0.2070 -> 0.1482; PQR2 0.1765 -> 0.1136). A phase shifter contributes
   `exp(i pi x)` and is `2 pi`-periodic, whereas `RY(pi x)` is `4 pi`-periodic in
   its argument, so copying the gate-model scale maps `x = -1` and `x = +1` to the
   same phase and folds half the feature range. Beyond that fix, the trend is
   direct evidence that the *nonlinearity* of the reservoir map is not what helps
   here — the same conclusion the `Linear-lag` control reaches in the qubit
   setting.
4. **Readout width matters more than the dynamics.** Doubling the readout from 10
   to 20 features improves the photonic mean from 0.1715 to 0.1372, while in the
   qubit setting the same doubling (QR1 -> QR2) moves the mean from 0.1095 to
   0.1098, i.e. not at all.
5. Neither photonic variant beats the corrected or selection-matched classical
   controls (HARX-aligned 0.1016, ESN-iso-20 0.1025-0.0974), so the photonic
   translation reproduces the paper's *behaviour* without rescuing its *claim*.

### Feature selection (paper Fig. 6a, claims C5 and C6) — reproduced

Greedy wrapper forward selection over the 13-column pool, reduced scope (QR1
only), 85 candidates, `SWEEP_COMPLETED`:

| k | Feature added | Test MSE | Test QLIKE | Validation MSE |
|---:|---|---:|---:|---:|
| 1 | RV | 0.1187 | 1.6194 | 0.1199 |
| 2 | MKT | 0.1115 | 1.5397 | 0.1161 |
| 3 | DP | 0.1102 | 1.5248 | 0.1116 |
| 4 | IP | 0.1120 | 1.5509 | 0.1148 |
| 5 | RV_q | 0.1125 | 1.5540 | 0.1116 |
| 6 | STR | 0.1062 | 1.4565 | 0.1096 |
| 7 | DEF | **0.1051** | **1.4427** | **0.1079** |
| 8 | INF | 0.1075 | 1.4768 | 0.1103 |
| 9 | RV_a | 0.1052 | 1.4458 | 0.1100 |
| 10 | SMB | 0.1090 | 1.5025 | 0.1106 |

This is the paper's strongest reproduced result. The greedy path recovers the
published optimal set **exactly and in the same order** —
`{RV, MKT, DP, IP, RVq, STR, DEF}` (paper Sec. IV.D) — and the MSE curve improves
then degrades with a minimum at `n1 = 7`, exactly the shape and optimum of
Fig. 6(a). Crucially the optimum is at `k = 7` under *both* the paper's test-split
scoring and the leakage-free validation scoring, so unlike the headline
comparison this claim does not depend on test-set selection. Claims **C5 and C6
are supported.**

### Verdict

| Aspect | Assessment |
|---|---|
| Metric agreement | **Quantitatively reproduced** at `S = 1` for QR1, QR2, HAR, AR1, AR3; close for HARX, ARMAX, RCX; **not reproduced** at `S = 5` |
| Feature-selection claims (C5, C6) | **Supported** — the optimal set and the `n1 = 7` optimum reproduce exactly, under both scoring splits |
| Implementation confidence | **HIGH** — the reimplementation matches the authors' own saved forecasts to float32 precision |
| Trend agreement | **Not reproduced.** With corrected and selection-matched baselines the reported quantum-over-classical ranking reverses |
| Claim support | **Unsupported.** The central claim that QRC "consistently outperforms benchmark models" does not survive (i) correcting the HAR-family indexing defect and (ii) giving classical controls the same best-of-100 selection budget |
| Reproducibility confidence | **HIGH** (the numbers reproduce; the conclusion does not) |
| Failure classes | **F6** baseline unfairness (primary), **F5** metric/protocol ambiguity (QLIKE definition, MCS seed, `S = 5` protocol), **F8** reproduction divergence at `S = 5` |
| Photonic status | **`PARTIAL_MERLIN_TRANSLATION`** |

This is a *negative reproduction with high implementation confidence*, which is
evidence against the paper's comparative claim rather than an inconclusive
result. The paper's own hedging is worth restating: the authors do not claim a
proven quantum advantage, and this reproduction supports that caution more
strongly than their table does.

## Limitations

- LSTM and LSTMX are not run, so two of the paper's eleven rows are absent. Both
  are well behind the decision boundary in the paper's own table.
- The Shapley interpretability analysis (Fig. 8) is not reproduced.
- Forward selection is reproduced for QR1 only.
- `S = 5` remains unresolved; a definitive comparison needs the authors' exact
  closed-loop indexing.
- The instance sweep uses one run per candidate (the reservoir and readout are
  deterministic given the coupling matrix, so the 100 instances *are* the
  replication) and only the authors' published coupling matrices; the
  fresh-draw variant is implemented but not run.
- MCS p-values are not bit-reproducible because the authors leave the bootstrap
  seed unset.
- All quantum and photonic results are shot-free analytic simulation; no
  finite-shot or hardware-noise study was performed, so the paper's Appendix C
  concentration argument is not independently tested.

## Tests

```bash
cd papers/qrc_volatility
pytest -q
```

- `tests/test_qrc_matches_reference.py` — pins the reservoir simulator against
  the authors' published `predict_result.csv` and against Table II, and asserts
  that the published HAR loss is reproducible *only* with the reference
  misalignment.
- `tests/test_cli.py` — CLI schema, paper-accurate defaults, at-least-two-photon
  guard for the photonic variants, run-evidence bundle, loud failure on an
  unknown experiment.
- `tests/test_smoke.py` — shared-runtime override wiring.

## Citation and License

```bibtex
@misc{li2026qrcvolatility,
  title  = {Quantum Reservoir Computing for Realized Volatility Forecasting},
  author = {Li, Qingyu and Mukhopadhyay, Chiranjib and Bayat, Abolfazl and Habibnia, Ali},
  year   = {2026},
  eprint = {2505.13933},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url    = {https://arxiv.org/abs/2505.13933}
}
```

Reproduction code follows the repository licence (see `/LICENSE`). The authors'
`Data.CSV`, `coeff_10.jld2` and `predict_result.csv` are not redistributed here;
`data/` is gitignored and the re-acquisition command is above.
