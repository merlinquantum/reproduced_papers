# VISITED_URLS.md — qrc_volatility (arXiv:2505.13933)

## Remote resources

| Resource | Local cache | Purpose | First access |
|---|---|---|---|
| <https://arxiv.org/pdf/2505.13933> | `$REPRO_SCRATCH_DIR/paper_main.pdf`, text at `paper_main.txt` | Primary paper PDF (v2, 9 Apr 2026) | 2026-08-11 13:30Z |
| <https://arxiv.org/abs/2505.13933v2> | — | Version check (v2 is the latest public version) | 2026-08-11 13:31Z |
| <https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting> | `$REPRO_SCRATCH_DIR/repo_snapshot/` (commit `d2e9b0a`) | Authors' released code and data, linked from the paper's "Code Availability" statement | 2026-08-11 13:33Z |
| <https://merlinquantum.ai/user_guide/> | — | MerLin API reference consulted for `MeasurementStrategy.mode_expectations` and `ComputationSpace` | 2026-08-11 14:12Z |

## Local resources

| Path | Purpose |
|---|---|
| `/reproduced_papers/data/qrc_volatility/Data.CSV` | Authors' normalised monthly feature panel, 816 rows 1950-01..2017-12 |
| `/reproduced_papers/data/qrc_volatility/coeff_10.jld2` | Authors' 100 saved reservoir coupling matrices (JLD2/HDF5, `ms` array) |
| `/reproduced_papers/data/qrc_volatility/authors_qr_predictions.csv` | Copy of the authors' `predict_result.csv`: 245 QR1/QR2 out-of-sample forecasts, used as the regression-test ground truth |
| `$REPRO_SCRATCH_DIR/repo_snapshot/Time_series.jl` | Reference Julia reservoir implementation (`Qreservoir`, `Quantum_Reservoir`, `coeff_matrix`, `compute_qlike`, metric rescaling constants) |
| `$REPRO_SCRATCH_DIR/repo_snapshot/Time_serial_Finance_regression.ipynb` | Julia driver: rolling ridge readout, feature sets, Shapley analysis |
| `$REPRO_SCRATCH_DIR/repo_snapshot/Reservoir_Learning.ipynb` | Python driver: HAR/HARX/AR/ARMAX baselines, MCS and Diebold-Mariano tests |
| `$REPRO_SCRATCH_DIR/repo_snapshot/classical_reservoir.ipynb` | Python `reservoirpy` RC/RCX baseline |
| `$REPRO_SCRATCH_DIR/repo_snapshot/LSTM.ipynb` | Python LSTM/LSTMX baseline |
| `$REPRO_SCRATCH_DIR/probe/` | Exploratory probes: data-reconstruction check, HAR alignment check, MerLin API probe, photonic encoding-scale probe |
| `/home/agent/MERLIN_COOKBOOK.md` | Repository MerLin patterns, read before Phase 4 |
| `papers/qrc_memristor/`, `papers/QORC/`, `papers/QRNN/` | Sibling reservoir-computing reproductions consulted for prior-art overlap (all different papers) |

## Notes on missing upstream files

The authors' notebooks read `Data_raw.csv` and `dff.csv`, which are **not** in
their repository. Both were reconstructed from the published `Data.CSV`: the
target was inverted with the `Min_RV`/`Max_RV` constants hard-coded in
`Time_series.jl`, and the ADF-based differencing rule was replayed (the ADF
statistic is invariant under the affine min-max rescaling, so the differencing
decision is recovered exactly — it reproduces their `diff_DP` / `diff_TB`
column names). The reconstruction is validated by HAR reproducing to four
decimals.
