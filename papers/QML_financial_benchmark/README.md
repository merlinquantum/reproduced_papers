# Quantum vs. Classical Machine Learning: A Benchmark Study for Financial Prediction - Reproduction

Reproduction of arXiv:2601.03802. Status: experiments (Phase 3), photonic adaptation (Phase 4, `PARTIAL_MERLIN_TRANSLATION`) and fair comparison (Phase 5) complete; documentation and notebook phases pending. `LOG.md` is the working log, and `VISITED_URLS.md` lists every resource consulted.

## Reference and Attribution

- Paper: Quantum vs. Classical Machine Learning: A Benchmark Study for Financial Prediction (arXiv preprint, 2026; cs.LG / quant-ph)
- Authors: Rehan Ahmad, Muhammad Kashif, Nouhaila Innan, Muhammad Shafique
- DOI/ArXiv: https://arxiv.org/abs/2601.03802
- Original repository: none released (searched GitHub, Hugging Face, Semantic Scholar and author pages; see `LOG.md`)
- License and attribution notes: cite the arXiv preprint above. All code in this folder is an independent reimplementation. `lib/metrics.py::diebold_mariano` is adapted from `papers/QARIMA`.

## Original Paper

The paper benchmarks simulated gate-based quantum models against classical models of the same family on three daily equity tasks:

1. **Directional classification** (Tables IX–XI). It predicts next-day up/down moves with four QNN families (fully quantum or hybrid, single- or multi-qubit readout) and compares them with shallow ANNs. There are three feature sets:
   - 3-D momentum indicators for 5 Turkish stocks;
   - 7-D cross-market returns for the S&P 500;
   - 64-D lagged returns for 5 U.S. stocks.
2. **Live trading** (Table XII). A QLSTM and a classical LSTM produce long/short/flat signals on the S&P 500 in four market regimes (2008–09, 2018–19, 2020–21, 2022–24). Strategies are scored by annualized return, Sharpe and Sortino.
3. **Volatility forecasting** (Tables XIII–XIV). It forecasts next-day 5-day realized variance with an angle- or amplitude-encoded QSVR, linear/poly/RBF SVR and GARCH(1,1), scored by QLIKE and Diebold–Mariano tests.

## Reproduction Scope (including Updates and Deviations)

Everything is implemented from scratch in PyTorch, with a batched exact-statevector simulator in `lib/statevector.py`. The paper used TorchQuantum with the same noiseless semantics. Decisions that differ from the paper or fill gaps in it are listed in `LOG.md` under "Phase 2 implementation decisions". The main ones:

- **Task 1 selection.** The primary protocol selects on validation folds 1–4 and tests on fold 5 (2020–2021). The paper's protocol averages over folds 1–5, which include the test window, and is reported alongside, labelled test-leaking.
- **Task 1 features.** The 3-D indicators use information up to day t−1. A same-day variant (`configs/e1_same_day_diagnostic.json`) checks for look-ahead.
- **Fair baselines added:**
  - a parameter-matched ANN and LSTM;
  - majority-class, logistic-regression and previous-day floors;
  - buy-and-hold and always-flat strategies;
  - HAR-RV, persistence, and an RBF SVR with a widely tuned γ.
- **QLSTM layout.** Reconstructed from the paper's parameter counts. It reproduces all four Table XII counts exactly.
- **Task 2 split.** The primary split follows Fig. 6: train on the history before each regime, backtest on the whole regime. The text's within-regime 70/10/10/10 split is available with `--split-mode within_regime`.

## Install and How to Run

```bash
# From the repository root
python -m venv .venv
.venv/bin/python -m pip install -r papers/QML_financial_benchmark/requirements.txt
```

Download the data (once):

```bash
# From the repository root. The python.org Python needs the certifi CA bundle.
SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") \
  .venv/bin/python papers/QML_financial_benchmark/utils/download_yahoo_data.py
```

Quick smoke runs (seconds each, not paper results):

```bash
# From the repository root
python implementation.py --paper QML_financial_benchmark --config configs/defaults.json
python implementation.py --paper QML_financial_benchmark --config configs/e5_reduced.json
python implementation.py --paper QML_financial_benchmark --config configs/e6_reduced.json
```

Paper-scale experiments (one run per ticker, regime or seed; see `configs/`):

```bash
python implementation.py --paper QML_financial_benchmark --config configs/e1_original.json --ticker KCHOL.IS --seed 0
python implementation.py --paper QML_financial_benchmark --config configs/e3_original.json --ticker AAPL --seed 0
python implementation.py --paper QML_financial_benchmark --config configs/e5_original.json --regime F2 --seed 0
python implementation.py --paper QML_financial_benchmark --config configs/e6_original.json --vol-ticker KCHOL.IS

# Equivalent from inside papers/QML_financial_benchmark
python ../../implementation.py --config configs/e1_original.json --ticker KCHOL.IS
```

## Configuration

`cli.json` is the authoritative list of paper flags: `--ticker`, `--feature-timing`, `--regime`, `--split-mode`, `--vol-ticker` and `--num-threads`. The shared runtime adds `--config`, `--seed`, `--dtype`, `--device`, `--outdir`, `--data-root` and `--log-level`.

Each config sets `task` and a section named after it (`directional`, `trading` or `volatility`):

| Config | Paper | Content |
|---|---|---|
| `defaults.json` | — | Lightweight Task 1 smoke run on KCHOL.IS |
| `e1_original.json` | Table IX | 3-D features, full QNN and ANN grids |
| `e1_same_day_diagnostic.json` | — | E1 with same-day indicators (leakage check) |
| `e2_original.json` | Table X | 7-D S&P 500 features, 7-qubit QNNs |
| `e3_original.json` | Table XI | 64-D features, amplitude QNNs on 2–6 qubits |
| `e3_hybrid_angle.json` | — | 64-D with the hybrid 64→n angle reading |
| `e5_original.json` | Table XII | QLSTM vs LSTM trading, one regime per run |
| `e5_reduced.json` | — | Trading smoke run |
| `e6_original.json` | Table XIII | Volatility, 5+5 lags, angle QSVR (10 qubits) |
| `e7_original.json` | Table XIV | Volatility, 16+16 lags, amplitude QSVR (5 qubits) |
| `e6_reduced.json` | — | Volatility smoke run |

## Data

- The source is Yahoo Finance daily prices, as in the paper. `utils/download_yahoo_data.py` downloads the 18 symbols of Tables III–V for 2000-01-01 → 2024-12-31 into `data/QML_financial_benchmark/yahoo_daily_prices/`, with `csv/<SYMBOL>.csv`, `raw/<SYMBOL>.json` and `manifest.json` (SHA-256 checksums).
- Override the data root with `DATA_DIR=/abs/path` or `--data-root /abs/path`.
- Limitations:
  - Yahoo revises adjusted prices. The snapshot (fetched 2026-10-21) is the paper's source, but not the paper's exact download. As a check, the ^GSPC series reproduces Table VI volatility, max drawdown and skew exactly.
  - Turkish zero-volume rows that repeat the previous close are Yahoo placeholders on exchange holidays, and are dropped.
  - Turkish adjusted closes are non-positive before mid-2004. Windows that reach them fail with an error.
- `data/QML_financial_benchmark/yahoo_daily_prices/README.md` has the details.

## Outputs

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` inside this folder:

```
run.log                 # lifecycle, per-epoch and evaluation log
run_status.json         # structured run state (experiment-logging contract)
config_snapshot.json    # resolved configuration
metrics.json            # test metrics and model selections
candidates.csv          # Tasks 1-2: every candidate's validation results
test_predictions.csv    # test-period probabilities or forecasts per model
hyperparameters.json    # Task 3: tuned SVR/QSVR settings per refit block
```

## Results Obtained and Comparison with the Paper

- All numbers are test-set values on this repository's Yahoo snapshot (fetched 2026-10-21).
- Values are mean ± std over seeds unless labelled otherwise.
- Every number traces to a sweep under `outdir/sweeps/` and a CSV in `results/`. The full ledger, statistics and gap diagnosis are in `LOG.md`.
- Scope labels follow the repository policy: `paper-accurate` (full protocol, ≥3 seeds), `partial` (reduced protocol), `preliminary` (single seed or single tuner seed), `diagnostic` (not the paper's protocol).

### Task 1: directional classification (Tables IX–XI)

Fold-5 test 2020–2021. Models are selected on validation folds 1–4 (leak-free); the paper's protocol includes the test window and changes AUC by ≤ 0.016.

| Experiment | QNN AUC | ANN AUC | ANN with QNN's parameter count | Paper QNN / ANN | Label |
|---|---|---|---|---|---|
| KCHOL 3-D, lagged indicators (E1) | 0.516 ± 0.009 | 0.511 ± 0.018 | 0.510 | 0.797 / 0.748 | paper-accurate, 5 seeds |
| Other 4 Turkish stocks, lagged (E1) | 0.47–0.51 | 0.49–0.51 | 0.50–0.52 | 0.71–0.90 | paper-accurate, 5 seeds each |
| KCHOL 3-D, same-day indicators | 0.926 ± 0.011 | 0.945 ± 0.003 | 0.927 | 0.797 / 0.748 | diagnostic, 5 seeds |
| S&P 500 7-D (E2) | 0.757 ± 0.008 | 0.757 ± 0.012 | 0.760 | 0.750 / 0.766 | paper-accurate, 5 seeds |
| AAPL 64-D, Table V timing (E3) | 0.622 ± 0.008 | 0.641 ± 0.001 | 0.641 | 0.635 / 0.597 | paper-accurate, 5 seeds |
| AAPL 64-D, strictly lagged (E3) | 0.525 ± 0.026 | 0.495 ± 0.021 | 0.497 | 0.635 / 0.597 | paper-accurate (strict timing), 5 seeds |
| BA, GILD, DVN, LNC 64-D, strictly lagged (E3) | 0.49–0.52 | 0.45–0.51 | 0.48–0.50 | 0.59–0.74 | paper-accurate (strict timing), 3 seeds each |

![Task 1 test AUC](results/e1_e3_test_auc.png)

**Headline claims:**
- **AAPL (C1): unsupported.** With the paper's likely feature timing, absolute AUCs match the paper. But the ANN, the parameter-matched ANN and a logistic regression all beat the QNN: QNN − ANN = −0.019, p = 0.006.
- **KCHOL (C2): unsupported.** With leak-free features every model is at chance. The paper's 0.71–0.90 Turkish AUCs are only reachable when the indicators include the price on the day being predicted (likely leakage in the original pipeline).
- **S&P 500 (E2):** quantitatively reproduced, with QNN and ANN tied.

### Task 2: live trading (Table XII)

- **Setup:** whole-regime backtest, 0.05% fee. QLSTM and LSTM are selected on seed 0 (full 30-candidate grid) and retrained with seeds 1–4.
- **Label:** `partial` (RED reduction rule), 5 seeds.

| Regime | QLSTM Sharpe | LSTM Sharpe | Buy & hold | Paper QLSTM / LSTM |
|---|---|---|---|---|
| F1 2008–09 | −0.19 ± 0.27 | +0.27 ± 0.30 | −0.33 | 0.39 / −0.05 |
| F2 2018–19 | +0.18 ± 0.48 | +0.12 ± 0.48 | +0.66 | 1.26 / 0.50 |
| F3 2020–21 | +0.74 ± 0.78 | +0.65 ± 0.37 | +0.82 | 0.77 / 0.90 |
| F4 2022–24 | +0.16 ± 0.22 | +0.23 ± 0.21 | +0.40 | −0.69 / 1.52 |

![Task 2 Sharpe](results/e5_sharpe_auc_rule.png)

- **Claim C5 (QLSTM better in F1 and F2): unsupported.** No regime shows a significant QLSTM − LSTM Sharpe difference.
- Test AUCs are 0.49–0.55, as in the paper.
- Random scores pushed through the paper's threshold calibration already give Sharpe ratios in about [−1.5, +0.9], so single-run Sharpe comparisons cannot separate the models.

### Task 3: volatility forecasting (Tables XIII–XIV)

- **Setup:** QLIKE (lower is better). Expanding window 2007–2024, about 4,300 test days. 5+5 lags; angle QSVR on 10 qubits.
- **Label:** `preliminary` (single Optuna tuning seed).

| Model | ^GSPC | AAPL | GARAN | KCHOL |
|---|---|---|---|---|
| GARCH, completed with the 4 known squared returns | **−8.590** | **−7.400** | **−6.713** | **−7.030** |
| SVR-RBF | −8.552 | −7.382 | −6.706 | −7.026 |
| QSVR, angle encoding | −8.547 | −7.380 | −6.704 | −7.008 |
| GARCH σ²-only (likely paper version) | −8.451 | −7.226 | −6.557 | −6.867 |
| SVR-linear | −8.544 | −7.296 | −4.280 | −6.966 |

![Task 3 QLIKE](results/e6_qlike_vs_qsvr.png)

- **Claim C7 (QSVR lowest QLIKE on KCHOL): unsupported.**
- The QSVR is within 0.001–0.005 of RBF on ^GSPC and AAPL, consistent with the paper.
- The QSVR beats linear and polynomial SVR in direction on all four tickers (C8), with significance in some cases.
- The paper's "QSVR beats GARCH" holds only against a GARCH that forecasts the 5-day realized variance from its one-day variance alone. With the known part of the window added, GARCH beats every kernel model.
- The amplitude-encoded QSVR's kernel is mathematically the squared cosine similarity of the feature vectors, raised to the power β (`INSIGHTS.md`).

### Photonic adaptation with MerLin (Phase 4, `PARTIAL_MERLIN_TRANSLATION`)

These are adaptations, not the paper's method. They run noiseless, analytic MerLin simulations (package `merlinquantum` 0.4.2, which reports `merlin.__version__` 0.4.1) (`lib/photonic_merlin.py`; configs `e6_merlin.json`, `e2_merlin.json`, `e3_table_v_merlin.json`).

| Task | Photonic model | Result | Qubit model (Phase 3) | Fair classical |
|---|---|---|---|---|
| Volatility (E6), 4 tickers | SVR on a multi-photon fidelity kernel (10 modes, 2–3 photons, angle phases re-uploaded after fixed random MZI meshes; FOCK amplitudes) | QLIKE ^GSPC −8.544, AAPL −7.365, GARAN −6.668, KCHOL −7.003 | −8.547, −7.380, −6.704, −7.008 | SVR-RBF −8.552, −7.382, −6.706, −7.026 |
| Directional E2, S&P 500 7-D | Photonic QNN (7 modes, 2–3 photons, trainable MZI meshes, UNBUNCHED threshold readout), 5 seeds | AUC 0.745 ± 0.005 | 0.757 ± 0.008 | ANN with QNN's parameters 0.769; logistic 0.767 |
| Directional E3, AAPL 64-D (Table V timing) | Hybrid photonic QNN (6–8 modes, 2 photons), 5 seeds | AUC 0.635 ± 0.004 (450–690 parameters) | 0.622 ± 0.008 (1,088–4,275 parameters) | ANN 0.641; logistic 0.640 |

- The photonic models reproduce the qualitative standing of the qubit models: within ±0.02 of them on 3/4 volatility tickers and on both classification tasks. They never beat the fair classical baselines.
- Trading (QLSTM) and the amplitude-encoded kernel were not translated: the trading protocol cannot separate models, and the amplitude kernel is classical on any platform.
- The hardware-aware details (photons, modes, input states, detectors, wall-clock, seeds) are in `LOG.md`, "Phase 4 hardware-aware result table".

### Fair comparison (Phase 5)

**The paper's baselines are not fair for its claims** (`MISSING_OR_UNFAIR` for all three tasks):
- the ANNs are not parameter-matched, and the paper's AAPL ANN is under-tuned;
- the LSTM pairs differ up to 4× in parameters;
- the RBF SVR gets a restricted γ while the QSVR tunes its feature map;
- the GARCH ignores the 4/5 of the 5-day RV already known at t.

Every claim is therefore judged against the fair controls declared in the planning phase, all run in the same runs as the quantum models:
- the parameter-matched ANN (±10% parameters) or LSTM (±25%);
- logistic regression, buy-and-hold and flat;
- an RBF SVR with γ searched widely under the same Optuna budget;
- the completed GARCH, HAR-RV and persistence.

![Quantum minus fair control](results/phase5_forest.png)

| Claim | Outcome against fair controls |
|---|---|
| C1 QNN beats ANN on AAPL | Unsupported. With the paper's timing, QNN − matched ANN = −0.019 AUC (p = 0.005). |
| C2 QNN beats ANN on KCHOL | Unsupported. A tie at chance level with leak-free features. |
| C3 ANN wins on most other tickers | Partially supported. The QNN never beats the matched ANN in any of 12 settings; classical models are significantly ahead in 2. |
| C5/C6 QLSTM better Sharpe or AUC | Unsupported. Ties in all four regimes; buy-and-hold is best or equal-best in F2–F4. |
| C7 QSVR lowest QLIKE on KCHOL | Unsupported. RBF and the completed GARCH are lower; the GARCH is significantly better on 3/4 tickers. |
| C8 QSVR beats linear and poly SVR | Partially supported. Direction 4/4; significant in 1/4 (linear) and 3/4 (poly). |
| C10 QSVR ≥ linear SVR | Supported in our runs (single tuning seed). |

- Simulated cost per training is 20× (QNN) to 150× (QLSTM) that of the classical counterparts (`results/phase5_timing.csv`).
- No quantum or photonic advantage is supported by any fair, commensurable comparison.
- The full table is `results/phase5_comparison.csv`; differences with CIs are in `results/phase5_differences.csv`.

## Limitations

- Phase 3 is single-machine and CPU-only.
- E5 uses the RED reduction rule: the grid search runs on seed 0 only.
- E6/E7 use one Optuna tuning seed.
- E8 (amplitude d = 16/64, training-window robustness) was not run.
- E3 with the Table V timing was run for AAPL only. The other four U.S. stocks' sweep was stopped because the laptop was overloaded.
- Undisclosed paper details that the reproduction had to assume:
  - Task 1 training settings (Adam, lr 1e-3, patience 30);
  - the Task 3 lag counts and test window;
  - the exact form of the paper's same-day leakage.
- The Yahoo snapshot is newer than the paper's undated download.
- Everything is noiseless statevector or MerLin analytic simulation; no QPU and no shot noise. Photonic results are adaptations with smaller grids than the qubit models.

## Tests

```bash
cd papers/QML_financial_benchmark
pytest -q
```

The tests cover:
- simulator correctness;
- Table XII and ANN parameter counts;
- the kernel identities;
- the metrics;
- data timing and splits (using the local snapshot);
- the CLI flags;
- an end-to-end default run.

## Citation and License

Cite arXiv:2601.03802. The code is under the repository's MIT license.
