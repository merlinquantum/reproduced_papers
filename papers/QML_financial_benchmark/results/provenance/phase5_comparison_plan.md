# Phase 5 Comparison Plan (extract of LOG.md)

Verbatim extract of the `## Phase 5 Comparison Plan` section and its run-count amendment from the paper's `LOG.md`, which the repository's `.gitignore` excludes from git. This file is committed **before** the Phase 5 result files (`results/phase5_*.csv`, `results/phase5_forest.png`), so git history independently timestamps the plan before the evidence it governs (audit item P5-08).

The heading times `08:10Z` / `08:15Z` below are wrong estimates. The plan text was written at 2026-10-23T05:23:59Z; see the timestamp-correction amendment in `LOG.md`.

---

## Phase 5 Comparison Plan — 2026-10-23T08:10Z

- **Written before** any Phase 5 table, figure or run.
- **Code state:** base commit e7beb6a + uncommitted paper folder; lib+configs fingerprint `f365626adbbbd947`.
- **Evidence:** Phase 3 sweeps S1–S12a (S12b superseded) and Phase 4 sweeps S13–S15, all verified by `utils/verify_sweeps.py`.

### Claim table (step 2)

| Claim | Compared systems in paper | Advantage axis | Paper evidence | Required control | Matching criterion | Possible confounders | Current evidence |
|---|---|---|---|---|---|---|---|
| C1: QNN beats the ANN on AAPL (+3.8 AUC, +3.4 pp accuracy) | Hybrid-SQ QNN vs [64–32–1] ANN | Accuracy and AUC at matched parameters ("parameter-matched ANN") | Table XI, one run | ANN tuned with the same budget; ANN at the QNN's parameter count; linear floor | Trainable parameters (±10% for the matched ANN); same features, folds, optimizer, early stopping, selection rule | Feature timing (G2); selection on the test window (R3); single run; under-tuned ANN | S2 (strict timing), S12a (Table V timing), 5 seeds each; S15 photonic |
| C2: QNN beats the ANN on KCHOL (+4.9 AUC) | Hybrid-MQ QNN vs [3–11–1] ANN | as C1 | Table IX | as C1 | as C1 | Same-day leakage (G1) | S1 (lagged), S3 (same-day diagnostic) |
| C3: classical ANN wins on most other tickers and on the 7-D S&P 500 | QNN vs ANN per ticker | Accuracy and AUC | Tables IX–XI | as C1 | as C1 | No signal on lagged features | S6 (E2), S7 (Turkish), S8a/b (U.S. strict); S14 photonic on E2 |
| C5: QLSTM higher Sharpe than LSTM in F1 and F2 | QLSTM vs LSTM grid winners | Risk-adjusted return (Sharpe, Sortino) | Table XII, one run per regime | LSTM at the QLSTM's parameter count; buy-and-hold; flat; logistic threshold strategy | Parameters (±25%); same split, loss, early stopping, τ grid, fee | Threshold-calibration noise (G3); seed-0 selection | S4, S9, S11 (5 seeds; `partial`) |
| C6: QLSTM AUC slightly higher in all regimes | as C5 | AUC (ranking skill) | Table XII | as C5 | as C5 | as C5 | as C5 |
| C7: angle QSVR lowest QLIKE on KCHOL; within 0.02–0.04 of RBF on S&P 500 and AAPL | QSVR vs SVR lin/poly/RBF, GARCH | Forecast loss (QLIKE) | Table XIII, one run | RBF with a wide γ search under the same budget; GARCH using the known part of the RV window; HAR-RV; persistence | Same features, refit schedule, Optuna trials per model, metric; DM test | Weak GARCH (A1); restricted RBF γ; tuning seed; test window | S5, S10 (single Optuna seed); S13 photonic |
| C8: angle QSVR beats linear and poly SVR | QSVR vs SVR-lin/poly | QLIKE | Table XIII | as C7 | as C7 | as C7 | S5 |
| C9: amplitude QSVR (d=32) competitive | Amplitude QSVR vs classical set | QLIKE | Table XIV | as C7 (on the 32-D features) | as C7 | Amplitude kernel = classical cosine kernel (Insight 1) | S5 (E7) |
| C10: QSVRs match or exceed linear SVR | QSVR vs SVR-lin | QLIKE | Conclusion | as C7 | as C7 | as C7 | S5 |

- **Out of scope:**
  - C4 is QNN-internal (no classical comparison).
  - C11 (fairness) is answered by this phase as a whole.

### Paper baseline evaluation (step 3)

| Task | Paper baseline | Fair for the claimed axis? | State |
|---|---|---|---|
| T1 (C1–C3) | ANN chosen from a width grid by validation AUC; reported as "parameter-matched" | **No.** QNN parameter counts are never reported, and the selected networks are not matched: in S12a QNN 1,088–4,275 vs ANN 727–6,273. The paper's AAPL ANN (0.597) is 0.044 below every ANN trained on the same features (S12a), so it was under-tuned. | `MISSING_OR_UNFAIR` |
| T2 (C5–C6) | LSTM chosen from the h × layers grid | **No** as a capacity match: e.g. Table XII F3 compares 1,426 vs 386 parameters. Buy-and-hold is reported only as a regime statistic, not as a strategy. | `MISSING_OR_UNFAIR` |
| T3 (C7–C10) | SVR lin/poly/RBF with γ in (scale, auto); GARCH(1,1) | **No.** RBF γ is restricted while the QSVR tunes feature-map hyperparameters (unequal tuning freedom). GARCH ignores the 4/5 of the target already known at t (A1, S10). No HAR-RV or persistence floor. | `MISSING_OR_UNFAIR` |

### Phase 1 baseline plan, validated (step 4)

- The Phase 1 Fair Baseline Plan (section "Fair Baseline Plan") declared exactly the additional controls that are needed:
  - T1: parameter-matched ANN (±10%), equal-budget ANN grid, majority, logistic and previous-day floors.
  - T2: parameter-matched LSTM (±25%), buy-and-hold, flat, logistic threshold strategy.
  - T3: RBF with wide γ, HAR-RV, persistence, plus GARCH.
- **Each control is valid against the step-4 criteria:**
  - same task, target, data access, metric and selection or stopping rule;
  - justified capacity match (trainable parameters) or equal tuning budget (Optuna trials);
  - simple and strong (logistic regression and HAR are standard floors);
  - no artificial handicap.
- **Additions made after Phase 1, recorded as amendments before their runs** (none based on a preferred result):
  - D20 completed GARCH, which is the standard way to forecast a partly-observed window. The σ²-only GARCH is kept as the paper-comparable variant (S10).
  - The Table V timing rerun (S12a), following gap diagnosis G2.
- **Implementation status:** these controls were implemented in Phase 2 inside the existing runner and task modules, so they share the data, splits, metrics and logging. They were executed in Phase 3 in the same runs as the quantum models, on the same machine.
- No new baseline code is needed in Phase 5. Step 7 adds focused tests for the controls that lack them.

### Frozen protocol (step 5)

- **Datasets and splits:** as in Phase 3.
  - T1: walk-forward fold 5, test 2020-01-02 → 2021-12-31; selection on validation folds 1–4 (`leak_free`, primary).
  - T2: history_then_regime 70/10/10/10, whole-regime test.
  - T3: expanding window, 720-day initial window, refit every 120 days, test 2007-11 → 2024-12.
- **Preprocessing:** min-max on training rows; target / training mean RV (T3 SVR models).
- **Training-set size:** identical across compared systems within each run (same rows).
- **Optimizer and stopping:** Adam lr 1e-3, batch 32, max 300 epochs, patience 30 on early-stop AUC (T1, T2). Optuna TPE with 50 trials per refit per model (T3).
- **Checkpoint:** best early-stop-AUC epoch (T1, T2); Optuna best trial refit on the full window (T3).
- **Metric and direction:**
  - T1: test AUC, maximize (accuracy secondary).
  - T2: test Sharpe, maximize (AUC secondary).
  - T3: QLIKE, minimize (MSE secondary).
- **Repetitions:**
  - T1: 5 seeds (S1, S2, S3, S6, S7, S12a, S14, S15); 3 seeds (S8a/b).
  - T2: 5 seeds per regime.
  - T3: 1 Optuna seed (deterministic models).
- **Aggregation:** mean over seeds. **Variability:** sample std. **Tests:**
  - seed-level one-sample t-test of paired differences (T1, T2);
  - seed-ensemble paired block bootstrap (T1);
  - Diebold–Mariano on QLIKE (T3).
  - α = 0.05 uncorrected, with the number of tests stated.
- **Tie tolerance:**
  - A difference whose seed-level 95% CI contains 0 (T1, T2) or whose DM p ≥ 0.05 (T3) is a **tie**.
  - Model-selection ties within 1e-4 go to fewer parameters (unchanged).
- **Matching calculation:**
  - T1 matched ANN width = round((P_QNN − 1)/(d + 2)), one hidden layer (`matched_ann_width`).
  - T2 matched LSTM = the grid LSTM with the smallest |P_LSTM − P_QLSTM|, within_tolerance flagged at ±25%.
  - T3: equal Optuna budget, same features.
- **Comparisons decided per claim:** quantum model vs **strongest fair control** (the best-mean of the paper baseline and the additional controls) and vs the **capacity-matched control**.
- **Run counts (expected = observed):** T1 75 runs, T2 16 runs, T3 16 runs (S5 8, S10 4, S13 4); photonic 14 (S13–S15).
- **Compute limit:** no new runs planned. Wall-clock is taken from run logs (per-epoch timing by model family) and sweep timestamps.
- **Planned extensions:** **none.** Phase 4 is an adaptation, not an extension. No condition outside the paper's claims is planned, given the laptop budget. Step 13 will be skipped.

### Amendment — 2026-10-23T08:15Z (run-count correction; before any Phase 5 evidence)

- The plan's expected run counts were miscounted. Correct counts, all observed COMPLETE in `results/sweep_verification.json`:
  - **T1 qubit:** 57 = S1 5 + S3 5 + S2 5 + S6 5 + S7 20 + S8a 4 + S8b 8 + S12a 5.
  - **T1 photonic:** 10 = S14 5 + S15 5.
  - **T2:** 20 = S4 4 + S9 12 + S11 4.
  - **T3:** 16 = S5 8 + S10 4 + S13 4, where S13's 4 are the photonic kernel.
- No protocol rule changes.
