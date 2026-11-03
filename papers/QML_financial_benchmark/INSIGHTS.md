# Insights: QML_financial_benchmark (arXiv:2601.03802)

Durable observations from this reproduction that are useful beyond it, grouped by theme. Evidence pointers refer to files in this folder.

## Leakage and feature timing

1. **A 3-D technical-indicator model at 0.80 AUC on next-day direction is a leakage signal, not a result.**
   - **Evidence.** With indicators computed up to t−1, every model on the five Turkish stocks is at chance (AUC 0.47–0.52). With the same-day close included, KCHOL reaches 0.93 (`results/gap_sanity.json`, `e1_kchol_sameday_statistics.csv`).
   - **Mechanism.** RSI and %K use P_t, and the label is P_t > P_{t−1}.
   - **Debugging heuristic.** When an AUC is far above the 0.55–0.65 typical for daily direction, run a same-day variant first. If it matches the paper, the paper's features almost certainly contain the target day.
2. **Cross-market timing is a hidden design choice that decides the result.**
   - **Strict lagging.** With all 64-D inputs strictly lagged, the U.S. stocks are at chance.
   - **Table V timing.** Admitting same-day Asian and European index returns, which close before the U.S. close, lifts AAPL to the paper's level (0.62–0.64).
   - **Lesson.** Record, for every input, the latest timestamp it uses relative to the label, and treat timing as a planning-phase risk (`results/gap_us64_timing.json`).
3. **Selecting on folds that include the test window manufactures apparent quantum wins.**
   - The paper's Fig. 4 protocol shifts test AUC by only 0.015 on average, up to 0.06 on chance-level settings.
   - But under it the QNN significantly beats its parameter-matched ANN in 4 chance-level settings (ULKER, AAPL strict, BA, LNC). Under the leak-free protocol it does so in none (`notebook.ipynb` §9.4).
   - Mechanism: the QNN is selected from a larger grid (24–72 candidates) on folds that include the test window, while the matched ANN is fixed by the QNN's size and gets no selection benefit.
   - Keep a leak-free protocol as primary, and give every compared model the same selection treatment.

## Baseline fairness

4. **"Beats GARCH" can mean "beats a GARCH that ignores known data".**
   - **The paper's GARCH.** The target, 5-day realized variance at t+1, has four of its five squared returns already known at t. The paper's GARCH forecasts it from the one-day σ² alone.
   - **The completed GARCH.** Adding the known part (`garch`, vs `garch_sigma2`) beats every kernel model on all four tickers. It is significantly better than the QSVR on three of them (`results/e6_with_garch_sigma2.csv`).
   - **General check.** Does each baseline use all information available at forecast time?
5. **Under-tuned classical baselines create apparent quantum wins.**
   - The paper's AAPL ANN (0.597) is beaten by our ANN grid (0.641) on the same features. Logistic regression (0.640) matches it.
   - The paper's RBF SVR gets γ ∈ {scale, auto}, while the QSVR tunes its whole feature map.
   - A parameter-matched model plus a linear floor, run in the same runs, settles most such claims.
6. **On these tasks, linear models capture the available signal.**
   - Logistic regression ties or beats every neural model where signal exists (E2: 0.767; AAPL Table V: 0.640).
   - When signal is low-dimensional and linear, no feature map, quantum or classical, has room to help.

## Trading metrics and statistics

7. **Sharpe ratios after threshold calibration need a null band.**
   - Random scores pushed through the paper's τ calibration give per-regime Sharpe ratios of about [−1.5, +0.9] (`results/gap_trading_null.json`).
   - Every paper and local Sharpe sits inside that band, and the seed std is 0.2–1.4.
   - Single-run Sharpe comparisons between models cannot carry a claim. Use ≥ 5 seeds and compare against buy-and-hold, which is best or equal-best in 3 of 4 regimes.
8. **Reported numbers can reveal undisclosed protocol details.**
   - The LSTM's reported annualized volatility equals the full-regime index volatility. So the backtest spans the whole regime, which contradicts the text's within-regime split.
   - The four QLSTM parameter counts fit q² + 18q + 18qL + 1 exactly, which pins down a layout the paper does not describe.
   - Cross-check reported numbers against each other before implementing.

## Quantum kernels and photonic translation

9. **A fixed unitary after the last data encoding is invisible to a fidelity kernel.**
   - |⟨ψ_i|U†U|ψ_j⟩|² = |⟨ψ_i|ψ_j⟩|². The paper's amplitude-encoded QSVR (fixed random layers after amplitude encoding) is the squared cosine similarity of the raw features, raised to β. That is a classical kernel on any platform (`tests/test_kernels.py`).
   - Depth only matters when data is re-uploaded between layers. The photonic kernel here therefore re-uploads phases after fixed random meshes.
10. **Photonic and qubit models land in the same place relative to classical baselines.**
    - **Kernel.** A 10-mode, 2–3-photon re-uploading fidelity kernel tracks the 10-qubit angle kernel within ±0.02 QLIKE on 3/4 tickers.
    - **Classifiers.** Small photonic QNNs (UNBUNCHED threshold readout) match the qubit QNNs on AAPL with 4–8× fewer parameters. On E2 they trail by 0.012 AUC.
    - **Against classical models.** Neither family beats a parameter-matched ANN, logistic regression or the completed GARCH.
    - **Implication.** Choosing an encoding or platform does not rescue a task without exploitable non-linear structure.
11. **Useful MerLin patterns (0.4.2):**
    - **Kernels.** Compute `MeasurementStrategy.amplitudes(FOCK)` once per point and form |A A†|². This is exact, and 5–100× faster than `FidelityKernel` at N = 180–720.
    - **Phase scaling.** Pass phases already scaled (π·x), because `add_angle_encoding(scale)` is ignored.
    - **Fixed meshes.** Seed them (`add_entangling_layer(trainable=False, seed=...)`) so kernels are reproducible.
    - **Single-qubit-style readout.** Build an SQ′ mask (P(last mode occupied)) from `output_keys` of a `probs(UNBUNCHED)` layer.
    - **Validation.** Check `FidelityKernel` against FOCK-amplitude overlaps before using it; see `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`.
12. **Photonic scaling limits.**
    - The FOCK dimension is C(m+n−1, n) and the UNBUNCHED dimension is C(m, n).
    - At m = 10, n = 3 the FOCK dimension is 220 and a 5,000-point kernel is cheap.
    - The dimensions grow combinatorially, so mode and photon counts beyond about 12 modes and 4 photons need care for expanding-window refits.

## Compute and operations

13. **Simulated quantum models cost far more than their classical counterparts.**
    - Per training: QNN 21 s vs ANN 1 s; QLSTM about 2,200 s vs LSTM 15 s; angle QSVR 53 s vs RBF 16 s per refit (`results/phase5_timing.csv`).
    - Budget RED-tier grids (QLSTM) as a seed-0 search plus retraining on the other seeds, and label them `partial`.
14. **Parallelism on a laptop is memory-bound.** More than about 6 single-threaded torch workers on 16 GB caused swapping and stalled every run. Size `--max-parallel` from free RAM, not core count.

## Open directions

- Table V timing for BA, GILD, DVN and LNC (S12b stopped). It would settle C3 under the paper's likely design.
- Per-family QNN ablation (C4). The data is already in `results/sweep_candidates/`.
- Task 3 Optuna seeds 1–2, and E8 (amplitude d = 16/64; training-window robustness).
- An honest photonic test on a task with known non-linear, high-dimensional structure, where a quantum feature map could plausibly matter.
