# Insights: QML_financial_benchmark (arXiv:2601.03802)

Durable observations that hold beyond this reproduction. Status: written during implementation (Phase 2), before any paper-scale results.

1. **A fixed unitary after the last data encoding is invisible to a fidelity kernel.**
   - For amplitude encoding with a fixed random unitary stack (Eq. 18), |⟨ψ_i|U†U|ψ_j⟩|² = |⟨x̂_i, x̂_j⟩|². The "quantum" kernel is the squared cosine similarity of the raw feature vectors, raised to the power β. Any number of random layers L changes nothing.
   - The same holds for data-independent layers appended after a single angle encoding. Depth only matters if the data is re-uploaded between layers.
   - Verified numerically in `tests/test_kernels.py`.
   - Consequence: amplitude-QSVR results (Table XIV) are results of a classical, cheaply computable kernel, so a "quantum vs classical kernel" comparison there is a comparison between classical kernels.
2. **Reported parameter counts pin down an underspecified architecture.**
   - The QLSTM layout is not described in enough detail to implement. The four Table XII counts (233, 476, 386, 656) fit exactly q² + 18q + 18qL + 1 for d = 4. That gives one shared Linear(d+q→q) embedding, six VQCs with 2q input parameters and 3q per layer, and a Linear(q→1) head.
   - The LSTM counts match `nn.LSTM` + `Linear(h, 1)` exactly.
3. **Reported risk numbers reveal the evaluation window.**
   - The LSTM's annualized volatility in F1 and F2 (0.3492, 0.1500) equals the full-regime S&P 500 volatility (34.9%, 15.0%). That is only possible if the backtest spans the whole regime almost always in the market.
   - This contradicts the text's 70/10/10/10 within-regime split (which leaves no test slice) and supports Fig. 6's train-on-history design.
4. **Photonic and qubit kernels and classifiers land in the same place relative to classical baselines.**
   - A multi-photon fidelity kernel (10 modes, up to 3 photons, re-uploaded phases) tracks the qubit angle kernel within ±0.02 QLIKE on 3 of 4 assets.
   - Small photonic QNNs match or exceed the qubit QNNs with 4–8× fewer parameters on AAPL.
   - Neither beats a tuned RBF SVR, a GARCH that uses the known part of the realized-variance window, or a parameter-matched ANN.
   - On these financial tasks the quantum feature space, qubit or photonic, is not the bottleneck: the signal is low-dimensional and linear models capture it.

