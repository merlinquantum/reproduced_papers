# Action Required: Send to MerLin Team

## Summary

Four MerLin items came out of this reproduction, ordered by priority:

1. **`FidelityKernel` does not return the state fidelity** for re-uploading feature maps. It returns a constant kernel of all ones when the angle encoding is the last circuit element, and its values change when a fixed final mesh is appended. A true fidelity kernel cannot depend on a fixed final unitary.
2. **`CircuitBuilder.add_angle_encoding(scale=...)` is silently ignored.**
3. **`FidelityKernel` is 5–100× slower** than computing FOCK amplitudes once and forming the Gram matrix, and the gap grows with the number of points.
4. **`merlin.__version__` reports 0.4.1** while the installed `merlinquantum` distribution is 0.4.2.

Each item has a standalone reproducer: `papers/QML_financial_benchmark/utils/merlin_feedback_reproducers.py`. Its output is saved in `results/merlin_feedback_reproducers.txt`.

## Context

The reproduction of arXiv:2601.03802 (quantum vs classical ML for financial prediction) adapted two of the paper's qubit models to MerLin:
- a multi-photon fidelity kernel inside an SVR, for volatility forecasting: 10 modes, 2–3 photons, phases π·x re-uploaded after fixed random meshes;
- photonic QNN classifiers with threshold (UNBUNCHED) readout.

Fidelity kernels are the natural photonic translation of quantum-kernel papers, and many reproductions in this repository use them (for example `papers/photonic_quantum_enhanced_kernels`, `papers/qSVM_qKNN`).

Because of items 1–3, this reproduction does not use `FidelityKernel` or the `scale` argument. It computes `MeasurementStrategy.amplitudes(FOCK)` once per point and forms |⟨ψ(x)|ψ(x′)⟩|² itself (`lib/photonic_merlin.py`).

**Environment:** `merlinquantum` 0.4.2 (`merlin.__version__` reports 0.4.1), torch 2.12.1, Python 3.12.4, macOS on Apple M4, CPU, float64.

## Items to Send

### High — FidelityKernel: values differ from the state fidelity (all ones when the encoding is last)

- **Type:** bug (probable defect; root cause not identified).
- **What happened:**
  - **Setup.** The feature map is fixed mesh `U0` → `add_angle_encoding` on all modes, optionally followed by a fixed mesh `U1`. The input state is 2 photons in 4 modes.
  - **Comparison.** We compared `FidelityKernel(FeatureMap(builder=..., input_parameters="x0"))` with the exact fidelity |⟨ψ(x)|ψ(x′)⟩|². The exact fidelity uses FOCK amplitudes from a `QuantumLayer` built from the same builder.
  - **Without `U1`:** every kernel entry is exactly 1.0, in all three computation-space settings (`None`, `FOCK`, `UNBUNCHED`). The exact fidelity's off-diagonal mean is 0.975.
  - **With `U1`:**
    - the exact fidelity is unchanged, as it must be, because a fixed final unitary cancels in ⟨ψ|ψ′⟩;
    - `FidelityKernel` changes to an off-diagonal mean of 0.952 (FOCK) or 0.960 (UNBUNCHED);
    - it still differs from the exact value by up to 0.085.
- **Evidence:**
  - `utils/merlin_feedback_reproducers.py`, check [2]; output in `results/merlin_feedback_reproducers.txt`.
  - The first observation was the all-ones kernel on 6 random points, during Phase 4 (`LOG.md`, "MerLin API findings").
  - Reproduced on 2026-10-23: the all-ones kernel in 2 separate runs with different random points, and the final-mesh dependence in 3.
- **Uncertainty:**
  - We have not traced the cause in `FidelityKernel`'s source. Candidates include how the adjoint circuit is built, or how the input parameters are routed to the second copy.
  - It is possible that `FeatureMap` applies its own input transform. That alone would not explain the all-ones output or the dependence on a fixed final unitary.
- **Why it matters:**
  - A kernel of all ones makes an SVR or SVM predict a constant, with no error raised.
  - Encoding-last feature maps are a common pattern, and the result looks like a legitimate (bad) quantum kernel.
  - A kernel that depends on a data-independent final unitary is not a fidelity kernel, so results built on it may be misattributed.
- **Suggested action:**
  - Add a unit test asserting `FidelityKernel(x, x′) == |⟨ψ(x)|ψ(x′)⟩|²` against FOCK amplitudes from the same builder.
  - Add a test that appending a fixed unitary leaves the kernel unchanged.
  - Then fix the construction.

### High — CircuitBuilder: `add_angle_encoding(scale=...)` has no effect

- **Type:** bug.
- **What happened:**
  - `scale` is recorded in the builder's `angle_encoding_specs`, but `QuantumLayer` applies the raw input as the phase.
  - Output probabilities with `scale=π` are identical to those with `scale=1`: max difference 0.0.
  - They differ from `scale=1` with π·x passed in: max difference 0.096.
- **Evidence:**
  - `utils/merlin_feedback_reproducers.py`, check [1].
  - Earlier known-answer test (1 photon, 4 modes): the measured overlap 0.176422 equals the θ = x prediction, not θ = πx (0.806637). See `LOG.md`, "MerLin API findings".
  - Diagnosed by reading the `CircuitBuilder.add_angle_encoding` source.
- **Why it matters:**
  - Users who set `scale` to map [0, 1] features onto a half or full turn silently get a much narrower phase range.
  - That changes kernel bandwidth and classifier expressivity without any warning.
  - Published comparisons built on it would be mis-specified.
- **Suggested action:** either apply `scale` in the forward pass, or remove the argument (or raise when it is not 1.0). Document the phase convention (θ = scale · x) in the docstring and the cookbook.

### Medium — FidelityKernel: cost grows much faster than "states once, then Gram matrix"

- **Type:** improvement.
- **What happened:**
  - **Setup:** 10 modes, 3 photons, FOCK space; the feature map is mesh → encoding → mesh; one Gram matrix on N random points.
  - **Timings** (two runs): `FidelityKernel` took 0.2–0.5 s at N = 180, 0.7–2.5 s at N = 360 and 10–13 s at N = 720.
  - **Alternative:** computing the N state vectors (FOCK dimension 220) once and forming |A A†|² took 0.007–0.23 s.
  - **Ratio:** 5–100×, growing with N. This is consistent with one circuit simulation per pair instead of one per point.
- **Evidence:** `utils/merlin_feedback_reproducers.py`, check [3]. The machine was an Apple M4 CPU running other work, so absolute times vary between runs; the trend with N does not.
- **Why it matters:**
  - In this reproduction's expanding-window protocol, the training set grows to about 5,000 days and the kernel is rebuilt for every Optuna trial (50 per refit, 36 refits per ticker).
  - At that scale `FidelityKernel` would dominate the cost.
  - The state-based route kept the photonic kernel at about 11 s per refit (`results/phase5_timing.csv`).
- **Suggested action:** for noiseless simulation (`shots=None`), have `FidelityKernel` compute each point's state once and return |⟨ψ_i|ψ_j⟩|² by matrix product. Keep the pairwise circuit only for sampling, or where the measured quantity requires it. Document the cost model.

### Low — Packaging: `merlin.__version__` disagrees with the installed distribution

- **Type:** bug (packaging).
- **What happened:** `importlib.metadata.version("merlinquantum")` returns 0.4.2, but `merlin.__version__` returns 0.4.1.
- **Evidence:** `results/merlin_feedback_reproducers.txt`, first line. Also recorded in the README's hardware-aware table and in `LOG.md`; a Phase 4 audit flagged the mismatch.
- **Why it matters:** reproductions report the MerLin version in their hardware-aware result tables. A wrong `__version__` makes results look as if they came from a different release, and breaks version-gated workarounds (such as the `scale` issue above).
- **Suggested action:** derive `__version__` from package metadata (`importlib.metadata.version`) at import time, or bump it in the release process.
