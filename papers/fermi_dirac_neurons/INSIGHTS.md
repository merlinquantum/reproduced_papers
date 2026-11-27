# Insights — Fermi–Dirac machines as quantizations of neurons

Durable lessons from reproducing arXiv:2605.24386. Read this before
reproducing quantum-neuron, activation-observable or teacher–student QML
papers.

## Baseline fairness

- **The paper's commuting baselines confound two things.** IM and FCIM see
  only p(z) = ⟨z|ρ|z⟩, while the targets depend on X and Y information. Any
  "non-commuting beats commuting" gap mixes quantization with measurement
  access.
  - The fair control is B2: the same activation applied to a linear
    function of the same measured Pauli expectations ⟨H_j⟩, at equal
    parameter count. It isolates "activation of the operator" from
    "activation of its expectation".
- **Tune every model's learning rate, not only the baseline's.** At the
  paper's η = 0.1:
  - the quantum neurons were under-trained, and B2 underfit on Fig. 8;
  - the photonic mesh lost to a classical model.

  All three conclusions flipped under per-model tuning on a held-out
  selection split.
- **Check whether the teacher lies in the student's hypothesis class.**
  - **When it does:** the paper's targets are tanh of a random Hamiltonian
    from the model's own term set. A tuned quantum neuron then fits them to
    round-off (test MSE ≈ 1e-31) and "wins" by 20+ decades. That is a
    well-specified-model effect, not a quantum learning advantage.
  - **When it does not:** with term sets that differ from the teacher's,
    learning barely moves at the paper's η (E16b).
- **Label rules decide the classification winner.** The authors' code
  labels states by sign Tr[H*ρ], linear in B2's features, so tuned logistic
  regression beats the quantum neuron by 3–5 points. Read the released code
  for the exact label and prediction rule before choosing a baseline.
- **"Linear in ρ" does not mean "learnable from the training set".** Full
  tomography features cannot fit a Fig. 9 target from 17 code states that
  do not span the operator space. Exactness arguments for linear baselines
  need informationally complete training data.

## Implementation pitfalls

- **Gradients of matrix functions.** Use Daleckii–Krein divided
  differences, not autograd through `eigh`: autograd divides by eigenvalue
  gaps and fails on the degenerate spectra of Z-only models and θ = 0. Use
  a single gradient operator V(F∘V†RV)V† per step.
- **`np.einsum` with three operands and no path optimization** loops over
  all indices (O(J d⁴)); it made 7-qubit training about 5000× slower. Use
  explicit matmuls.
- **Apple Accelerate's ZHEEVD** (`numpy.linalg.eigh`) fails at d = 4096.
  `scipy.linalg.eigh(driver="evr")` works.
- **Loss definitions:** the loss logged before the final update (the
  convention of the authors' CSVs) and after it differ by one step. Declare
  which one is the metric; a recheck mismatched at 3e-4 until this was
  pinned down.
- **Unbounded models diverge at large learning rates.** Linear models do;
  bounded tanh models do not. A sweep with large η needs per-model
  divergence handling (freeze, mark ineligible), and summarizers must
  tolerate NaN for models marked diverged.

## Trainability (barren plateaus)

- **Haar-random input states cause ≈ 2⁻ⁿ gradient concentration** for
  every Hamiltonian family, commuting or not, at every temperature. This is
  data-induced, from Var_ρ Tr[Oρ] ≈ (Tr O²/d − (Tr O/d)²)/(d+1).
- **Structured inputs** (basis, GHZ, product) with k-local H at T = 2 show
  no exponential decay up to n = 10–11.
- **Training on Haar data stalls by n ≈ 6–8,** because the target variance
  itself shrinks ≈ 2⁻ⁿ. Report the loss relative to the target variance.
- **T ∝ √n does not rescue trainability;** it scales the gradients down.
- **Separating exponential from polynomial decay over n = 4…10 is
  fragile.** Pre-declare the rule (ΔAIC, CI on slopes) and expect
  "undecided" cells.

## Contextuality of the activation observable

- **The Kirby–Love noncontextuality of a Pauli term set is preserved by
  the functional calculus.** φ(H) of a noncontextual set stays
  noncontextual at every temperature, so no temperature switches
  contextuality on.
- **Contextuality of the term set does not by itself help learning or
  gradients** at equal parameter count (E15, E16).
- **It does make the output costlier to compute from few Pauli
  measurements.** The ratio of strings needed, contextual over
  noncontextual, grows exponentially: ×1.3–1.95 per qubit, about 1 at n = 4
  and 25 at n = 10.

## Photonic translation

- **The passive-Hamiltonian case of the paper's continuous-variable neuron
  is exactly linear-optical.**
  - With H = Σ h_ij a_i†a_j and h = U† diag(ε) U,
    Tr[g(H)ρ] = Σ_n p_U(n) g(ε·n).
  - In MerLin: one `add_entangling_layer` mesh, `probs(FOCK)` with
    photon-number-resolving detection, then a classical readout. The
    convention h = U† diag(ε) U was found by testing all four orderings.
- **The commuting limit** (no mesh) is a classical neuron on photon-count
  statistics.
- **Interacting terms** (cross-Kerr) and squeezing are not expressible this
  way.
  - Strong Kerr at half filling restores Haar-rate gradient concentration
    for structured inputs.
  - Linear optics keeps the decay slower than 1/d.
- **Inert parameters.** About m/2 of the m(m−1) phases of
  `add_entangling_layer` are trailing output phases with zero gradient for
  photon counting. Exclude them from per-parameter studies and effective
  parameter counts.
- **Optimizer.** Mesh phases need an adaptive optimizer (Adam) or tuned
  rates. Plain GD at 0.1 left the exactly representable target half
  unfitted.
- **Detector costs.** Threshold detectors and collision-free postselection
  cost about 0.1 decade of MSE at (N, m) = (4, 8). 1000 PNR shots per state
  are as good as exact: each shot gives a bounded sample of g(ε·n).

## Resource planning

- **Calibrate first.** Time the largest cell and measure its peak memory
  before quoting a sweep duration or launching parallel workers.
- **Schedule by memory, not by core count.** Schedule against a memory
  budget with per-task estimates. Dense d × d state storage at 9 qubits
  (≈ 6 GiB per run) crashed a 16 GB machine when run 6-wide.

## Promising directions

- **The misspecified-teacher test** with tuned rates would decide whether
  any quantum-neuron advantage survives outside its own hypothesis class.
- **A hybrid network vs the noncontextual bound (η ≤ 2/3)** on the
  rock-paper-scissors task of `papers/contextuality_inductive_bias`
  (planned E14, not run). The targets must require η > 2/3, otherwise the
  test is uninformative.
- **Classical simulability of the passive photonic neuron through
  few-body correlators at moderate T** (not tested).
