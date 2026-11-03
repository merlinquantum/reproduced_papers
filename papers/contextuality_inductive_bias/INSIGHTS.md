# Insights — Contextuality and Inductive Bias in QML (arXiv:2302.01365)

Durable notes for future reproductions. The evidence and the full numbers
are in `README.md` and `LOG.md`.

## Scientific results worth remembering

- **Inductive bias ≠ quantum advantage.**
  - A linear constraint on label expectations (here zero-sum,
    sum_k E[Y_k] = 0) helps generalisation in every model family tried:
    qubit, photonic and classical.
  - It is easy to hard-code classically. A softmax over the extremal points
    of the constrained behaviour polytope (here the six hexagon vertices)
    on the same Fourier features was the best model overall.
  - For any paper that claims a constraint "cannot be easily encoded"
    classically, build that baseline first.
- **Contextuality did not explain any gap.**
  - Trained bias-encoding models have contextuality witnesses
    eta = 0.18-0.64, all below the 2/3 needed for contextuality (Theorem
    4.1).
  - 99.9% of test behaviours lie inside V_{2/3}, where noncontextual models
    are not constrained.
  - A provably noncontextual photonic readout (group occupation) matched
    the contextual-capable one.
  - Before attributing performance to contextuality, check whether the
    task's data distribution reaches the region that noncontextual models
    cannot express.
- **The theory reproduces exactly with small LPs.** Lemma 4.1 (bound 5/2;
  3 without the preparation equivalence) and Theorem 4.1 (eta = 2/3, tight)
  are linear programs over noncontextual decompositions (scipy HiGHS).

## Evaluation and metric pitfalls

- **Check train/test preprocessing symmetry in the author code.** The
  notebook scales training inputs by pi/2 but evaluates test KL on unscaled
  inputs.
  - The published curves need this inconsistency.
  - Under consistent scaling, the headline margin shrinks about tenfold
    (2.9e-3 to 3e-4).
  - Running several pre-declared evaluation protocols (`paper`, `code`,
    `fork`) turned a possible disagreement into an explanation.
- **Published "baseline" curves can come from a weaker baseline than the
  paper describes.** The generic-model curve matches only the author code's
  repeated CNOT(1->2), not the Fig. 10 entangler. The Fig. 10 entangler is
  20-27% better.
- **Train NLL identifies the plotted configuration when test KL cannot.**
  The final train NLL separated L = 1 from L = 2 (0.6839 vs 0.6815 against
  the paper's 0.681); test KL did not.
- **Use one fixed test set.** Redrawing it per run dominates run-to-run
  noise for deterministic models: the surrogate seed std was ≤ 2e-5 on a
  fixed set.
- **Overfitting minima read off the test curve are not evidence.** Use a
  separate validation split (here data seed 201, n = 1500) for early
  stopping and tuning.

## Optimiser and trainability behaviour

- **HardTanh-output surrogates are unstable under Adam.**
  - The paper's surrogate is stable only with tiny-lr SGD. Under Adam at
    lr ≥ 3e-3, the output saturates at ±1 within 1-2 steps, an observed
    label gets probability 0, and the NLL becomes infinite.
  - Even at lr 1e-3, the run with L2 = 0.1 diverged after 880-1830 steps,
    and the run with L2 = 0 ended with an infinite validation NLL.
  - Define an eligibility rule (all tuning seeds finite) before tuning.
- **Equal tuning changes conclusions.** The paper's tuned-Adam quantum model
  against fixed-lr SGD surrogates looked like a 2.9e-3 advantage. With equal
  validation-based tuning (Adam plus L2), the L = 1 surrogate beats the
  paper's L = 2 quantum model.
- **L2 grids must match the model's weight scale.**
  - The penalty is the sum of squared weights against a per-label mean
    NLL.
  - For MLP weights of size O(0.3), L2 ≥ 0.01 collapses the model to a
    constant, and L2 = 0 overfits; the optimum was 1e-3.
  - For the surrogate's tiny weights, the optimum was 0.1-1.
- **Overfitting at about 91 parameters on 1500 samples is real.**
  Unregularised MLPs and the degree-2 polynomial reached a validation NLL
  worse than the constant model (ln 2), although the polynomial class
  contains the truth.
- **Exact statevector gradients suffice.** The paper also uses backprop
  through a statevector (JAX), not parameter-shift. Torch Adam and SGD match
  optax to 1e-12.

## Photonic translation

- **Photon-number conservation enforces linear label constraints
  exactly.**
  - A readout E_k = sum_m c_{k,m} <n_m>, with column sums constant over
    modes, gives sum_k E_k = const for every input and parameter. The
    violation is < 1e-14.
  - This is the photonic analogue of generators that commute with a
    conserved observable. A literal dual-rail emulation of the qubit ansatz
    is not passive linear optics and was rejected.
- **A grouped readout gives a free noncontextual control.** If each task
  owns its own set of modes and the map from occupations to behaviours is
  injective and barycentric, the model is provably noncontextual. It
  isolates "bias" from "contextuality" at the same parameter count.
- **`ComputationSpace.UNBUNCHED` renormalises.** It is collision-free
  postselection, so a constraint that holds under FOCK also holds after
  renormalisation.
  - Training drifts towards bunching: keep rates fell to 0.44-0.63
    (N = 2) and 0.07-0.22 (N = 3).
  - Always report the keep rate.
  - PNR detection (FOCK) removed the caveat and performed slightly better.
- **More photons did not help at fixed modes.** N = 3 in 6 modes was worse
  than N = 2 (5.1-5.7e-3 vs 4.4e-3), with far lower keep rates.
- **Seed MerLin layers through the global torch RNG.** `QuantumLayer` draws
  initial phases from it: call `torch.manual_seed` immediately before
  construction.

## Baseline fairness

- When a paper's surrogate has 48-1030x more parameters, the
  parameter-matched comparators should include:
  - a ceiling whose class contains the truth (here a degree-2 polynomial,
    since win probabilities are quadratic in the features);
  - matched models with and without the constraint, to isolate the bias.
- **Compactness is the remaining quantum/photonic advantage here.** At
  about 91 parameters, the biased quantum and photonic models beat every
  classical model of the same size. Only the 8748-parameter bias-encoded
  surrogate beats them.

## Scaling limitations

- The L = 2 surrogate has 31250 features per sample. Dense test features
  take 2.5 GB in float64; a factorised (Kronecker) contraction gives the
  same function and gradients in about 0.64 GB. Runs take 23-49 min on
  CPU, depending on load.
- With m = 3 tasks, the constrained polytope has 6 vertices, so classical
  bias encodings are trivial. The paper's argument for quantum models may
  only bite at larger m, where the polytope description may grow quickly.
  This is untested.

## Promising directions

- Repeat the fair comparison at larger m, or with constraints whose
  polytope has exponentially many vertices.
- Model photon loss with postselection on N, and add a keep-rate
  regulariser for threshold-detection models.
- Tune L2 for the quantum and photonic models, and re-tune the L = 2
  surrogate; neither was done here.
