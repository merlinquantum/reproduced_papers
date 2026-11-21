# Insights — The trainability of photonic quantum circuits (arXiv:2607.21544)

Durable observations for future reproductions involving Haar-random linear
optics, photon-number observables, shot-based training or photonic
trainability claims.

## Exact Haar moments of photon-number observables

- **Truncate the irrep sum at the observable's order.** A photon-number
  monomial of order |alpha| is a sum of normally ordered p-body operators with
  p ≤ |alpha|, so its projection onto irreps k > |alpha| is exactly zero.
  Second moments `mu2 = sum_k ||P_k(rho)||^2 ||P_k(O)||^2 / d_k` (Mhiri et al.
  Cor. 1) then need only `g_l` for l ≤ |alpha|. This makes M = 250-2000 with
  n = M/2 exact in seconds.
  - Use exact integers or fractions: the alternating sum in the irrep norms
    cancels catastrophically in floats at n ≳ 20.
  - Group the lowered patterns by (support part, off-support total). The
    off-support sum has a closed form, `C(N + r - 1, A + r - 1)`.
- **Validate against independent computations before trusting a figure.**
  Useful checks:
  - an explicit-matrix lowering map at small n;
  - `sum_k d_k = |H|^2` and `sum_k ||P_k(O)||^2 = tr O^2`;
  - Mhiri's closed form for P2 against summed projector second moments;
  - Haar Monte Carlo with Perceval SLOS (|z| < 4).
- **The input state sets the decay exponent of the circuit variance.**
  - Uniform inputs (`|1,...,1>`, `|2,...,2>`) have exactly zero projection on
    the k = 1 irrep (their one-body reduced state is proportional to the
    identity). The leading contribution starts at k = 2, and Var_circ decays
    as ~M^-3.
  - The alternating input `|1,0,1,0,...>` has a k = 1 component, and Var_circ
    decays as ~M^-1 (local slopes -0.996 and -2.995 at M = 2000).
  - Either way the decay is polynomial, so fixed-order PNMs stay trainable.
- **Do not decide polynomial versus exponential decay from a short range.** On
  M = 40-340, an exponential fits the decaying branch better (AIC) for every
  Fig. 3a PNM and for high powers in Fig. 4, because the curve is still
  turning over after its peak. The trend of the local slope
  `d log Var / d log M` is the sharper diagnostic. Extend the range when the
  exact computation is cheap.
- **Bounds derived from fixed-K limits fail when the support grows with M.**
  The paper's Eq. 16 (`Var_samp ≥ (2^K - 1) nu^{2K}`) is violated at every
  density for K = M/2. With n ≈ K photons, the product of K number operators
  is nearly a projector, and the circuit variance collapses instead.

## Random-observable ensembles

- **The term-selection scheme can change results by orders of magnitude.**
  Fig. 5c,d ("5M randomly selected PNMs of weight ≤ 4") came out 150-240x low
  when monomials were drawn uniformly. At large M, uniform-over-monomials
  almost never draws single-mode powers such as n^4, which dominate both
  variances. Drawing exponent patterns (integer partitions) uniformly matches
  the paper.
- **Diagnose random-ensemble gaps analytically.** With i.i.d. zero-mean
  coefficients, the expected Var_samp and Var_circ are exactly
  `E[c^2] × sum over terms of the per-term variance` (cross terms vanish).
  Computing the per-pattern variances and mixing them with each candidate
  sampling scheme's probabilities identifies the scheme in minutes, before any
  full run.
- **Positive coefficients do not make the quantum sample variance quadratic.**
  Sums of positive photon-number products are nearly constant, because the
  total photon number is conserved. Their sample variance grows like |A|, and
  in the C9 comparison even more slowly (|A|^0.53). Arguments that rely on
  sign cancellation need checking against this.

## Classical comparison (estimation cost)

- **Lim-Oh (term-wise) does scale as |A|^2, but with a large constant.** Each
  factor n^a must be truncated at the photon number n to be Hilbert-Schmidt.
  Per-term norms then grow with n (`sum_k k^4 = 8,772` at n = 8), so
  "O(1) per term" assumptions fail at fixed density.
- **Exact classical evaluation is the baseline to beat for fixed-order
  polynomials.** For collision-free Fock inputs,
  `<a^+_J a_J> = sum_{|I| = p} |Perm(U[I, J])|^2`. A fixed-order PNP costs
  `|A| × C(n, p)` small permanents. At M = 16, |A| = 150 this gave the exact
  loss in 46 ms, against a projected 1,620 s for Lim-Oh at eps = 0.1. Quantum advantage
  claims must target observables whose expansion is exponential (for example
  deep NN observables) and must include exact and emulator baselines.
- **Neyman allocation is the fair way to split classical samples across
  terms.** `N = (sum |c| sigma)^2 / eps^2` is never more than equal allocation,
  and it gives the classical side its best case.

## Shot-based training and gradients

- **GPSR frequency bound.** For phase parameters with n photons, the loss is a
  trigonometric polynomial of degree ≤ min(order, n), so GPSR with
  R = min(order, n) is exact. It stays exact under partial distinguishability,
  uniform loss without post-selection, and threshold detection (all tested
  against autograd). Post-selected losses are ratios of trigonometric
  polynomials, and fixed-shift GPSR is not exact for them.
- **Shot budgets in the paper's range are generous at small M.** With 10,000
  shots per shifted loss, shot-based GPSR matches the exact-gradient
  trajectory almost perfectly at M = 8. At M = 12 it reaches ~0.8x of it, and
  it beats an equal-shot SPSA control. Both readings of "samples per gradient
  estimate" lead to the same conclusion.
- **Learning-rate selection must be a coordinated sweep.** One candidate per
  run, a dated plan taken from `date -u`, coordinator status files. Selections
  embedded in a training run are not auditable.

## Photonic hardware effects (MerLin)

- **Uniform loss with post-selection on n photons gives exactly the lossless
  statistics,** because loss commutes with passive linear optics. The cost is
  the accepted-shot fraction t^n, which a variance ratio does not show.
- **Threshold detection bounds the observable but does not help
  trainability.** Var_samp of n0^2 drops 15x at M = 8, Var_circ by a similar
  factor, so the ratio is unchanged within ~30%.
- **Partial distinguishability (V = 0.7) changes the variance ratio by ≤ 18%**
  at M ≤ 8. Hong-Ou-Mandel `P(1,1) = (1 - V) / 2` is a quick check that the
  noise model is wired correctly.
- **`ComputationSpace.UNBUNCHED` is collision-free post-selection with
  renormalization, not threshold detection.** It returns P(1,1) = 1 for HOM.
  Model threshold detectors with `pcvl.Detector.threshold()` via
  `pcvl.Experiment`.

## Useful implementation patterns

- **Haar initialisation of a trainable Clements mesh.** Decompose a Haar
  unitary exactly with Clements nulling, using phase-only MZIs
  `B · PS(delta) · B · PS(phi)`. Then commute the leftover diagonal through
  the left blocks with `MZI(d, p)^+ D = D' MZI(d, p')`, where
  `p' = arg(d1/d2)`. The result reconstructs U to 1e-15. The input-side phases
  are irrelevant for Fock inputs and need no training.
- **Convention traps (verified numerically):**
  - Perceval's `Unitary(M)` is the transpose of the paper convention
    `a_i^+ -> sum_j U_ij a_j^+`.
  - Lim-Oh's matrix V is the transpose of that convention, so their
    `V^dagger alpha` is `conj(U) @ alpha` here.
  - Check each against a permanent at M = 4.
- **MerLin for GPSR.** Pass the whole `(P × 2R, P)` batch of shifted parameter
  vectors through `input_parameters` in one call; values map in circuit order.
  MerLin computes full distributions, so the practical limit is a Fock
  dimension of about 1e5. Hardware-scale circuits (48 modes, 6 photons,
  dimension 2.3e7) need a sampling path.

## Open directions

- The NN-observable regime (2^L-order polynomials with exponentially many
  terms), against exact evaluation after expansion and against
  marginal-correlator emulators: the paper's strongest advantage argument,
  untested here.
- An analytic derivation of the M^-1 and M^-3 decay laws from the k ≤ order
  irrep sum: the numerics strongly suggest closed forms.
- Shot-based training at PT-2 scale with a sampling backend, and the loss
  cost of post-selection there.
