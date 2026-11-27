# Insights - power_of_QNN

Durable, evidence-backed observations. Each one names its evidence.

## I-01: The published Fig. 3a classical curve does not come from the stated model (2026-10-20, Phase 01)

- **Stated model:** the paper and the upstream README say the Fig. 3a classical network is
  [4,4,4,2], a bias-free network with d=40.
- **Cached array:** the F_hat array behind the published curve,
  `Effective_dimension_plots/data/fhat4_[4 4 4 2]_ed.npy` in amyami187/effective_dimension
  v1.0.0, has shape (100, 28, 28). It therefore comes from a d=28 network.
- **Re-derivation:** applying Eq. 2 (gamma=1, n from 1e3 to 1e6) to the cached arrays
  reproduces `plot_fig/source_data/effective_dimension.txt` to 4 decimals for all three
  curves:
  - QNN: 0.821 -> 0.921;
  - easy model: 0.674 -> 0.815;
  - classical: 0.514 -> 0.611, from the d=28 array.
- **Stated model's values:** the d=40 [4,4,4,2] F_hat that the authors used for Fig. 2 gives a
  classical normalized effective dimension of 0.541 -> 0.534. It is lower overall and not
  monotone in n, with a minimum of about 0.521 at n=1e4.
- **Implications:**
  - The ordering QNN > easy > classical holds either way.
  - The published classical curve's values and upward trend do not describe the stated
    model.
  - Any comparison against the paper's classical curve must say which model it refers to.
  - Some effective-dimension computations sample parameters from U[0,1], not the paper's
    [-1,1]^d. The authors' cached arrays were produced that way (`model_base.py`:
    thetamin=0, thetamax=1).

Evidence: LOG.md, "Paper-versus-code discrepancies" item 6, and the Session Handoff of
2026-10-20.

**Update 2026-10-21 (Phase 03):** the d=28 network is identified as **[4,4,2,2]**. Among all
10 bias-free d=28 topologies with <= 3 hidden layers, it alone reproduces the published
classical curve (within 0.003 at every n, mean F_hat rank 20.99 vs 21.09 for the cached
array). The published Fig. 3a therefore compares 40-parameter quantum models with a
28-parameter classical network. Evidence: LOG.md "Gap diagnosis: Fig. 3a classical curve".

## I-02: The QNN's Fisher entries vanish exponentially with qubits, like the easy model's (2026-10-21, Phase 03)

The paper's Figs. 2 and 9 plot F_hat, normalized so that mean tr F_hat = d, which removes the
overall scale. Sec. 3.4 links barren plateaus to the *unnormalized* trace
(tr E[F] <= d (omega_S + omega_S^2)).

- **Measured trend:** the per-parameter trace tr F / d for S = 4, 6, 8, 10 (theta ~ U[0,1],
  N(0,1) inputs, 3-5 draws, cv <= 3%) shrinks by:
  - x0.494 per qubit for the QNN (ZZ, reps 2): 3.56e-2 -> 5.17e-4;
  - x0.528 per qubit for the easy model: 3.22e-2 -> 6.96e-4.
- **Cross-check:** our S=6/8/10 quantum F_hat equal the authors' Fig. 9 arrays element-wise
  (<= 2.8e-14), so the authors' own data contain this decay.
- **Interpretation:**
  - The normalized-shape statement is reproduced. The QNN's max eigenvalue stays ~4.5-5.8,
    while the easy model's outliers grow to ~54.
  - But by the paper's own criterion both quantum models show barren-plateau-like vanishing
    Fisher entries.
  - The global parity readout is a plausible reason: cost functions built on global
    observables are known to produce barren plateaus.
  - Any claim of barren-plateau "resilience" needs the unnormalized scale.
- **Evidence:** `results/fisher_trace_scaling.png`; LOG.md "Phase 03 Results".

## I-03: Table 1's Fisher-Rao norm ordering is set by the parameter-initialization range (2026-10-21, Phase 03)

The authors compute the Fisher-Rao norm as theta^T F_hat theta, with F_hat normalized to
trace d at the trained theta. It therefore scales roughly with ||theta||^2.

- **The authors' initialization ranges** (quantum U[0,2pi], classical U[0,1]): QNN 115.6,
  easy 109.9, classical 63.3. This matches the paper's 117.8 / 104.9 / 46.5.
- **All models initialized in U[-1,1]:** QNN 9.7, easy 6.9, classical 259.9.
- The ordering reverses while the training losses barely change, so the "~250% higher" claim
  measures parameter scale, not capacity. Evidence: LOG.md "Phase 03 Results" (E-03b, A-01).

## I-04: In a native photonic model, chip size sets the capacity comparison, not the encoding (2026-10-22, Phase 04)

- **Setup:** the MerLin adaptation (`lib/photonic_merlin.py`) at d=40, s_in=4.
- **6 modes / 3 photons:** the Fisher rank caps at ~24/40, so the normalized effective
  dimension is 0.45-0.49 for both encodings. That is below the classical [4,4,4,2] (0.51-0.54)
  and far below the qubit QNN (0.82-0.92).
- **8 modes / 4 photons:** the rank rises to ~33.6 and the hard model reaches 0.50-0.58, above
  [4,4,4,2] for n >= 1e4.
- **Encoding effect:** the hard re-uploaded encoding beats the easy one at both chip sizes, but
  with a gap (0.02-0.09) much smaller than the chip-size effect.
- **Barren plateaus:** the photonic Fisher scale vanishes per added input at the same rate as
  the qubit models (x0.47 vs x0.49).
- **Implication:** a "photonic effective dimension" comparison at matched d is meaningless
  without stating the number of modes and photons. The rank cap (unitary degrees of freedom of
  the chip vs d) should be checked before reading capacity differences as encoding effects.
- **Evidence:** `results/phase04_tables.md`, `results/merlin_effective_dimension_d40.png`;
  LOG.md "Phase 04 Results".

**Update 2026-10-24 (design searches 1-2, LOG.md Amendments 2-3):** design, not photonics, set the low planned score.
- **Levers:** interleaved re-uploading lifts the photonic hard model from 0.49 to 0.76 at
  6 modes. Adding 8 modes and a spin-emitter cluster input reaches 0.877 (confirmed), within
  0.02 of the qubit QNN.
- **Dead phases:** part of the original rank deficit was phases placed after the last beam
  splitter on their mode. Evidence: LOG.md, Amendments 2-3.

## I-05: The paper's 4-qubit QNN is a single-photon interferometer (QLOQ [4]) (2026-10-24, Phase 04)

- **Construction:** with qubit logic on qudits (arXiv:2411.03878), all 4 qubits are encoded in
  one photon over 16 modes. Every gate of the paper's circuit becomes passive linear optics:
  - H and RY become beam splitters on mode pairs;
  - P and ZZ become phase shifters;
  - CX becomes a mode permutation.
- **Equivalence:** the resulting F_hat equals the qubit F_hat element-wise (<= 4.9e-15), so the
  effective dimension is identical (0.82 -> 0.92).
- **Implication:** the capacity reported for the "quantum neural network" needs neither
  entanglement between particles nor multi-photon interference. A 16-dimensional
  single-particle unitary is classically simulable in O(16^2) per sample.
- **Consequence for the paper:** its capacity claims are statements about a 16-dimensional
  linear model class, not evidence of a quantum resource. The barren-plateau-like decay of the
  Fisher scale (I-02) is unaffected.
- **Evidence:** `outdir/run_20261024-121233`;
  `tests/test_photonic_merlin.py::test_qloq_single_photon_reproduces_qubit_qnn_exactly`.

## I-06: A native photonic design can exceed the qubit QNN's effective dimension, but only with an entangled emitter input and extra re-uploading (2026-10-28, Phase 04)

- **Result:** at matched d=40, s_in=4 and theta ~ U[0,1], the photonic 10-mode design (10 modes, 5 photons,
  linear-cluster emitter input, 3 interleaved data re-uploads) reaches 0.9096 / 0.9295 at
  n = 1e5 / 1e6. The qubit QNN on the same draws reaches 0.8926 / 0.9165 with 3 data uploads and
  0.8952 / 0.9186 with the paper's 2. Every gap is resolved at > 3 std (fresh draws 10-14).
- **Ingredients:**
  - **Entangled input:** separable inputs get *worse* on bigger chips (8 -> 12 modes:
    0.82 -> 0.75 at n=1e5), while cluster inputs stay at 0.88-0.89.
  - **Extra re-uploading:** it helps only the photonic model (0.891 -> 0.911; qubit
    0.896 -> 0.895).
- **Limits:**
  - The advantage disappears with theta in [0, 2pi] (tie).
  - It is a Fisher-geometry capacity statement on ideal hardware, not a learning result.
    When trained (Phase 05), the photonic 10-mode design was the worst learner of all models (I-07).
  - The photonic state space (2002 Fock states) is much larger than the qubit register (16).
- **Evidence:** `outdir/run_20261025-102902`, `outdir/run_20261025-103433`,
  `outdir/run_20261025-103537`; LOG.md Amendment 4.


## I-07: Higher effective dimension did not mean better learning, for qubit or photonic models (2026-10-28, Phase 05)

- **Setting:** the same optimizer, init U[-1,1], 100 iterations, and a per-model lr selected on
  held-out data (or training loss for the paper's d=8 task); 100 fresh seeds.
- **Paper claim C-04 reversed:** at d=8 on Iris 0/1 the linear [4,2] (0.009) and the paper's
  [4,1,1,1,2] (0.142) reach a lower training CE than the QNN (0.230). The paper's Table 1 ordering
  came from one shared lr and an all-positive classical init.
- **Photonic 10-mode design:** at d=40 it has the highest normalized effective dimension of all models
  (Phase 4: 0.910 vs 0.893 for the qubit QNN). Yet it has the worst held-out CE on both Iris tasks
  (0.44-0.46 vs 0.05-0.30), although its accuracy is >= 0.99: its outputs stay near 50/50.
- **Lesson for photonic design searches:** effective dimension counts how many parameter
  directions change the output distribution, not how far the class probability can move. Design
  searches should also score output margin or a short training run, not effective dimension
  alone.

- **Follow-up (design search 4, LOG.md Amendment 5):** the cause is an output-margin ceiling of the 10-mode cluster design
  (single-input max p ≈ 0.8), not parity readout in general. Searching capacity and loss together
  selected the photonic 6-mode design (6 modes / 3 photons, cluster input). It keeps the qubit-level effective dimension, and
  on Iris 1/2 it beats both qubit QNNs and ties the classical models. It still loses on Iris 0/1.
  Adding modes or a cluster input moves a design along a capacity-versus-loss trade-off.

- **Out-of-selection check (breast cancer, MNIST 3 vs 5):** the photonic 6-mode design beats both qubit QNNs on both
  tasks (MNIST accuracy 0.824 vs 0.719), so the photonic-over-qubit result holds on tasks that
  played no part in selection. Every quantum model, photonic or qubit, stays far behind an 8-parameter
  linear classifier (MNIST 0.879). Among quantum models at matched d, a small cluster-input
  photonic chip is the strongest learner tested here. No quantum model is competitive with
  classical baselines on these tasks.

Evidence: LOG.md "Phase 5 K1 result", "Phase 5 extension results X1, X2", "X3, X4"; "Amendment 5" screening and confirmation;
`results/phase05_tables.md`; runs `outdir/run_20261028-152522`, `-155259`, `-160751`.

## I-08: Practical heuristics for native photonic classifiers in MerLin (2026-10-29, Phases 04-05)

- **Re-uploading first:** interleaving data encodings with trainable meshes was the largest
  capacity lever (+0.27 normalized effective dimension). Feed-forward and PNR readout added
  nothing here.
- **Check for dead phases:** a phase after the last beam splitter on its mode never affects
  detection. Order meshes PS -> BS, and test that every parameter has a nonzero gradient.
- **Score both capacity and margin:**
  - Before calling a design good, maximize p(target class) for a single input over all phases.
    Values well below 1 (photonic 10-mode: about 0.8) predict a high cross-entropy floor whatever the capacity.
  - Larger chips and emitter cluster inputs trade margin for capacity.
  - 6 modes / 3 photons with a cluster input was the best compromise found (the photonic 6-mode design).
- **Detector modelling:** `ComputationSpace.UNBUNCHED` postselects and renormalizes (37-43%
  kept here). Threshold detection is FOCK simulation grouped by click pattern.
- **Emitter states:** use `spin_gate(pi/2, 0)` everywhere for a linear cluster, or a pi/2
  preparation followed by identity gates for GHZ. Project the emitter with `final`: tracing it
  out (`final=None`) leaves a classical mixture.
- **Per-sample parameters:** passing trainable phases as an `input_parameters` group gives one
  parameter row per sample (needed for the Fisher). Columns follow name order, so name the
  parameters so that name order equals circuit order.
- **Cost:** simulation time grows with the Fock dimension. The photonic 6-mode design (56 states) ran at 1-3x
  the qubit QNN's wall-clock; the 10-mode one (2,002 states) at 35-160x.

Evidence: LOG.md Amendments 2-5, "MerLin API Notes"; `results/phase04_design_screens.md`;
`tests/test_photonic_merlin.py`; `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`.

## I-09: Methodology lessons that changed verdicts in this reproduction (2026-10-29, Phases 01-05)

- **Released raw arrays are the best oracle:**
  - re-deriving a published curve from the authors' cache exposed the wrong classical model
    (I-01);
  - element-wise Fisher matching fixed the Hadamard convention.
- **Tune every model's learning rate before comparing training speed.** A single shared lr plus
  an all-positive classical init produced the paper's Table 1 ranking. Per-model selection on
  held-out data reversed it (C-04).
- **Normalized quantities hide scale:**
  - the normalized Fisher spectrum looked stable while the unnormalized trace halved per qubit
    (I-02);
  - the Fisher-Rao norm followed ‖θ‖² (I-03).
- **Confirm a selected design on data it was not selected on.** The photonic 6-mode design was chosen on Iris 1/2. Its
  advantage over the qubit QNN held (and grew) on breast cancer and MNIST 3 vs 5. The 10-mode design's
  capacity win did not survive the first learning test.
- **Read "x% randomized labels" literally:** overwriting x% with uniform labels changes only
  about x/2% of them.

Evidence: LOG.md (paper-versus-code discrepancies, Phase 03 results, Phase 5 plan and results);
`results/phase03_tables.md`, `results/phase05_tables.md`.

