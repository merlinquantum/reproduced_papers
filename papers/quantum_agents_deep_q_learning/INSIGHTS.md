# Insights: deep Q-learning with a parametrized quantum circuit (Skolik et al. 2022)

Reusable notes for reproductions of reinforcement-learning papers, of
quantum-circuit function approximators, or of photonic adaptations of
gate-model circuits. Numbers are for Cart Pole v0 and the 4x4 deterministic
Frozen Lake, 10 learners per configuration unless stated.

## Implementation pitfalls

- **Check every hyperparameter against where the released code consumes it.**
  A key that is absent from a script can default to something different from
  the paper table: here the Cart Pole replay memory was unbounded (a `deque`
  with `maxlen=None`) instead of the tabulated 10000, and the Frozen Lake
  script updated the model every step instead of every 5. These two settings
  moved the 5-layer circuit's mean time to solve from 617 to 303 episodes
  (the paper's figure shows ~320) and Frozen Lake from 3/30 failures to none.
  Run such differences as single-change variants of the minimal arm before the
  full sweep.
- **Released circuits can differ from the circuit figure.** The Frozen Lake
  code adds a final rotation layer and uses all-to-all CZ gates. With the
  figure's ring ansatz only 4/10 learners solve, because the last RZ layer
  and the last CZ layer commute with a Z readout and cannot change it.
- **Metric definitions hide in details.** The Frozen Lake error to the optimal
  Q-values matches the paper's starting value (0.54) only over the 44
  non-terminal state-action pairs and with initial angles in [-0.05, 0.05];
  over all 64 pairs, or with larger initial angles, it does not.
- **Reproduce the optimizer rule, not just its name.** The released code uses
  Keras Adam (epsilon 1e-7 added before bias correction of the step size,
  AMSGrad for the circuit angles only, plain Adam for input and output
  weights). A per-agent masked Adam with its own step counters is needed when
  learners update at different times.
- **Vectorise independent learners with a leading learner dimension.** All
  parameters carry shape `(learners, ...)`; one process then trains 10 learners
  with results identical to separate runs (tested), at about a tenth of the
  per-learner cost on a CPU. Build each layer's operator once and apply it to
  the whole batch instead of gate by gate.
- **Seeds are not portable across environment packages.** gym 0.17.2 and
  Gymnasium 1.4.0 have equal dynamics but different random streams and
  observation dtypes; the original runs used no seeds anyway, so only
  distribution-level agreement is meaningful.
- **Run subsets first.** Runs are deterministic: a 5-learner, 2000-episode run
  is a bit-identical prefix of the 10-learner, 5000-episode run, so reduced
  runs can be launched first without wasting compute.

## Optimizer and trainability behaviour

- Hyperparameters dominate: a size-matched NN with 67 parameters solved 5/10
  untuned and 10/10 after a 6-setting grid (batch size x target-update
  interval), a larger change than the difference between model families.
- Trainable output weights (learning rate 0.1, 100x the circuit's) make the
  output range adapt; fixed [0,1] never solves (0/10 in every setting), fixed
  [0,90] solves; the paper reports fixed [0,180] as the most
  hyperparameter-sensitive, in our runs it solved 10/10 (best settings) and
  9/10 (sub-optimal).
- Removing re-uploading or the trainable input scaling slows learning and
  makes it less reliable (sub-optimal settings: 8/10 and 5/10 solved vs 10/10
  for the full model) but does not stop it in the qubit model; the photonic
  model degrades much more (3/10 and 5/10 with the best settings).
- arctan versus tanh input encoding made no measurable difference, also not
  for the ablated arms.

## Photonic translation considerations

- **`ComputationSpace.UNBUNCHED` is a post-selected subspace**, not threshold
  detection: expected clicks sum to exactly the photon number. Threshold
  detectors without post-selection need the full Fock space collapsed to
  click / no-click per mode.
- **A bounded observable per action is the photonic counterpart of
  `(<Z> + 1) / 2`**: the click probability of a threshold detector lies in
  [0, 1]. Different modes' click probabilities are coupled (the total is at
  most the photon number), unlike independent Z observables.
- **Encode once, then apply linear optics = one fixed unitary.** A phase
  encoding followed by any mesh cannot represent a general function of 16
  basis states: Frozen Lake learners solved 0/10 for 6 modes / 3 photons at
  5, 10 and 15 layers, for 8 / 4, and for a dual-rail input-state encoding.
  The supervised fit of Q* (no RL) plateaued at an error of 0.134-0.154 for
  all of them, also for 10 / 5. Re-injecting the state before every layer removes the limit (error
  0.06 at 5 layers, 0.01 at 10; agents solve 7/10, 10/10, 10/10).
- **Layer placement matters.** Phase shifters before detection and phases on a
  single input Fock state change nothing, so a trainable mesh must precede the
  first and follow the last encoding.
- **A per-sample input Fock state cannot be combined with per-sample phase
  inputs** in MerLin 0.4.2; prepare the state exactly with one interferometer
  per mode pair and a phase of `pi * (1 - bit)`.
- **Passing every circuit phase as a batched input** lets independent learners
  share one forward call and keeps their parameters in agent-stacked tensors,
  so optimizer, target network and training loop are reused unchanged.
- The CPU was faster than an L4 GPU for 56-state Fock spaces (40.5 vs 68.9 ms
  per 10-learner step).

## Baseline fairness concerns

- The paper's NNs start at 182 parameters against 62 for its best small
  circuit and have an unreported tuning budget; its "NN about twice as fast"
  statement compares unmatched models.
- At matched size and equal tuning the order flips with size: PQC faster at
  ~65 parameters (median 214 vs 298 episodes), NN faster at ~300 (232 vs 332).
  Both rest on 10 learners and one selection step; no advantage is implied.
- Use separate selection seeds, a declared tie rule and the same grid for both
  families; the selected setting of the larger PQC coincided with the paper's
  own setting.
- Compute cost points the other way in simulation: NN runs took 29-263 s,
  circuit runs 1947-6770 s (different hardware, reported for context only).

## Scaling limitations

- Wall-clock is dominated by learners that never solve and run all 5000
  episodes: the three slowest qubit arms (10 learners each, 3 in parallel)
  took 3.7 h on an L4; the 15 photonic configurations took 16.5 h on 8 CPU
  threads.
- Queue systems that run one job per machine need the concurrent configurations
  inside a single launch (a small coordinator script with per-run logs).
- Unbounded replay was estimated to need up to about 0.9 GB per 10-learner
  process (plan-time estimate, not profiled).

## Negative results

- Fig. 1 ring ansatz on Frozen Lake (4/10 solved).
- Planned photonic Frozen Lake design with the state encoded once (0/10),
  8 modes / 4 photons (0/10), dual-rail input-state encoding (0/10); 10 modes /
  5 photons only screened with the supervised fit.
- Not reproduced from the paper: "does not solve" for Cart Pole without
  re-uploading or without trainable input scaling (8/10 to 10/10 solve), and
  complete failure of fixed [0,180] with sub-optimal settings (9/10 solve).
  The paper's own Fig. 4a curves for the ablated arms sit near 190, so the
  text is likely stricter than its data.

## Debugging heuristics

- A **known-answer regression** (fit the analytic Q-values supervised) separates
  a representation limit from an exploration or optimisation failure in a few
  minutes; train longer and with another learning rate to rule out
  under-training.
- **Compare against an independent simulator** (gate-by-gate NumPy for the
  qubit circuit, Perceval SLOS for the photonic one) and check that a
  learner's results do not depend on its batch mates.
- When a probability or gradient is exactly zero, check the light cone before
  suspecting the framework: one brickwork layer cannot move a photon more than
  one mode per column.
- A test helper can be the buggy part: the first cross-check failed because
  the reference built every learner's encoding from learner 0's weights.

## Metric ambiguities

- "Episodes to solve": 0-based index of the episode meeting the rule, with
  unsolved learners censored at the cap; report solved fraction next to means.
- Statements about "agents that solve" versus "average curve reaches" are not
  interchangeable; mean score curves can approach the threshold while some
  learners never solve.
- The paper's text and figure disagree on the best average circuit (25 layers,
  500 episodes in the text; ~350 for 25 layers and ~320 for 5 layers in the
  figure) and on the best NN ((20,20), 250 in the text; ~250 for (30,30) and
  ~430 for (20,20) in the figure).
- Values read from unlabelled figures carry about +-20 episodes; state it.

## Promising future directions

- Photonic Frozen Lake without re-uploading: photon-number-resolving readout,
  adaptive measurements or nonlinear elements (not tested).
- Tune and size-match the photonic model with the same grid as the baselines;
  run its depth scan; add noise, loss and shot budgets before any hardware
  claim.
- Confirm the Cart Pole numbers with the original TensorFlow stack and
  unbounded replay, if a legacy environment can be built.
- Extend to environments with larger Q-value ranges, where the output-range
  argument (trainable output weights) matters most.
