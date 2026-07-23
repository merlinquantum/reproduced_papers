# Insights — CV-QPINN for PDEs

Durable observations from the reproduction work, intended for reuse by
future QPINN / CV-quantum / PINN reproductions.

## Scientific insights

- **Consistency-loss trick is generic.** The "second-output ≈ derivative of
  first-output" trick decouples the size of the loss-graph from the
  derivative order. It works equally well for a CV-quantum model and for a
  small classical FFN, and it should slot into any other quantum-PINN
  implementation that exposes more than one output. Paper attribution
  belongs to Panichi et al. 2025, but the idea is library-agnostic.
- **Paper's "fair" baseline is fragile.** When the classical PINN is given
  the *same* consistency-loss training scheme used by the QPINN, it can
  match or beat the QPINN on the same parameter budget for both the 1D
  Poisson and 1D heat problems we tested. Quantum advantage claims from
  Table IV should be read with that caveat in mind — the gap reported
  there is plausibly attributable to optimisation effort rather than to
  the quantum architecture.
- **Trace normalisation is largely automatic** in the Killoran ansatz when
  every gate is implemented as a unitary on a sufficiently large Fock
  truncation. The trace-loss term in §III.B is needed by Strawberry Fields
  because SF's TensorFlow backend computes density matrices that *can*
  drift; in a pure-state autograd simulator it stays at 1 to within
  `1e-15` without any penalty.

## Implementation insights

- **PyTorch `matrix_exp` is differentiable** and is the cleanest way to get
  CV-gate matrices on a Fock truncation without writing Laguerre-polynomial
  formulae by hand. Batched `matrix_exp` works for input-dependent
  displacements (the `D(x)` encoding) at 64+ batch sizes.
- **Cutoff `n` must dominate both `max x` and `max |u|`.** For our
  Poisson on `[0, π/2]` the analytic |u| is `1/16`, so cutoff 8 is
  plenty. For the heat equation the displacement encodes both x and t,
  so 10–20 is required at the corners; we observed sub-percent norm
  leakage at cutoff 10 and essentially exact normalisation at cutoff 15.
- **Two-mode gates dominate the wall-clock.** A 4 + 4 layer model with
  cutoff 10 needs ~4 BS matrix exponentials of dim 100 per forward pass,
  which becomes the bottleneck. For deeper experiments a cached BS
  unitary (re-built only on parameter update, not on every forward)
  would be worth investigating — currently the BS matrix is rebuilt
  every batch even though its parameters do not depend on the batch
  index.
- **One-vs-two-qumode input encoding.** Encoding the input on the same
  qumode that will be measured as `u` (mode 0) leaves the `ux` qumode
  in vacuum at the start of training; we observed that this initially
  makes the consistency loss the dominant term. A coherent way to fix
  this would be to encode the input on *both* qumodes via a (1, x)
  displacement; we did not need this in practice because Adam converges
  the consistency residual within ~200 epochs.

## MerLin / photonic translation insights

- **CV ↔ linear-optics is not a translation problem.** Squeezing,
  displacement, and the Kerr non-linearity have no clean MerLin analogue.
  Pretending otherwise (e.g. by replacing squeezing with a passive
  beam-splitter mesh) destroys the inductive bias the CV paper relies
  on. A scientifically honest MerLin photonic counterpart is therefore
  a *different* architecture that reuses the *consistency-loss training*
  scheme — that's what we ship as `poisson_merlin`.
- **Threshold-detection probabilities can produce smooth function
  approximators** once a small trainable linear head maps the output
  probability distribution to a scalar. We hit RMSE ≈ 1e-2 on Poisson
  with `n_modes=6`, 3 photons, 3 entangling layers, and 162 params —
  comparable to the paper's classical PINN baseline.
- **MerLin `add_angle_encoding` scale matters.** For inputs on
  `[0, π/2]` we use `scale = π/2`; using the default `π` collapses the
  phase wrap-around and stalls training.

## Trainability and optimisation insights

- **Cosine annealing with warm restarts** (§IV.B of the paper) is helpful
  but not strictly necessary. We obtained comparable smoke-quality
  metrics with a flat 0.01–0.05 Adam learning rate; we ship the schedule
  hooks but leave it disabled by default.
- **IC pre-training amortises well.** For the heat equation,
  pre-training only on the IC loss (no PDE, no consistency, no BC) for
  ~10% of total epochs jump-starts the network to a state where the
  full loss is well-conditioned. The paper recommends 300 IC-only
  epochs; we observed that 60 already shifts the bulk of subsequent
  improvement to the PDE term.

## Anti-patterns to avoid

- Do **not** check an analytic solution against arbitrary boundary
  conditions without verifying it satisfies them. We initially shipped
  `u(x) = sin(4x)/16 - x/4` as the Poisson reference and chased a 0.23
  RMSE for several runs before realising the `- x/4` term violates
  `u(π/2) = 0`. The true solution is `sin(4x)/16`.
- Do **not** trust `merlin.__version__` — the PyPI metadata is
  authoritative.
- Do **not** treat the classical PINN baseline in Table IV as
  necessarily fair without re-running it with the same training
  enhancements (consistency loss, cosine LR schedule, IC pre-training).
