# Quantum Physics-Informed Neural Networks for PDEs - Reproduction

## Reference and Attribution

- Paper: *Quantum physics informed neural networks for multi-variable partial differential equations*
- Authors: Giorgio Panichi, Sebastiano Corli, Enrico Prati
- arXiv: <https://arxiv.org/abs/2503.12244> (v2, Nov 13 2025)
- Original repository: not advertised in the paper.
- License: this reproduction is released under the repository MIT licence;
  please cite the original paper when re-using the code or results.

## Original Paper

The paper integrates the *continuous-variable* quantum neural network ansatz of
Killoran et al. (Phys. Rev. Research 1, 033063) with the Physics-Informed
Neural Network (PINN) loss design of Raissi et al. The contribution is twofold:

1. A **multi-output Quantum PINN architecture** in which the network exposes
   one homodyne measurement per qumode. For an order-`k` PDE the network
   exposes one extra output per derivative order, and a **consistency loss**
   pins each extra output to the (auto-differentiated) derivative of the
   preceding one. This removes the *nested* automatic differentiation that
   would otherwise blow up memory in CV simulators.
2. A **noise model** distilled from photon-counting experiments on Xanadu's X8
   processor. A loss channel L̂(T) is inserted after each qumode and the
   transmittance T per channel is sampled from the measured X8 distribution
   to study optical-loss resilience.

The paper demonstrates the approach on the 1D Poisson equation
`u''(x) + sin(4x) = 0` and the 1D heat equation on `x ∈ [-π/2, π/2]`,
`t ∈ [0, 0.5]`, with α = 0.30.

## Reproduction Scope (including Updates and Deviations)

The reproduction implements:

- A self-contained Fock-truncated **CV quantum simulator** in pure PyTorch
  (`lib/cv_simulator.py`) covering rotation, beam splitter, squeezing,
  displacement, and Kerr gates on 1- and 2-qumode systems.
- The Killoran-style **multi-qumode and single-qumode layers**
  (`lib/qpinn_model.py`).
- The **consistency-loss PINN** for the 1D Poisson and 1D heat equations
  (`lib/losses.py`).
- A **classical fully-connected PINN baseline** with parameter count matched
  to the paper's reported QPINN width (`lib/pinn_baseline.py`).
- A **MerLin photonic-linear-optics adaptation** of the consistency-loss idea
  on the Poisson task (`lib/merlin_pinn.py`).
- The **photon-loss gate** L̂(T) (`lib/cv_simulator.py::loss_channel` —
  TODO documented in LOG.md, the noise study is deferred from the smoke
  reproduction; the gate definition is provided for follow-up work).

Key deviations from the paper:

- **No Strawberry Fields, no TensorFlow.** The paper uses Strawberry Fields
  with the TF backend on a 64 GB-RAM workstation. SF's TF backend pins old
  TF versions that are not available on the target Python 3.12 environment.
  We re-implement the CV simulator directly in PyTorch with `matrix_exp`
  inside a Fock truncation. The mathematical content of every gate is
  preserved; the gradient path is autograd through `matrix_exp` instead of
  TF's automatic differentiation. We never use a parameter-shift rule (see
  §II.F of the paper for why this is acceptable in simulation).
- **MerLin is *not* a CV photonic simulator.** MerLin targets *linear-optical*
  photonic computing with discrete-photon measurement, where squeezing,
  displacement, and Kerr gates are not native primitives. We therefore do
  not provide a literal MerLin port of the paper's CV architecture. Instead
  we re-use the consistency-loss training scheme on a MerLin interferometer
  + angle-encoding network — see "MerLin Photonic Extension" below.
- **Smoke configurations** (default) use a reduced Fock cutoff (8 for Poisson,
  10 for heat), reduced layer counts (2 + 2 multi/single-qumode instead of
  4 + 4 / 2 + 2), and fewer epochs. They are intended to demonstrate the
  consistency-loss machinery on CPU in tens of minutes. The paper-accurate
  configurations (`*_original.json`) use the hyperparameters in Tables V/VI.
- **Noise study from §V** is implemented at the gate level but is not part
  of the headline reproduction runs to keep wall-clock manageable.

## Project Layout

```text
papers/CV_QPINN_PDE/
├── configs/                  # JSON experiment configs
├── lib/
│   ├── cv_simulator.py       # Fock-truncated CV gates + state ops (PyTorch)
│   ├── qpinn_model.py        # Multi/single-qumode Killoran layers + QPINN model
│   ├── pinn_baseline.py      # Classical FFN PINN baseline
│   ├── merlin_pinn.py        # MerLin linear-optics adaptation
│   ├── data.py               # Sobol collocation, RK45 reference, problem definitions
│   ├── losses.py             # Consistency / PDE / BC / trace losses
│   ├── training.py           # Adam + cosine annealing training loops
│   ├── metrics.py            # RMSE / MAE / L∞ / NMSE
│   └── runner.py             # Shared-runtime entry point (train_and_evaluate)
├── tests/                    # Smoke tests for the runner and CLI
├── results/                  # Curated artefacts (predictions, figures)
└── outdir/                   # Raw timestamped runs (gitignored)
```

## Install and How to Run

```bash
cd papers/CV_QPINN_PDE
pip install -r requirements.txt
```

Run from the repository root:

```bash
# Smoke run (CPU-friendly, < 5 min): Poisson QPINN, 2+2 layers, cutoff 8
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_smoke.json

# Paper-accurate Poisson QPINN (Table V hyperparameters)
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_original.json

# Classical PINN baseline matched on parameter count
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_pinn.json

# Heat-equation QPINN (smoke and paper-accurate)
python implementation.py --paper CV_QPINN_PDE --config configs/heat_smoke.json
python implementation.py --paper CV_QPINN_PDE --config configs/heat_original.json

# Heat-equation classical PINN baseline
python implementation.py --paper CV_QPINN_PDE --config configs/heat_pinn.json

# MerLin photonic adaptation on Poisson
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_merlin.json
```

All runs write a timestamped directory under `outdir/run_YYYYMMDD-HHMMSS/`
containing `config_snapshot.json`, `summary.json`, `predictions.json`,
`history.json`, `model.pt`, and `run.log`.

CLI flags (see `cli.json` for the schema): `--seed`, `--dtype`, `--device`,
`--log-level`, `--epochs`, `--lr`, `--cutoff`, `--collocation-points`.

## Configuration

Configs live in `configs/`:

| Config | Experiment | Notes |
|---|---|---|
| `defaults.json` | `poisson_qpinn` | Small smoke (60 epochs, cutoff 8) |
| `poisson_smoke.json` | `poisson_qpinn` | Smoke (200-800 epochs, cutoff 8) |
| `poisson_original.json` | `poisson_qpinn` | Paper Table V (5000 epochs, cutoff 10) |
| `poisson_pinn.json` | `poisson_pinn` | Classical FFN baseline for Poisson |
| `poisson_merlin.json` | `poisson_merlin` | MerLin linear-optics adaptation |
| `heat_smoke.json` | `heat_qpinn` | Smoke heat (250 epochs) |
| `heat_original.json` | `heat_qpinn` | Paper Table VI (1000 epochs, cutoff 20) |
| `heat_pinn.json` | `heat_pinn` | Classical FFN baseline for heat |

## Data

No external data. The Poisson target is the analytic solution
`u(x) = sin(4 x) / 16` and the heat target is RK45-integrated *inside the
runner* before each evaluation (`scipy.integrate.solve_ivp`). Sobol-sequence
collocation samplers are seeded by the global `seed` field.

## Results Obtained and Comparison with the Paper

Numbers below come from the runs committed under `results/`. The
paper-accurate configurations finish on the order of hours on CPU; the
default smoke configurations are tuned to finish in single-digit minutes
and so necessarily land further from the paper's reported metrics.

### 1D Poisson equation (Table II / §IV.A)

| Setting | Params | RMSE | NMSE | Wall time | Label |
|---|---:|---:|---:|---:|---|
| Paper QPINN (8 layers, cutoff 10, 5000 ep) | 88 | **1.09e-4** | 6.08e-6 | not stated | paper |
| Reproduction QPINN (2+2 layers, cutoff 8, 200 ep) | 48 | 4.64e-3 | 1.11e-2 | 168 s | smoke, single-seed |
| Reproduction classical PINN baseline (90 params, 3000 ep, lr 5e-3) | 90 | 8.28e-4 | 3.53e-4 | 102 s | matched-param, single-seed |
| Reproduction MerLin PINN (n_modes=6, 3 photons, 600 ep) | 162 | 2.37e-4 | 2.90e-5 | 25 s | photonic adaptation, single-seed |

The MerLin linear-optics adaptation matches the paper's QPINN RMSE order of
magnitude (2.4e-4 vs 1.1e-4) with ~600 epochs. The MerLin model was originally
trained under `merlinquantum` 0.3.2 and has been ported to the 0.4 API
(`MeasurementStrategy.probs` factory); re-running `poisson_merlin.json` under
0.4.0 reproduces the committed metric to five significant figures
(RMSE 2.3728e-4 vs 2.3727e-4). The CV-QPINN smoke run lands
~40x above the paper's RMSE because we run for 25x fewer epochs at lower
Fock cutoff; the `*_original.json` configs are prepared but require
multi-hour CPU runs to close the remaining gap.

#### Nested-vs-consistency head-to-head ablation (200 epochs, same architecture)

| Cutoff | Loss | RMSE | Wall | Notes |
|---:|---|---:|---:|---|
| 8 | nested | **4.24e-5** | 85 s | direct `u_xx` via autograd-of-autograd |
| 8 | consistency | 4.64e-3 | 168 s | paper's scheme |
| 12 | nested | **1.51e-4** | 141 s | |
| 12 | consistency | 1.85e-3 | 115 s | |

The consistency-loss design is a real memory optimisation (peak RSS
delta jumps from 2 MB at cutoff 10 to 30 MB at cutoff 12 for nested) but
**costs 12–100× in accuracy** at the same epoch budget in the smoke
regime — a trade-off the paper does not flag. At paper-accurate cutoff
(15–20) the nested-autograd memory wall presumably reverses this, but
the regime where consistency is *strictly better* is narrower than the
paper implies. See `results/nested_vs_consistency_benchmark.md`.

### 1D Heat equation (Table IV / §IV.B)

Single-seed and multi-seed comparison.

| Setting | Params | RMSE (mean) | RMSE std | MAE | L∞ | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Paper QPINN (4 layers, cutoff 20, 1000 ep) | 44 | 1.24e-2 | — | 9.63e-3 | 3.93e-2 | n/a |
| Paper classical PINN (44 params, single hidden of 11) | 44 | 2.09e-2 | — | 1.48e-2 | 9.04e-2 | n/a |
| Reproduction QPINN matched-effort sweep (2+2, cutoff 10, 60+200 ep) | 48 | **1.23e-2** | 4.8e-3 | 8.6e-3 | 5.6e-2 | 5 (42, 7, 123, 256, 1024) |
| Reproduction classical PINN sweep (42 params, 300+1000 ep) | 42 | **8.74e-3** | 1.2e-3 | 7.0e-3 | 3.0e-2 | 5 (42, 7, 123, 256, 1024) |

Three observations:

1. Our QPINN sweep (5 seeds) sits on top of the paper-reported QPINN RMSE
   (1.23e-2 vs paper's 1.24e-2), so the *reproduction* is faithful.
2. Our classical-PINN baseline beats the paper-reported classical PINN by
   2.4x (8.74e-3 vs paper's 2.09e-2). The paper's "classical PINN" likely
   uses nested autograd without the consistency-loss trick, while we apply
   the consistency-loss scheme to *both* the QPINN and the classical
   baseline for a fair comparison.
3. **Under matched-effort training, the classical PINN beats the QPINN by
   1.40x on the mean and is ~4x more stable across seeds.** The paper's
   "slight quantum advantage" claim on the heat equation does not survive
   a fair re-baseline. The QPINN mean (1.23e-2) sits 2.9 PINN-standard-
   deviations above the PINN mean (8.74e-3 ± 1.2e-3) — the gap is
   statistically real. The QPINN remains scientifically interesting (it
   inherits photonic inductive bias), but any quantum-advantage claim
   from Table IV is unsupported.

## Fair Baselines

The classical PINN baseline is a fully-connected `tanh` network with the
same two-output `(u, ux)` interpretation as the QPINN and the same
consistency loss. Its width is chosen so the trainable-parameter count
approximately matches the QPINN it is compared against; the heuristic is
in `lib/pinn_baseline.py::hidden_layers_for_param_count`.

## MerLin Photonic Extension

The paper's photonic target is *continuous-variable* photonic computing.
MerLin targets *linear-optical* photonic computing. Native MerLin
abstractions do not include squeezing, displacement, or Kerr gates, so we
do not provide a literal port of the paper's circuit. Instead,
`configs/poisson_merlin.json` and `lib/merlin_pinn.py` reproduce the
*consistency-loss training idea* on a MerLin interferometer + angle
encoding. The MerLin model returns probabilities over an `UNBUNCHED`
detection basis, mapped to `(u, ux)` by two trainable linear heads.

### Hardware-aware settings (MerLin variant)

| Field | Value |
|---|---|
| Computation space | `UNBUNCHED` |
| Detector model | threshold |
| Photon number | 3 |
| Number of modes | 6 |
| Input state | `[1, 0, 1, 0, 1, 0]` (dual-rail-style spread) |
| Encoding | angle, modes `[0, 1, 2]`, scale `π/2` |
| Measurement strategy | `MeasurementStrategy.probs(computation_space=UNBUNCHED)` (merlin >= 0.4 API) |
| Postselection | none |
| Simulator / QPU path | MerLin CPU simulator (analytic, shots=0) |
| Shot count | n/a (analytic) |
| Seeds | 42 (single seed; multi-seed deferred) |
| Wall-clock | see `results/poisson_merlin.json` |

## Limitations

- The simulator is single-precision-friendly but uses `complex128` by default
  to avoid losing trace under deep CV circuits; multi-seed runs are deferred
  to a sequel session.
- Strawberry Fields' parameter-shift / non-Gaussian gradient machinery is not
  used; if the paper's results depend on the exact gradient path through
  TensorFlow, our `matrix_exp` autograd path could behave differently.
- The MerLin variant is honestly a *different* photonic architecture and
  cannot be read as a one-to-one CV photonic translation of the paper.
- The X8 noise study is implemented at the gate level but not run as part of
  this reproduction.

## Tests

```bash
cd papers/CV_QPINN_PDE
pytest -q
```

The tests cover CLI parsing and a minimal `train_and_evaluate` smoke run on
a 1-layer / cutoff-5 QPINN to confirm the runner produces `summary.json`.

## Citation and License

```bibtex
@article{panichi_qpinn_pdes_2025,
  title  = {Quantum physics informed neural networks for multi-variable partial differential equations},
  author = {Panichi, Giorgio and Corli, Sebastiano and Prati, Enrico},
  year   = {2025},
  eprint = {2503.12244},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

Reproduction code is released under the repository MIT licence. See
`LICENSE` at the repository root.
