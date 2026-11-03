# Reproduction details

Reference material for the [README](../README.md): the choices made where the paper is silent, how the circuits are simulated, the names used in configs and results, the command-line flags, and how to rerun and analyse the sweeps.

## Deviations and choices

| Item | Paper | Choice here | Why |
|---|---|---|---|
| Data window | 35,064 hourly values, dates not given | 2020-10-22 … 2024-10-21 (first 24 hourly columns of each daily file) | Matches the first and last rows of Table II and the reported length exactly; enforced by a SHA-256 fingerprint |
| Normalization | Min-max "in each dataset" | Min-max fitted on train | Avoids using eval/test statistics; the per-split reading (`--normalization per_split`) leaves the headline unchanged |
| Ansatz angles | "Trainable w" | Fixed, uniform in [0, 2π), seeded | Reservoir computing trains only the readout |
| Observable | "Pauli Z" | Z⊗4 | Chosen on the evaluation split: eval RMSE 0.0215 vs 0.0764 for Z on one qubit |
| Wiring and update | k_i ∈ {0, 1, 2} recurrent inputs per neuron | Uniform random sources, synchronous update, 3 passes, no state carried between time steps | Simplest reading of Figs. 1 and A1; sequential update is tied within 1 % on the evaluation split |
| Quantum kernel | Eq. 6 written K(R, Y) | K(r(t), r(t')) between reservoir outputs | Kernel ridge needs a kernel between inputs; read literally, K(R, Y) would compare reservoir outputs with targets. The feature map is a product state, so K is evaluated in closed form |
| λ | Not given | 10-value log grid, n-normalized, selected on eval RMSE | Same protocol for every readout |
| Distributed merge | Not given | Contiguous feature partition, mean of the instance predictions | Summing would double the prediction, and giving the full r(t) to every instance would make the linear SRMR cells identical to SRSR, which the paper's tables contradict |
| Classical reservoir | ReservoirPy, no hyperparameters | Library defaults (paper baseline) | Most plausible reading of an unreported configuration; an ESN tuned on the evaluation split is added as an additional baseline |
| Noise construction | Not given | Depolarizing channels after native gates and a symmetric readout flip, from the App. B medians; exact expectations, no shots | Uses only the numbers the paper reports (the paper's Tables B1 and B2) |
| Best-cell selection | Best of 8 cells, chosen on test | Reported with the paper's rule and with selection on the evaluation split | The paper's rule uses the test set for selection |

## Simulation engine

The paper simulates its circuits with Qiskit and, for an unstated part, NVIDIA CUDA-Q. We simulate the same circuits exactly with a small numpy implementation instead:

- **Quantum neuron** (`lib/quantum_neuron.py`): the part of the circuit after the encoding, its noise and the observable do not depend on the data. They are folded once per neuron into a 16 × 16 matrix, so each input window costs one small tensor contraction.
- **Quantum kernel** (`lib/kernels.py`): the feature map has no entangling gate, so the fidelity is a product of single-qubit overlaps (our derivation), computed from one Bloch vector per qubit and input.

Both match Qiskit to 1e-10 or better, ideal (Statevector) and noisy (Aer density matrix): see `tests/test_quantum_neuron.py`, `tests/test_kernels.py` and Section 2 of the notebook. We chose this for three reasons: the paper's setup cannot be copied as described (it does not say which simulations use CUDA-Q, nor give versions or simulator settings), CUDA-Q has no macOS ARM build, and the neuron is the bulk of the cost. A timing on the ideal circuits (Apple M5 CPU, Qiskit 2.5.2, single run; `results/checks/simulation_benchmark.txt`):

| Circuit | Qiskit | Ours |
|---|---|---|
| Quantum neuron, one 60-neuron reservoir (60 neurons × 3 passes × 35,064 windows) | 30 min (`StatevectorEstimator`, one batched job) | 8 s |
| Quantum kernel, Gram matrix on the full training split (24,539 points, 20 features on 10 qubits) | 38 s (one statevector per point, then one matrix product) | 8 s |

The gain is large for the neuron (about 230×; a reproduction run evaluates many reservoirs, noisy ones included) and small for the kernel, where a well-written Qiskit implementation is also fast. Noisy Qiskit simulation was not timed. To rerun the timing, from the repository root:

```bash
python papers/distributed_qrc/utils/benchmark_simulation.py
```

## Cell names

A cell is `<reservoir type>|<readout type>`; these names are used in configs, `metrics.json` and `results/`. For example, `classical|linear` is the paper baseline.

| Reservoir type | In the paper? | Meaning |
|---|---|---|
| `classical` | yes | ReservoirPy reservoir with library defaults (paper baseline) |
| `quantum_ideal` | yes | Network of 4-qubit quantum neurons, exact ideal simulation |
| `quantum_noisy` | yes | Same circuits with the hardware-informed noise model (App. B, Table III mapping) |
| `photonic` | no (MerLin adaptation) | One frozen 4-mode, 2-photon chip per neuron |
| `esn_tuned` | no (tuned-ESN baseline) | The `classical` reservoir with spectral radius, leak rate and input scaling tuned on the evaluation split (12 settings) |
| `raw_lags` | no (no-reservoir control) | No reservoir; the readout sees the 4-lag window directly |

| Readout type | In the paper? | Meaning |
|---|---|---|
| `linear` | yes (classical ridge, Eq. 5) | Linear ridge; an intercept is added by default (`--linear-intercept false` for the paper's Eq. 5) |
| `quantum_kernel_ideal`, `quantum_kernel_noisy` | yes ("quantum ridge") | Kernel ridge with the paper's fidelity kernel, ideal or noisy |
| `photonic_kernel` | no (MerLin adaptation) | Kernel ridge with a 2-photon fidelity kernel |
| `rbf_kernel` | no (classical-kernel and no-reservoir controls) | Kernel ridge with an RBF kernel (γ and λ tuned on the evaluation split) |
| `persistence` | no (no-reservoir control) | The forecast is the last observed hour (only with `raw_lags`) |

## Command-line flags

Paper-specific flags (`cli.json` is the authoritative list):

| Flag | Effect | Default |
|---|---|---|
| `--observable {z_all,z0}` | Quantum neuron observable | `z_all` |
| `--update {synchronous,sequential}` | Reservoir propagation order | `synchronous` |
| `--normalization {train,per_split}` | Min-max fit | `train` |
| `--artifact-handling {keep,interpolate}` | Zero/negative loads in the series | `keep` |
| `--merge {mean,sum}` | Merge rule of distributed readouts | `mean` |
| `--noise-mapping {paper,ibm_brisbane,ibm_marrakesh,ionq_aria1}` | Device per component: Table III, or one device for all | `paper` |
| `--noise-scale X` | Multiplies every App. B error rate | `1.0` |
| `--reservoir-shots N`, `--kernel-shots N` | Finite-shot estimation | exact |
| `--linear-intercept {true,false}` | Intercept of the linear ridge | `true` |
| `--kernel-fit-subsample N` | Training points of the final kernel fit | full split |
| `--neuron-scale X`, `--kernel-scale X` | Photonic encoding scales (MerLin) | `1.0`, `2.0` |

The shared runtime adds `--config`, `--seed`, `--device`, `--outdir`, `--log-level` and `--data-root`.

## Sweeps

A sweep runs one config for several seeds (here 0, 1, 2) and, when needed, several candidate settings, then collects every run and a `summary.json` in `outdir/sweep_<ID>/`. All results of the reproduction come from sweeps. The IDs are the names the runs were recorded under, so they are kept as is. An ID reads `<architecture>-<variant>`:

- **Architecture:** `E1` = SRSR, `E2` = MRSR, `E3` = SRMR, `E4` = MRMR (the same numbers as the configs `e1_srsr_…` to `e4_mrmr_…`).
- **Variant:** `R2` = reproduction of the paper (second run, fully planned before launch; the first pass was superseded), `CLASSICAL` = fair classical baselines, `MERLIN` = MerLin photonic adaptation.
- **Other sweeps:** `B1B3-RAW` (the no-reservoir control, shared by all architectures; config `no_reservoir_classical.json`), `C6-SRSR` / `C6-MRSR` / `C6-MRMR` (noise side check), and the diagnostics listed in `results/results_summary.md`.

From the repository root:

```bash
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-R2 --config configs/e1_srsr_original.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-CLASSICAL --config configs/e1_srsr_classical.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-MERLIN --config configs/e1_srsr_merlin.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id B1B3-RAW --config configs/no_reservoir_classical.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id C6-SRSR --config configs/noise_check_srsr.json --seeds 0 1 2 \
  --candidates '[{"noise_mapping": "paper", "noise_scale": 10}, {"noise_mapping": "ibm_brisbane", "noise_scale": 10}]'
```

## Analysis

Each script reads the sweeps' raw `metrics.json` files and writes to `results/` (from the repository root, with `O=papers/distributed_qrc/outdir` and `U=papers/distributed_qrc/utils`):

| Supports | Script | Output |
|---|---|---|
| Message 1: the numbers reproduce | `python $U/compare_with_paper.py $O/sweep_E1-R2` (one sweep per architecture) | `results/data/<arch>_vs_paper.csv` |
| | `python $U/headline_statistics.py $O/sweep_E{1,2,3,4}-R2` | `results/data/headline_statistics.json` |
| | `python $U/plot_paper_items.py $O data/distributed_qrc/sources/arxiv_source_v1/tqe_tran.tex` | `results/paper_items/` |
| Message 2: the gain is not quantum | `python $U/fair_comparison.py $O`, then `fair_comparison_table.py` and `plot_fair_comparison.py $O` | `results/data/fair_comparison.json`, `results/fair_comparison_table.md`, `results/fair_comparison.png` |
| Message 3: the architecture holds | `python $U/merlin_comparison.py $O` | `results/data/merlin_comparison.json` |
| Claims C3–C5 and the gaps in *Limitations* | `cross_architecture_table.py`, `robustness_verdicts.py`, `shot_gap_analysis.py`, `noise_distribution_comparison.py`, `classical_seed_spread.py`, `diebold_mariano.py` | `results/data/` |
| Checks | `verify_sweeps.py`, `verify_merlin_sweeps.py`, `benchmark_simulation.py`, `merlin_fidelitykernel_repro.py` | `results/checks/` |
| Summary | `python $U/documentation_tables.py $O`; `build_notebook.py`, then `run_notebook.py` | `results/results_summary.md`, `notebook.ipynb` |

Each script's docstring gives its exact arguments.

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with:

```text
config_snapshot.json   # resolved configuration
run.log                # lifecycle, readout fits, evaluations
run_status.json        # status, code commit and hash, dataset fingerprint
metrics.json           # every cell: eval/test MAE, RMSE, R², test MAE/RMSE in MW
test_predictions.npz   # test predictions of every cell and the targets
```
