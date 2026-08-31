# Quantum vs. Classical Time-Series Benchmark — Reproduction

This project reproduces the central result of Fellner, Kreplin, Tovey, and Holm,
*Quantum vs. classical: A comprehensive benchmark study for predicting time
series with variational quantum machine learning*.

The paper compares five variational quantum models with three classical models
on chaotic time-series forecasting. Its central result is reproduced: the
classical models rank substantially better overall. A MerLin study included in
this project also evaluates a photonic dressed QNN and three photonic reservoir
models.

## Reference and Attribution

- Authors: Andreas Fellner, Christian Kreplin, Daniel Tovey, and Christian Holm
- Journal: *Machine Learning: Science and Technology* 7, 010501 (2026)
- Preprint: [arXiv:2504.12416v2](https://arxiv.org/abs/2504.12416)
- DOI: [10.1088/2632-2153/ae365f](https://doi.org/10.1088/2632-2153/ae365f)
- Original code: [VariationalQMLTimeSeriesBenchmark](https://github.com/tobias-fllnr/VariationalQMLTimeSeriesBenchmark)
- Released data and results: [DaRUS 5559](https://doi.org/10.18419/DARUS-5559)

## Main Results

The result that matters for the paper is the ranking computed from its released
grid-search results over all 27 forecasting tasks:

| Rank | Model | Mean rank | Type |
|---:|:---|---:|:---|
| 1 | LSTM | 1.78 | classical |
| 2 | RNN | 2.70 | classical |
| 3 | le-QLSTM | 2.85 | quantum |
| 4 | MLP | 4.22 | classical |
| 5 | d-QNN | 4.70 | quantum |
| 6 | ru-QNN | 5.33 | quantum |
| 7 | QLSTM | 7.04 | quantum |
| 8 | QRNN | 7.37 | quantum |

The mean rank is **2.90 for the classical models** and **5.46 for the quantum
models**. The ordering matches the paper's Figure 5 exactly.

![Model ranking across all 27 tasks](results/ranking_all_models.png)

The independent live implementation gives the same qualitative result. The
400-epoch runs are useful as implementation checks, but not as the primary
scientific comparison: model ordering changes when validation-plateau training
is used.

| Six-task mean rank | 400 epochs | Validation plateau, cap 3000 |
|:---|---:|---:|
| LSTM | 2.67 | **1.50** |
| photonic-dQNN | **1.83** | 2.67 |
| MLP | 2.17 | 3.00 |
| RNN | 3.67 | 3.50 |

The apparent photonic lead at 400 epochs is therefore a training-budget effect.
Under the paper's stopping rule, LSTM ranks first.

![Effect of training budget on ranking](results/budget_effect.png)

### Photonic reservoir result

The reservoir study uses 8 optical modes, 4 photons, encoding scale `37.6991`
(`12π`), leak `0.3`, and 3 memristors. Only the linear readout is trained.

The final six-task comparison against the classical baselines is:

| Model | Mean rank |
|:---|---:|
| LSTM | **2.50** |
| photonic-seqRC | **2.50** |
| photonic-memRC | 2.83 |
| RNN | 4.17 |
| MLP | 4.33 |
| photonic-RC | 4.67 |

The per-task median test MSE values are:

| Model | Hénon k=1 | Hénon k=4 | Mackey k=1 | Mackey k=140 | Lorenz k=1 | Lorenz k=25 |
|:---|---:|---:|---:|---:|---:|---:|
| photonic-RC | 1.681e-3 | 3.214e-2 | 8.787e-5 | 3.089e-2 | 1.482e-4 | 2.214e-2 |
| photonic-seqRC | 1.324e-6 | 3.009e-3 | 1.692e-4 | 2.716e-2 | 2.649e-6 | 1.376e-2 |
| photonic-memRC | 2.851e-6 | 3.801e-3 | 1.055e-4 | 2.517e-2 | 3.397e-6 | 1.407e-2 |

The sequential reservoir and the memristive reservoir are close. Their
geometric-mean test-MSE ratio, `seqRC/memRC`, is `0.89`. This does not isolate a
benefit from the memristor: sequential processing and readout capacity explain
the robust improvement over the static reservoir.

The reservoir result is an extension of the paper, not evidence of quantum
advantage. It covers six tasks, three evaluation seeds, and one selected optical
configuration.

![Tuned reservoir comparison](results/reservoir_search.png)

## Original Paper

The benchmark contains 27 tasks:

- Dynamical systems: Mackey–Glass, Hénon, and Lorenz.
- Forecast horizons:
  - Mackey–Glass: `1`, `70`, `140`
  - Hénon: `1`, `2`, `4`
  - Lorenz: `1`, `13`, `25`
- Input sequence lengths: `4`, `8`, `16`.

The paper trains each model with Adam, learning rate `1e-3`, batch size `64`,
and MSE loss. Training stops on a validation plateau. Each grid-search
configuration uses ten seeds, and the reported value is the median test MSE of
the best-validation model.

The compared model families are:

- Quantum: d-QNN, ru-QNN, QRNN, QLSTM, and le-QLSTM.
- Classical: MLP, RNN, and LSTM.

## Reproduction Scope and Deviations

This project contains three complementary evaluations:

1. Exact aggregation of the authors' released grid-search CSVs across all 27
   tasks.
2. An independent live implementation of all eight model families on Hénon
   `k=1` and `k=4`, using one representative configuration and three seeds.
3. A MerLin evaluation of a photonic dressed QNN and three frozen photonic
   reservoirs on six tasks.

The live experiments use three seeds instead of ten and do not repeat the
paper's complete hyperparameter grid. Gate-model simulation uses PennyLane
`default.qubit`, analytic execution, and backpropagation. The photonic models
use analytic MerLin simulation.

Data preprocessing follows the released pipeline. Its scaling uses information
outside the training partition, so absolute MSE values are optimistic. This
affects all compared models and does not alter the rank-based headline result.

## Project Layout

| Path | Purpose |
|:---|:---|
| `cli.json` | Paper-specific command-line schema |
| `configs/defaults.json` | Lightweight runnable configuration |
| `lib/models.py` | Five quantum and three classical model families |
| `lib/photonic.py` | Variational photonic dressed QNN |
| `lib/reservoir.py` | Static, sequential, and memristive photonic reservoirs |
| `lib/data.py` | Dataset loading, scaling, and time-window construction |
| `lib/trainer.py` | Training and validation-plateau stopping |
| `lib/runner.py` | Shared-runtime entry point |
| `utils/sweep.py` | Reduced live model sweep |
| `utils/run_reservoirs.py` | Six-task, two-budget comparison |
| `utils/search_reservoirs.py` | Reservoir selection and final evaluation |
| `utils/report_search.py` | Reservoir tables and figure |
| `utils/compare_all_models.py` | Combined comparison and coverage report |
| `original_results/` | Authors' released result CSVs |
| `results/` | Curated metrics, tables, models, and figures |

## Installation and Quick Start

From the repository root:

```bash
python -m pip install -r papers/variational_qml_ts_benchmark/requirements.txt

python implementation.py --paper variational_qml_ts_benchmark \
    --config configs/defaults.json
```

To regenerate the main result tables and figures without training:

```bash
cd papers/variational_qml_ts_benchmark
python utils/plot_paper_figures.py
python utils/report_search.py
python utils/compare_all_models.py
```

## Running Individual Models

All models use the repository-level runtime. For example:

```bash
# Classical LSTM
python implementation.py --paper variational_qml_ts_benchmark \
    --model lstm --ansatz layers_1 --hidden-size 16 \
    --dataset henon_1000 --sequence-length 4 --prediction-step 4 \
    --epochs 400

# Gate-based QRNN
python implementation.py --paper variational_qml_ts_benchmark \
    --model qrnn --ansatz paper_no_reset --num-qubits 4 --hidden-size 2 \
    --dataset henon_1000 --sequence-length 4 --prediction-step 1 \
    --epochs 400

# Photonic dressed QNN
python implementation.py --paper variational_qml_ts_benchmark \
    --model photonic --ansatz photonic --num-qubits 6 --hidden-size 3 \
    --dataset lorenz_1000 --sequence-length 4 --prediction-step 25 \
    --epochs 3000 --use-convergence 1

# Photonic memristive reservoir
python implementation.py --paper variational_qml_ts_benchmark \
    --model photonic_memristor \
    --ansatz reservoir_scale37.6991_leak0.30_mem3 \
    --num-qubits 8 --hidden-size 4 \
    --dataset mackey_1000 --sequence-length 4 --prediction-step 140 \
    --epochs 10000 --use-convergence 1
```

For photonic models, `--num-qubits` denotes optical modes and `--hidden-size`
denotes photons. The authoritative command-line schema is `cli.json`.

## Reproducing the Experiment Suites

Run these commands from `papers/variational_qml_ts_benchmark`. Sweep scripts
resume by skipping output directories that already contain `metrics.json`. Do
not run two copies of the same sweep concurrently.

### Paper ranking

```bash
python utils/plot_paper_figures.py
```

### Reduced live sweep

```bash
python utils/sweep.py --epochs 400 --seeds 0 1 2 --workers 8
python utils/run_extras.py
python utils/plot_sweep.py
```

### Six-task model comparison

```bash
python utils/run_reservoirs.py --seeds 0 1 2 --arms fixed conv \
    --max-epochs 3000 --workers 8
python utils/plot_reservoirs.py --arm auto
```

This evaluates seven models on six tasks with three seeds and two training
budgets, for 252 runs.

### Reservoir selection and evaluation

```bash
python utils/search_reservoirs.py --stage 1 12 2 3 4 \
    --tune-seeds 0 1 --eval-seeds 0 1 2 --epochs 400 \
    --conv-epochs 3000 --workers 8

python utils/search_reservoirs.py --stage 5 --eval-seeds 0 1 2 \
    --previous-conv-cap 3000 --conv-epochs 10000 --workers 8

python utils/report_search.py
```

The search uses validation MSE for selection. It evaluates optical geometry,
encoding scale, and memristor dynamics on the two Hénon tasks, then evaluates
the selected configuration on all six tasks. The final result combines every
cell that satisfied the plateau rule with the corresponding 10000-epoch result
for cells requiring the larger budget.

## Configuration and Outputs

`configs/defaults.json` is a five-epoch MLP smoke configuration. Named configs
provide Hénon runs for LSTM, d-QNN, le-QLSTM, and the photonic model.

Each shared-runtime execution writes:

```text
outdir/run_YYYYMMDD-HHMMSS/
|-- config_snapshot.json
|-- run.log
|-- metrics.json
|-- losses.csv
|-- loss_curve.png
`-- best_validation_model.pt
```

The main curated outputs are:

- `results/claim_summary.md`: complete 27-task paper ranking.
- `results/sweep_table.md`: reduced live experiment.
- `results/reservoir_table.md`: six-task, two-budget comparison.
- `results/reservoir_search.md`: selected reservoir comparison.
- `results/all_models_comparison.md`: combined results and coverage matrix.

## Data

The datasets are stored under:

```text
data/variational_qml_ts_benchmark/
```

Each system contains 1000 time points. Sliding windows are divided by index into
60% training, 20% validation, and 20% test samples.

## Limitations

- The exact paper comparison is derived from the authors' released results;
  independently trained models cover a smaller experimental matrix.
- The live gate-based quantum models cover two of 27 tasks. The photonic models
  cover six of 27 tasks.
- Live runs use three seeds and one representative model configuration rather
  than ten seeds and a complete hyperparameter grid.
- Thirteen tuned reservoir cells reached the 10000-epoch limit: static RC 7/15,
  memristive RC 3/13, and sequential RC 3/14. Their MSE values remain upper
  bounds under the stopping protocol.
- The reservoir comparison demonstrates competitiveness on the tested subset,
  not quantum advantage or general superiority across time-series problems.

## Tests

Run the tests from the paper directory:

```bash
cd papers/variational_qml_ts_benchmark
pytest -q
```

The 19 tests cover configuration, data shapes, model construction, forward
passes, smoke training, reservoir behavior, convergence-cell selection, and
report aggregation.

## Citation and License

Please cite the original paper and dataset when using this reproduction. The
original implementation remains subject to its repository license. This
reproduction follows the license of the `reproduced_papers` repository.
