# QCNN-ID: A Quantum-Classical Hybrid Model for IoT Intrusion Detection — Reproduction

## Reference and Attribution

> Marwen Amara, Sami Mnasri, Thierry Val.
> **QCNN-ID: A Quantum-Classical Hybrid Model for IoT Intrusion Detection.**
> 29th KES International Conference on Knowledge Based and Intelligent
> Information and Engineering Systems, Osaka (Japan), Sep 2025.
> HAL: [hal-05080861](https://hal.science/hal-05080861v1).

Released implementation: <https://github.com/MarwenPhd/QCNN_ID_on_MIot_dataset>

## Original Paper

The paper proposes a hybrid quantum-classical CNN for binary intrusion
detection on a healthcare IoT network-traffic dataset:

1. Z-score normalise the raw flow features.
2. Reduce to 8 principal components via PCA.
3. Scale each component to `[0, π]`.
4. Angle-encode the 8 components into 8 qubits with `Ry` (Eq. 4).
5. Apply a parameterised variational ansatz `A(θ)` of rotations + CNOTs
   (Eq. 5).
6. Measure `<Z_i>` on each qubit (Eq. 6) and feed the expectation vector
   into a classical binary head.

The classical baseline in the paper text is a `Linear(input) → 128 → 64 →
output` MLP with ReLU + dropout(0.3) between layers.

Headline claims:
- ~99% accuracy on both QCNN and CNN (Fig. 3, Sec. 4.1).
- Fewer trainable parameters on the QCNN side.
- Substantially fewer false negatives for the QCNN (22 vs 465 from Fig. 4).
- Higher QCNN wall-clock cost due to simulator overhead (Fig. 3(e)).

## Reproduction Scope, Claims, and Deviations

### What is reproduced

- The full preprocessing pipeline (`StandardScaler → PCA(8) → min-max to
  [0, π]`) as described in Sec. 3.1.
- The CNN classical baseline as described in Sec. 3.3
  (`lib/models.py::CNNClassifier`).
- The paper-text-described 8-qubit gate-model hybrid
  (`lib/models.py::QCNNClassifier`), implemented in PennyLane with
  Qiskit-style ZZFeatureMap and RealAmplitudes circuit structure,
  following the provided author source code.
- A MerLin photonic extension (`lib/models.py::PhotonicClassifier`):
  8 modes / 4 photons, UNBUNCHED computation space, angle encoding,
  threshold detection, no postselection.


### Deviations

- **Current comparison uses a shared PCA feature view.** The runner feeds
  `CNNClassifier`, `QCNNClassifier`, and `PhotonicClassifier` the same
  `n_components` PCA-reduced split. This keeps the model comparison
  feature-aligned.
- **Gate-model implementation.** `QCNNClassifier` is implemented in
  PennyLane with a Qiskit-style `ZZFeatureMap` + `RealAmplitudes` circuit
  structure.
- **Photonic implementation is an adaptation.** The paper is gate-model;
  the MerLin branch uses a photonic reservoir to compute quantum features.
  It uses 8 optical modes, 4 photons, fixed random interferometers,
  angle-encoded phase shifts, and a trainable linear head on the
  Fock-probability vector.
- **Binary classification head.** The local implementation uses one output
  logit for each model: `logit(x) = W·E(x)+B`, optimised with
  `BCEWithLogitsLoss`. Reported class labels use the usual fixed rule
  `sigmoid(logit) >= 0.5`.
- **Compute tier.** The curated numbers below are CPU runs generated
  locally with the pinned reproduction environment. They include smoke/short
  runs, a 10k-row all-model comparison, and a 20k-row 3-seed
  CNN-vs-MerLin run. The original `intrusion_original.json` QCNN-vs-CNN
  20k-row, 20-epoch, 3-seed config
  remains RED-tier on CPU: the 10k-row QCNN comparison already took
  about 24 minutes for one seed.
- **Claim C3 is tested on sampled runs, not the full paper split.** The
  local 10k-row all-model run does not reproduce the paper's 22-vs-465
  false-negative claim: QCNN has 35 false negatives versus 4 for CNN.


## Project Layout

```
papers/QCNN_ID/
├── README.md             ← this file
├── cli.json
├── requirements.txt
├── notebook.ipynb        ← pedagogical walkthrough
├── configs/
│   ├── defaults.json                       ← smoke config (4 k rows, 3 epochs, all families)
│   ├── current_short_1000_e10.json         ← legacy quick notebook-sized run
│   ├── current_notebook_1000_e40.json      ← current notebook-sized run (~3 min QCNN)
│   ├── intrusion_original.json             ← paper-text PQC vs CNNClassifier (3 seeds)
│   ├── intrusion_merlin.json               ← Photonic vs CNNClassifier (3 seeds)
│   ├── intrusion_photonic_convergence_10000_e40.json ← photonic convergence diagnostic
│   └── intrusion_full_compare.json         ← all 3 model families (1 seed)
├── lib/
│   ├── data.py        ← 3-CSV loader + StandardScaler + PCA + π-scaling
│   ├── models.py      ← CNNClassifier, QCNNClassifier, PhotonicClassifier
│   ├── training.py    ← shared train/eval loop, per-epoch metrics
│   └── runner.py      ← train_and_evaluate(cfg, run_dir) — shared-runtime entry
├── tests/             ← model + data smoke tests
├── outdir/            ← per-run timestamped outputs
└── utils/             ← analysis/plotting helpers
```

## Install and How to Run

```bash
pip install -r papers/QCNN_ID/requirements.txt
```

Then, from the repository root, use the shared runtime:

```bash
# Smoke run (4k rows, 3 epochs, CNNClassifier + PhotonicClassifier + QCNNClassifier)
python implementation.py --paper QCNN_ID

# Curated notebook-sized comparison (1000 rows, 40 epochs, all 3 models)
python implementation.py --paper QCNN_ID --config configs/current_notebook_1000_e40.json

# Multi-seed: paper-text-described hybrid vs classical baseline (20 k rows, 20 epochs, 3 seeds)
python implementation.py --paper QCNN_ID --config configs/intrusion_original.json

# MerLin photonic adaptation
python implementation.py --paper QCNN_ID --config configs/intrusion_merlin.json

# Side-by-side comparison of all three model families (single seed, 15 epochs)
python implementation.py --paper QCNN_ID --config configs/intrusion_full_compare.json

# Full-dataset run (RED tier for the quantum variant on CPU)
python implementation.py --paper QCNN_ID --config configs/intrusion_original.json --subset_size 0
```

Each run writes a timestamped directory under `papers/QCNN_ID/outdir/`
containing:

- `config_snapshot.json`
- `run.log`
- `metrics.json`         (summary + per-epoch + per-seed)
- `training_curves.png`  (loss/accuracy/precision/recall/ROC-AUC/time)
- `confusion_matrices.png`
- `train_predictions.csv`
- `test_predictions.csv`

CLI overrides are supported:

```bash
python implementation.py --paper QCNN_ID --config configs/intrusion_original.json \
  --epochs 5 --subset_size 5000 --seed 7
```

## Configuration

See `cli.json` for the full schema. Main keys:

| Key | Meaning | Default |
|---|---|---|
| `data_dir` | Directory under `data_root` containing the three ICU CSVs | `QCNN_ID` |
| `data_csv` | Legacy single-CSV path (overrides `data_dir` when set) | — |
| `subset_size` | Rows to sample after concatenation (0 = use all 188 k) | `4000` (smoke) |
| `test_size` | Train/test split fraction | `0.3` |
| `n_components` | PCA components (= number of qubits/modes) | `8` |
| `n_qubits` | Qubits in the gate-model ansatz | `8` |
| `ansatz_reps` | Repetitions of the variational ansatz | `1`–`2` |
| `lr_qcnn` | Adam learning rate for the gate-model QCNN | `2e-2` for `intrusion_full_compare` |
| `lr_photonic` | Adam learning rate for the MerLin photonic head | `5e-2` |
| `models` | Subset of `cnn_classifier`, `qcnn_classifier`, `photonic_classifier` | varies per config |
| `seeds` | List of integer seeds to average over | `[42]`–`[42, 7, 123]` |

## Data

**Source.** The IoT Healthcare Security Dataset
(<https://github.com/imfaisalmalik/IoT-Healthcare-Security-Dataset>),
distributed as `Dataset/ICUDatasetProcessed.zip`. The Kaggle mirror
(<https://www.kaggle.com/datasets/faisalmalik/iot-healthcare-security-dataset>)
is the same data but requires a Kaggle login.

**Expected layout.** Unzip the archive and place the three CSVs under
the repository's shared data root:

```
data/QCNN_ID/
├── Attack.csv                  ← 80 126 rows, label = 1
├── environmentMonitoring.csv   ← 31 758 rows, label = 0
└── patientMonitoring.csv       ← 76 810 rows, label = 0
```

The data loader uses the shared runtime's `data_root` (defaults to
`<repo>/data/`); override with `--data-root` or the `DATA_DIR`
environment variable if your files live elsewhere.

**Format.** Each CSV has the same 52 columns: 50 raw features (TCP, MQTT,
IP, frame timing), one textual `class` column (dropped in preprocessing),
and one binary `label` column.

## Results Obtained and Comparison with the Paper

Latest curated artifacts were generated on CPU using a Python environment
made from `requirements.txt`, with the MerLin / Perceval stack installed.
Stable copies are stored under `results/`; timestamped raw runs remain
under `outdir/`. The current code writes both train/test prediction CSVs
for downstream analysis.

Curated artifacts:

- [latest_summary.json](results/latest_summary.json)
- [results/smoke_defaults_4000_e3/](results/smoke_defaults_4000_e3/)
- [results/current_short_1000_e10/](results/current_short_1000_e10/) legacy quick run
- [results/current_notebook_1000_e40/](results/current_notebook_1000_e40/)
- [results/intrusion_full_compare_10000_e15/](results/intrusion_full_compare_10000_e15/)
- [results/intrusion_photonic_convergence_10000_e40/](results/intrusion_photonic_convergence_10000_e40/)
- [results/intrusion_merlin_20000_e20_s3/](results/intrusion_merlin_20000_e20_s3/)

Each stable run folder contains `metrics.json`, figures, `run.log`,
`train_predictions.csv`, and `test_predictions.csv`. Prediction CSVs contain
one row per model/sample with `y_true`, `y_pred`, `logit`, and `probability`.
The bottom-right panel of `training_curves.png` is the ROC curve when the
run performs binary classification. Smoke runs keep the full ROC view; curated
experiment configs zoom to FPR `[0, 0.2]` and TPR `[0.8, 1.0]`.

### Run inventory

| Stable result folder | Source run | Rows | Epochs | Seeds | Models | Role |
|---|---|---:|---:|---:|---|---|
| `smoke_defaults_4000_e3` | `outdir/run_20260625-111813` | 4,000 | 3 | 1 | CNN, Photonic, QCNN | compatibility smoke |
| `current_short_1000_e10` | `outdir/run_20260625-115855` | 1,000 | 10 | 1 | CNN, QCNN, Photonic | legacy quick demo |
| `current_notebook_1000_e40` | `outdir/run_20260625-151031` | 1,000 | 40 | 1 | CNN, Photonic, QCNN | current notebook run |
| `intrusion_full_compare_10000_e15` | `outdir/run_20260625-134908` | 10,000 | 15 | 1 | CNN, Photonic, QCNN | main all-model comparison |
| `intrusion_photonic_convergence_10000_e40` | `outdir/run_20260625-151010` | 10,000 | 40 | 1 | Photonic | photonic convergence diagnostic |
| `intrusion_merlin_20000_e20_s3` | `outdir/run_20260625-112007` | 20,000 | 20 | 3 | CNN, Photonic | MerLin multi-seed check |

`intrusion_original.json` is still available but was not launched in this
pass. Extrapolating from the 10k-row QCNN run (`1428 s` for one seed, 15
epochs) makes the 20k-row, 20-epoch, 3-seed QCNN config a several-hour CPU
job rather than a routine documentation refresh.

### Main all-model comparison

Run: `results/intrusion_full_compare_10000_e15/` (`subset_size=10000`,
`test_size=0.3`, 7,000 train rows, 3,000 test rows, PCA-8,
`ansatz_reps=2`, `seed=42`, `lr_qcnn=0.02`).

| Model | Accuracy | Macro prec | Macro rec | ROC-AUC | Trainable params | Train time (s) | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CNNClassifier` (PCA-8 MLP) | **0.9983** | **0.9985** | **0.9981** | **1.0000** | 9,473 | **1.30** | **4** |
| `PhotonicClassifier` (8 modes / 4 photons) | 0.9853 | 0.9876 | 0.9826 | 0.9993 | 71 | 3.46 | 44 |
| `QCNNClassifier` (8 qubits, reps=2) | 0.9733 | 0.9722 | 0.9732 | 0.9972 | **33** | 1427.61 | 35 |

Confusion matrices use rows=true labels `[benign, attack]` and
columns=predictions `[benign, attack]`:

| Model | Confusion matrix |
|---|---:|
| `CNNClassifier` | `[[1736, 1], [4, 1259]]` |
| `PhotonicClassifier` | `[[1737, 0], [44, 1219]]` |
| `QCNNClassifier` | `[[1692, 45], [35, 1228]]` |

The photonic model remains above the gate-model QCNN in accuracy, while the
QCNN now has slightly fewer false negatives than the photonic adaptation.

### Photonic convergence diagnostic

Run: `results/intrusion_photonic_convergence_10000_e40/` (`subset_size=10000`,
photonic only, 40 epochs, same split scale as the all-model comparison).

| Epoch | Accuracy | Macro rec | Test loss |
|---:|---:|---:|---:|
| 1 | 0.7913 | 0.7522 | 0.6284 |
| 6 | 0.9680 | 0.9620 | 0.3149 |
| 8 | 0.9730 | 0.9679 | 0.2608 |
| 13 | 0.9853 | 0.9826 | 0.1793 |
| 18 | **0.9857** | **0.9830** | 0.1340 |
| 40 | **0.9857** | **0.9830** | 0.0631 |

Interpretation: predictions stabilize by about epoch 18, while BCE loss
continues improving afterward. Large accuracy steps can still happen because
many samples cross the fixed logit decision boundary (`logit = 0`,
equivalently `sigmoid(logit) = 0.5`) together, but no threshold sweep or
calibration is used in the reported metrics.

### MerLin multi-seed check

Run: `results/intrusion_merlin_20000_e20_s3/` (`subset_size=20000`,
14,000 train rows, 6,000 test rows per seed, seeds `[42, 7, 123]`).
The confusion matrices below are aggregated across the three seed-specific
6,000-row test splits.

| Model | Accuracy mean ± std | Macro prec | Macro rec | ROC-AUC | Trainable params | Mean train time (s) | Aggregated false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CNNClassifier` | **0.9988 ± 0.0004** | **0.9988** | **0.9988** | **0.9999** | 9,473 | 4.80 | **12** |
| `PhotonicClassifier` | 0.9887 ± 0.0034 | 0.9903 | 0.9867 | 0.9996 | 71 | 6.92 | 201 |

Aggregated confusion matrices:

| Model | Confusion matrix |
|---|---:|
| `CNNClassifier` | `[[10377, 9], [12, 7602]]` |
| `PhotonicClassifier` | `[[10383, 3], [201, 7413]]` |

The photonic model remains conservative: it produces very few false positives
(3 across 10,386 benign test examples) and more false negatives than the CNN.
This is now reported strictly at the fixed sigmoid 0.5 operating point.

### Notebook-sized demo

Run: `results/current_notebook_1000_e40/` (`subset_size=1000`, 700 train rows,
300 test rows, PCA-8, `ansatz_reps=1`, `seed=42`). This replaces the old
e10 notebook run for discussion; the previous `current_short_1000_e10` folder
is kept only as a legacy quick artifact.

| Model | Accuracy | Macro prec | Macro rec | ROC-AUC | Trainable params | Train time (s) | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CNNClassifier` | **0.9967** | **0.9972** | **0.9960** | **0.9998** | 9,473 | **0.89** | **1** |
| `PhotonicClassifier` | 0.9800 | 0.9835 | 0.9758 | 0.9962 | 71 | 0.66 | 6 |
| `QCNNClassifier` | 0.9367 | 0.9422 | 0.9282 | 0.9813 | **25** | 198.10 | 15 |

Confusion matrices:

| Model | Confusion matrix |
|---|---:|
| `CNNClassifier` | `[[176, 0], [1, 123]]` |
| `PhotonicClassifier` | `[[176, 0], [6, 118]]` |
| `QCNNClassifier` | `[[172, 4], [15, 109]]` |

### Claim-by-claim outcome

| Claim | Paper statement | Local outcome | Notes |
|---|---|---|---|
| C1 | QCNN and CNN converge to comparable ~99% accuracy | **Not reproduced for QCNN on sampled local runs.** | CNN reaches 0.9983 on the 10k all-model run; QCNN reaches 0.9733. In the longer notebook-sized e40 run, QCNN reaches 0.9367 while CNN reaches 0.9967. |
| C2 | QCNN uses fewer trainable parameters than CNN | **Confirmed structurally.** | QCNN has 25 params at `ansatz_reps=1` and 33 at `ansatz_reps=2`, versus 9,473 for CNN. |
| C3 | QCNN has fewer false negatives than CNN | **Not reproduced on sampled local runs.** | 10k all-model run: CNN 4 FN, QCNN 35 FN. Notebook e40: CNN 1 FN, QCNN 15 FN. |
| C4 | QCNN has higher wall-clock cost | **Confirmed.** | 10k all-model run: QCNN 1427.61 s versus CNN 1.30 s. Notebook e40: QCNN 198.10 s versus CNN 0.89 s. |
| Text consistency | 8 PCA components / 8 qubits vs "four input features" in conclusion | **Paper text is internally inconsistent.** | The implementation follows the 8-component pipeline described in Sec. 3.1. |
| n/a | MerLin photonic adaptation | **Adaptation, not a paper reproduction.** | Photonic reaches 0.9853 in the 10k all-model e15 run, 0.9857 after 40 photonic-only epochs, and 0.9887 ± 0.0034 over three 20k sampled runs. It is more accurate and much faster than the local gate-model QCNN, while remaining far smaller than the CNN. |

Current conclusion: with the local implementation and sampled CPU runs, the
classical PCA-MLP (`CNNClassifier`) remains the strongest reproduced baseline:
it is the most accurate model and has the fewest false negatives. The gate-model
QCNN keeps the paper's low-parameter property but it does not reproduce the paper's
headline accuracy or false-negative claims under these configs.

**MerLin model result.** The MerLin photonic adaptation is exploratory, not a
claim from the original gate-model paper, but it is the strongest quantum-side
local baseline in these artifacts. On the 10k all-model run, it reaches 0.9853
accuracy in 3.46 s, above the gate-model QCNN's 0.9733 accuracy and far
faster than its 1427.61 s training time; It is about halfway between
the QCNN and the CNN score of 0.9983.
On the 20k/3-seed run, it reaches 0.9887 ± 0.0034 accuracy with only 3 false
positives across 10,386 benign test examples, although it still has 201 false
negatives versus 12 for the CNN. With 71 trainable parameters, MerLin is larger
than the QCNN (33 params) but far smaller than the CNN (9,473 params).

## Fair Baselines

Two baselines are run alongside the paper-text QCNN:

1. **CNN classical** (`CNNClassifier`): the same `Linear(d) → 128 → 64`
   encoder the paper itself describes, followed by one binary logit. It uses
   the same split and PCA-reduced feature view as the quantum branches.
2. **MerLin photonic** (`PhotonicClassifier`): the same PCA-8 inputs
   pushed through a photonic angle-encoded circuit; threshold detection;
   trainable binary `nn.Linear` head on the 70-bin probability vector.

All three models train on identical preprocessed splits with the same
optimiser family (Adam, fixed `lr_cnn`, `lr_qcnn`, and `lr_photonic`). Parameter
counts, accuracy, precision, recall, ROC-AUC, and training time are reported
side-by-side in `metrics.json` and on the comparison figures.

## MerLin Photonic Extension

| Field | Value |
|---|---|
| Computation space | UNBUNCHED |
| Detector model | threshold |
| Photon number | 4 |
| Number of modes | 8 |
| Input state | `[1, 0, 1, 0, 1, 0, 1, 0]` |
| Encoding | angle, one phase per mode; data already scaled to `[0, π]` |
| Measurement strategy | `MeasurementStrategy.PROBABILITIES` |
| Postselection | none |
| Simulator / QPU | MerLin CPU simulator, analytic (`shots = 0`) |
| Output size | `C(8, 4) = 70` |
| Trainable head | `nn.Linear(70, 1)`, i.e. `W·E(x)+B` |

The MerLin variant is intentionally an *adaptation*, not a strict
reproduction: the paper is gate-model, while the photonic implementation is
ours. Its role in this reproduction is therefore comparative: it tests whether
a hardware-aware photonic feature map can provide a stronger low-parameter
quantum-side baseline than the local gate-model QCNN.

## Limitations

- Reduced `subset_size` for speed in the multi-seed configs; the full
  188 k-row run is RED-tier on CPU for the quantum variant.
- The paper-text-described ansatz is underspecified; we used the provided
  implementation by the authors.
- Per-class precision/recall is reported as the macro-average; the paper
  does not specify per-class vs macro.

## Tests

```bash
cd papers/QCNN_ID
pytest -q -s
```

Tests cover: forward shapes and gradient flow for every model family,
the 3-CSV data pipeline (synthetic ICU layout), and the legacy
single-CSV compatibility shim. `-s` is required to disable pytest's
output capture, which interacts poorly with this container's `/tmp`
isolation.

## Citation and License

Original paper:

```bibtex
@inproceedings{amara2025qcnnid,
  title     = {QCNN-ID: A Quantum-Classical Hybrid Model for IoT Intrusion Detection},
  author    = {Amara, Marwen and Amara, Marwa and Mnasri, Sami and Val, Thierry},
  booktitle = {29th KES International Conference},
  year      = {2025},
  address   = {Osaka, Japan},
  note      = {HAL: hal-05080861}
}
```

This reproduction is released under the same license as the parent
repository (see `LICENSE` at the repo root).
