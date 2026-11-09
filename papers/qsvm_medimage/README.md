# Quantum Kernel Advantage over Classical Collapse — CPU reproduction audit and photonic adaptation

This directory provides a minimal CPU-oriented adaptation of:

- the paper [*Quantum Kernel Advantage over Classical Collapse in Medical Foundation Model Embeddings*](https://arxiv.org/abs/2604.24597v1);
- the original repository [`sebasmos/qml-medimage`](https://github.com/sebasmos/qml-medimage);
- the imported upstream revision [`9e800373`](https://github.com/sebasmos/qml-medimage/tree/9e80037305d683b0e70c94b8fa7dd648e1bac82b).

It also contains a separate photonic-kernel adaptation implemented with MerLin 0.4.

> **Scope:** the experiments use raw PneumoniaMNIST pixels as an open-data
> surrogate. They do not reproduce the paper's controlled-access MIMIC-CXR
> insurance-classification task or its frozen foundation-model embeddings.

## Paper in brief

The paper studies binary insurance classification from chest radiographs:

- `Private insurance` is the minority class;
- `Medicaid / Medicare` is the majority class.

Each image is first converted into a frozen embedding using MedSigLIP-448,
RAD-DINO, or a general-purpose ViT. PCA compresses the embedding to `q`
features.

The compared classifiers receive the same PCA features:

- a linear SVM;
- a validation-tuned RBF SVM;
- a QSVM using the paper's BSP qubit feature map and a compute-uncompute
  fidelity kernel.

The primary metric is minority-class F1. It becomes zero when a classifier
never detects the minority class, even if its overall accuracy remains
apparently reasonable.

The paper reports:

- 18/18 QSVM wins against the untuned linear SVM;
- 7/7 wins against the tuned RBF SVM;
- majority-class collapse of the linear SVM on most embedding seeds.

## What this repository evaluates

The controlled-access reference reproduction was not run.

Instead, this directory provides:

- a CPU execution path derived from the imported QSVM implementation;
- linear and tuned-RBF classical baselines;
- an open-data PneumoniaMNIST surrogate;
- a four-protocol audit of preprocessing and kernel normalization;
- a separate MerLin photonic fidelity-kernel adaptation.

The local experiment uses:

- at most 500 images per seed;
- 400 training, 50 validation, and 50 test images;
- PCA dimensions `q=4` and `q=6`;
- ten deterministic subset/split seeds, from 0 to 9;
- a fixed MerLin circuit seed equal to 0.

These local seeds control subsampling and splitting. They are not equivalent
to the paper's seed-specific foundation-model embeddings.

## Paper protocol versus imported upstream code

The audit focuses on two differences.

### MinMax preprocessing

The imported preprocessing fits `StandardScaler` and PCA on training data.

By default, however, its final `MinMaxScaler` is fitted on training plus the
current held-out split:

- [`data_prepare_cv`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/process.py).

This allows validation or test extrema to influence the representation used
for training. The local audit therefore compares:

- the preserved `train + held-out` behavior;
- a `train only` variant.

### Trace normalization

The paper states that the square training Gram matrix and its associated
validation/test cross-kernel must be divided by the same training-kernel
trace; see Section III.C and Equation (3) of the
[paper](https://arxiv.org/html/2604.24597v1#S3.SS3).

In the imported implementation:

- [`normalize_kernel_trace`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/core.py)
  divides square matrices by their trace but returns rectangular matrices
  unchanged;
- [`apply_hybrid_kernel`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py)
  applies this function separately to the square training matrix and the
  rectangular held-out cross-kernel;
- the same path is used for [validation and test](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L832-L833).

The local audit therefore compares:

- the preserved `square only` behavior;
- a `train trace` variant that scales both matrices with the same training
  trace.

No conclusion about author intent is drawn from these implementation details.

## MerLin photonic adaptation

MerLin is absent from both the paper and the imported repository.

The local MerLin path:

1. applies the same local PCA preprocessing;
2. encodes each PCA vector into a photonic quantum state;
3. computes a fidelity kernel;
4. passes the resulting precomputed kernel to scikit-learn's SVM.

The overlaps are evaluated exactly on CPU with `shots=None`. The circuit is a
native MerLin photonic feature map, not a translation or resource match of the
paper's BSP qubit circuit.

Two local MerLin variants are evaluated:

- an unnormalized fidelity kernel;
- a train-trace-normalized fidelity kernel.

These are local analysis choices, not upstream MerLin behaviors.

## Installation

From this paper directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment pins MerLin 0.4.0 and a compatible scikit-learn version.

## Data

The local experiments use PneumoniaMNIST.

Prepare the shared dataset copy with:

```bash
python scripts/prepare_pneumoniamnist.py \
  --download_path ../../data/qsvm_medimage/pneumoniamnist.npz \
  --output_path ../../data/qsvm_medimage/pneumoniamnist_train.pkl
```

Each 28×28 image is flattened to 784.

The local classes are:

- `normal`, used as the minority class;
- `pneumonia`, used as the majority class.

The raw pixels occupy the upstream `embedding` field for compatibility, but
they are not foundation-model embeddings.

## Running the catalogue entry

From this paper directory:

```bash
python ../../implementation.py --list-papers
python ../../implementation.py --paper qsvm_medimage --help
python ../../implementation.py --paper qsvm_medimage
```

The default catalogue configuration runs a small CPU MerLin experiment.

## Running the complete local protocol matrix

From this paper directory:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=../../data/qsvm_medimage/pneumoniamnist_train.pkl \
Q_VALUES=4,6 \
SEEDS=0,1,2,3,4,5,6,7,8,9 \
MAX_SAMPLES=500 \
CIRCUIT_SEED=0 \
LEAKAGE_MODES=legacy,train_only \
TRACE_MODES=legacy_square_only,train_trace \
RESULT_ROOT=outdir/qsvm-protocol-matrix \
bash scripts/run_table1_adaptation.sh
```

The launcher evaluates four protocol IDs:

- `legacy_leak__legacy_trace`;
- `train_only__legacy_trace`;
- `legacy_leak__train_trace`;
- `train_only__train_trace`.

For the QSVM, `legacy_trace` means square-only trace scaling.

For MerLin, the same launcher label maps to no kernel normalization. It is
only a shared protocol identifier and does not imply that MerLin inherits
behavior from the imported repository.

The QSVM and linear SVM use `C=1`.

The RBF baseline selects `C` from:

```text
0.01, 0.1, 1, 10, 100
```

using validation minority-class F1. The test set is not used to choose `C`.

## Results and analysis

All numerical results, paired comparisons, W/T/L counts, protocol-sensitivity
tables, and interpretation are kept in
[`notebook.ipynb`](notebook.ipynb).

The notebook reads the committed result artifacts and does not recompute
kernels or retrain models.

Committed artifacts:

- [per-seed protocol results](results/protocol_matrix_n500_q4_q6/protocol_results_per_seed.csv);
- [aggregated protocol summary](results/protocol_matrix_n500_q4_q6/protocol_summary.csv);
- [generated Markdown summary](results/protocol_matrix_n500_q4_q6/protocol_summary.md).

The upstream-protocol-only results are the `legacy_leak__legacy_trace` rows of
the per-seed file.

### Curating generated protocol results

`run_table1_adaptation.sh` writes raw runs and the three aggregated protocol
artifacts under `RESULT_ROOT`, which defaults to a directory under `outdir/`.
It does not copy generated files into the committed `results/` directory.

After a completed run using `RESULT_ROOT=outdir/qsvm-protocol-matrix`, curate
the aggregate artifacts with:

```bash
bash scripts/curate_protocol_results.sh
```

The script checks that all three aggregate files exist, copies them to
`results/protocol_matrix_n500_q4_q6/`, and verifies each copy against its
source. It does not run experiments, calculate kernels, or regenerate the
aggregates.

If the raw protocol directories exist but the aggregate files do not, first
run only the aggregator, then curate its output:

```bash
.venv/bin/python scripts/aggregate_protocol_matrix.py \
  --result_root outdir/qsvm-protocol-matrix
bash scripts/curate_protocol_results.sh
```

Alternative locations can be selected explicitly:

```bash
SOURCE_ROOT=outdir/another-run \
CURATED_ROOT=results/another-protocol-matrix \
bash scripts/curate_protocol_results.sh
```

The script copies results only. When curating a new run, record its provenance
(sample sizes, splits, seeds and protocols) in this README.

## Limitations

- The reference MIMIC-CXR reproduction was not run.
- Pneumonia detection is different from insurance classification.
- Raw pixels are different from frozen medical foundation-model embeddings.
- The local matrix evaluates only `q=4` and `q=6`.
- Each seed contains only 50 test images.
- Local seeds control subsets and splits, not embedding generation.
- No local paired significance test was performed.
- The linear-collapse behavior reported by the paper is not reproduced by
  this surrogate.
- The MerLin circuit is not an implementation or resource match of the BSP
  qubit circuit.
- The MerLin feature map, circuit seed, photon/mode resources, and SVM `C`
  were not optimized.
- Local protocol findings do not establish whether the original MIMIC-CXR
  results are valid.
- The paper itself is limited by noiseless simulation, SVM-only classical
  baselines, the proxy task, and post-hoc selection of the DT9 data stratum.

## Citation and license

```bibtex
@article{cajas2026qml,
  title   = {Quantum Kernel Advantage over Classical Collapse in Medical
             Foundation Model Embeddings},
  author  = {Cajas Ord\'{o}\~{n}ez, Sebasti\'{a}n Andr\'{e}s and
             Ocampo Osorio, Felipe and
             Koh, Dax Enshan and Al Attrach, Rafi and Marzullo, Aldo and
             Guerra-Adames, Ariel and Andrade, J. Alejandro and Goh, Siong Thye
             and Chen, Chi-Yu and Gorijavolu, Rahul and Yang, Xue and
             Hebdon, Noah Dane and Celi, Leo Anthony},
  journal = {arXiv preprint arXiv:2604.24597},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.24597}
}
```

The imported code is attributed above. Its CC BY-NC-SA 4.0 terms are
preserved in [`LICENSE`](LICENSE).

PneumoniaMNIST is distributed through
[MedMNIST v2](https://medmnist.com/). The dataset is CC BY 4.0 and the
MedMNIST code is Apache-2.0.
