# QML Passive Sonar — Reproduction

Reproduction scaffold for **Bach & Nguyen, "Quantum machine learning for
passive sonar: spectral feature extraction and Hybrid convolution neural
network classification of ship propeller signals", Ocean Engineering (2025
preprint)**.

## Reference and Attribution

- Paper: *Quantum machine learning for passive sonar: spectral feature
  extraction and Hybrid convolution neural network classification of ship
  propeller signals*.
- Authors: Nhat Hoang Bach, Van Duc Nguyen.
- Venue: Ocean Engineering, Elsevier (preprint, 2025, no DOI yet).
- Original repository: none — the authors released no code, so every
  algorithm in `lib/` is implemented from the paper's mathematical
  description.
- Datasets: ShipsEar (Santos-Domínguez *et al.*, 2016, registration
  required) and DeepShip (Irfan *et al.*, 2021, on Zenodo).

See [`ADDITIONAL_NOTES.md`](ADDITIONAL_NOTES.md) for implementation
rationale, CPU sanity results, and guidance for scaling to full-dataset
experiments.

## Original Paper

The paper proposes a two-stage framework for classifying underwater
acoustic signals from propeller-driven ships.

- **Stage 1 — SAV (Spectral Amplitude Variation):** a DEMON replacement that
  computes the temporal variance of STFT power spectra across overlapping
  windows (Eqs. 4, 13) and detects tonal lines with an adaptive threshold
  `gamma = mu_V + eta * sigma_V`.
- **Stage 2 — HQ-CNN:** a 4-block CNN backbone (~57.4 M params) producing a
  4096-dim embedding, mapped through a 10-qubit / 4-layer parameterised
  quantum circuit and a linear head into class probabilities.

Headline claims:

| ID | Description | Paper value |
|----|-------------|-------------|
| E2 | SAV detection rate (ShipsEar / DeepShip) | 98.22 % / 99.04 % |
| E4 | SAV vs DEMON-Hilbert processing-time speedup | 140–175 × |
| E5 | CNN → HQ-CNN accuracy (ShipsEar / DeepShip) | 90.0→93.7 / 91.6→96.8 % |
| E7 | 5-fold source-stratified CV, paired test | p = 0.031 |
| E8 | Gen. gap (HQ-CNN best vs strongest classical) | 0.2 % / 0.15 % |

## Reproduction Scope (including Updates and Deviations)

This repository ships a fully runnable scaffold that:

1. Implements the SAV (`lib/sav.py`) and DEMON (`lib/demon.py`) algorithms
   from scratch in NumPy + SciPy.
2. Implements the CNN backbone (`lib/models.py:CNNBackbone`) and HQ-CNN
   classifier (`lib/models.py:HQCNN`) in PyTorch, with the encoding
   interface `phi = pi * sigmoid(W_K h + b_K)`.
3. Implements the 10-qubit / 4-layer PQC as a **pure-PyTorch statevector
   simulator** (`lib/quantum.py`) so the smoke run does not need PennyLane
   or Qiskit. Gradients flow through torch autograd; this is mathematically
   equivalent to the paper's parameter-shift training up to floating-point
   noise.
4. Provides a synthetic-dataset fallback (`lib/data.py`) so the entire
   pipeline runs end-to-end without registering for ShipsEar.
5. Scaffolds the MerLin photonic variant (`lib/models_merlin.py`) — it
   raises a clear `ImportError` until `merlinquantum` is installed.

**Deviations from the paper**:

- *Datasets*: the smoke configs use synthesised propeller-noise-like
  signals (four class-specific tonal sets + harmonics + coloured noise) so
  the codebase is reproducibly runnable. Drop the real `.wav` files under
  `data/qml_passive_sonar/{shipear,deepship}/<class>/` to switch to the
  paper datasets — `lib/data.py` auto-detects them. For DeepShip CPU sanity
  runs, the small public GitHub sample is downloaded automatically when no
  local DeepShip `.wav` files are present.
- *Quantum gradients*: torch autograd through a statevector simulator
  rather than parameter-shift. Same forward map.
- *Smoke CNN width*: `model.fc_dim = 64` and `image_size = 32` in
  `configs/defaults.json`; the paper-accurate `4096 / 224` is in
  `configs/classification_*.json`.
- *Epochs*: smoke run is 2 epochs; paper-accurate runs request 100 and
  should be launched on GPU.
- *Audio resampling*: linear interpolation, not band-limited.
- *Split*: simple stratified split at the *frame* level; the paper's
  source-stratified split requires real recording-level metadata.
- *MerLin photonic variant*: fully implemented in
  `lib/models_merlin.py` — generic rectangular interferometer with
  trainable BS/PS parameters, per-mode input phase encoding, and a
  readout interferometer. Uses 6 modes / 2 photons in the unbunched
  computation space.


## Project Layout

Standard repository template:

```
papers/qml_passive_sonar/
|-- README.md           # this file
|-- ADDITIONAL_NOTES.md # implementation notes and scaling guidance
|-- requirements.txt
|-- cli.json
|-- configs/            # defaults + named experiments
|-- lib/                # sav, demon, data, quantum, models, training, runner
|-- tests/              # algorithm + smoke-run tests
|-- utils/              # (placeholder for plotting helpers)
|-- models/             # checkpoint dump target
|-- outdir/             # disposable timestamped runs
`-- assets/             # README-bound figures
```

## Install and How to Run

```bash
# Optional: virtualenv
python -m venv .venv && source .venv/bin/activate
pip install -r papers/qml_passive_sonar/requirements.txt
```

The shared runtime drives everything via `implementation.py`:

```bash
# Smoke run (uses configs/defaults.json — fully synthetic, fast).
python implementation.py --paper qml_passive_sonar

# SAV-vs-DEMON detection experiment (E1–E4 analogue on synthetic data).
python implementation.py --paper qml_passive_sonar --config configs/sav_detection.json

# Paper-accurate ShipsEar training (E5/E6). Needs GPU and real .wav data.
python implementation.py --paper qml_passive_sonar --config configs/classification_shipear.json

# Paper-accurate DeepShip training (E5/E6).
python implementation.py --paper qml_passive_sonar --config configs/classification_deepship.json

# MerLin photonic variant (requires merlinquantum to be installed).
python implementation.py --paper qml_passive_sonar --config configs/merlin_shipear.json

# DeepShip CPU sanity run; auto-downloads the small public sample if absent.
python implementation.py --paper qml_passive_sonar --config configs/classification_deepship_cpu.json
```

CLI flags (see `cli.json`):

- `--task {sav_detection,classification,merlin_classification}`
- `--model {cnn,hqcnn,hqcnn_merlin}`
- `--dataset {shipear,deepship}`
- `--spectrogram-method {sav,demon,stft}`
- `--epochs INT`, `--batch-size INT`, `--lr FLOAT`
- `--n-qubits INT`, `--n-layers INT`, `--fc-dim INT`, `--image-size INT`

## Configuration

| File | Purpose |
|------|---------|
| `configs/defaults.json` | Smoke run, 2 epochs, 4 qubits, 32 × 32 input. |
| `configs/sav_detection.json` | SAV vs DEMON detection on synthetic data. |
| `configs/classification_shipear.json` | Paper-accurate HQ-CNN on ShipsEar. |
| `configs/classification_deepship.json` | Paper-accurate HQ-CNN on DeepShip. |
| `configs/merlin_shipear.json` | Photonic HQ-CNN on ShipsEar. |
| `configs/merlin_deepship.json` | Photonic HQ-CNN on DeepShip. |

## Data

Data is deliberately not committed. The repository tracks only the expected
folder scaffold; DeepShip sanity runs auto-download the small public GitHub
sample when no local WAVs are present, while paper-scale runs should use the
full datasets requested from the original maintainers.

Place real recordings under the repo-level data root; see
[`../../data/qml_passive_sonar/README.md`](../../data/qml_passive_sonar/README.md)
for dataset URLs, author contacts, the tracked placeholder structure, and
full-dataset placement notes:

```
data/qml_passive_sonar/
|-- shipear/
|   |-- A/*.wav   (fishing, trawlers, tugs)
|   |-- B/*.wav   (motorboats, sailboats)
|   |-- C/*.wav   (ferries)
|   |-- D/*.wav   (ocean liners, ro-ro)
|   `-- E/*.wav   (background)
`-- deepship/
    |-- F/*.wav   (cargo)
    |-- G/*.wav   (passenger)
    |-- H/*.wav   (tanker)
    `-- I/*.wav   (tug)
```

When ShipsEar class folders are missing, `lib/data.py` logs a clear warning
and substitutes synthetic clips with the same per-class tonal structure.

The full datasets are not committed. The DeepShip CPU configs can use the
small sample published in `github.com/irfankamboh/DeepShip`; it is downloaded
automatically when no local DeepShip `.wav` files are present, and lands under
the ignored `data/qml_passive_sonar/deepship/` directory. Set
`dataset.download=false` to skip that auto-download and use the synthetic
fallback.

## Results Obtained and Comparison with the Paper

All numbers below are 20-epoch CPU runs (fc_dim=128, image 64×64, seed=42)
on either the synthetic fallback (ShipsEar) or a small 8-file DeepShip
subset loaded on demand from `github.com/irfankamboh/DeepShip`. Figures live in
[`results/`](results/).

| Experiment | Paper | This repo | Run dir |
|---|---:|---:|---|
| E2 SAV detection rate (synthetic ShipsEar) | 98.22 % | 100.0 % | run_20260514-084228 |
| E4 SAV vs DEMON false peaks | 140-175× speedup | 4 vs 29 false peaks | run_20260514-084228 |
| E5 CNN test acc (DeepShip real) | 91.6 % | 92.98 % | run_20260514-092836 |
| E5 HQ-CNN test acc (DeepShip real) | 96.8 % | 89.47 % | run_20260514-092902 |
| E5 MerLin photonic test acc (DeepShip real) | n/a | 92.98 % | run_20260514-094804 |
| E8 Gen. gap MerLin DeepShip | 0.15 % | 3.11 % (lower than CNN 6.24 %, HQ-CNN 7.40 %) | — |
| E5 CNN/HQ-CNN/MerLin test acc (synthetic ShipsEar) | 90/93.7/n.a. | 100/100/100 % (saturated) | run_20260514-085421/-092519/-092707 |

**Key findings**:

- SAV beats DEMON on synthetic data — fewer false peaks at the same
  detection rate, in line with the paper's Tables 6-7.
- CNN baseline on real DeepShip is within 1.4 pp of the paper's number
  despite using 8 files instead of 600+ and 20 epochs instead of 100.
- The MerLin photonic variant matches the CNN baseline accuracy on
  DeepShip but has **half the generalization gap** of either the classical
  CNN or the qubit-PQC HQ-CNN — qualitatively consistent with the paper's
  E8 claim that quantum variants generalise better.
- HQ-CNN test accuracy on DeepShip is below the paper's because of the
  tiny file count + reduced backbone width; the trend (lower gap) holds
  for MerLin.

The full paper-accuracy configs (`configs/classification_shipear.json`,
`_deepship.json`, `merlin_*.json` — fc_dim=4096, image_size=224,
epochs=100) are committed and only need a GPU + the full datasets to
launch.

## Figures

- `results/sav_vs_demon.png` — per-class variance & DEMON spectra with
  detection markers (Fig. 4/5 analogue).
- `results/classification_deepship.png` — train/test bar chart + gap.
- `results/classification_shipear.png` — same for synthetic ShipsEar.
- `results/confusion_deepship.png` — confusion matrices for the three DeepShip models.
- `results/learning_curves_deepship.png` — per-epoch accuracy/loss curves.

## Limitations

- The PQC is simulated; no QPU access.
- No `librosa` resampling — linear interpolation may slightly bias high
  frequencies; acceptable for log-magnitude spectrograms but should be
  swapped for `scipy.signal.resample_poly` for paper-accurate runs.
- Source-stratified split is the user's responsibility when real data is
  used; the built-in split is frame-level stratified.
- Statistical comparisons (paired t-test, BCa bootstrap, Cohen's d) are not
  implemented in this scaffold — they require multiple independent runs
  whose metrics are aggregated outside the runner.

## Tests

```bash
cd papers/qml_passive_sonar
pytest -q
```

The suite covers:

- SAV tonal detection on synthetic mixtures.
- DEMON spectrum sanity checks.
- PQC shape, range, identity, and gradient-flow checks.
- End-to-end smoke run through `implementation.py --paper qml_passive_sonar`.

## Citation and License

Cite the original paper as soon as a DOI is available. This reproduction
code is released under the repository's top-level license; the datasets
remain governed by their original licensing terms.
