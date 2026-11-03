# Error-mitigated photonic quantum circuit Born machine

## Reference and Attribution

Salavrakos, Sedrakyan, Mills, Mansfield and Mezher (2025),
[Error-mitigated photonic quantum circuit Born machine](https://doi.org/10.1103/PhysRevA.111.L030401),
Physical Review A **111**, L030401. [arXiv](https://arxiv.org/abs/2405.02277).

Original code: [Quandela/photonic-qcbm](https://github.com/Quandela/photonic-qcbm/tree/d593d209128e403644b9aee595c474ce551391ae).

Gaussian targets and SPSA follow that reference. MerLin and Perceval
are Quandela dependencies; retained SDK notices are in the [sampling notes](lib/SAMPLING.md#dependency-notices).

## Overview

This project trains a photonic circuit Born machine to fit a target distribution
and compares photon recycling `photon_recycling()` with lossy post-selection at equal initial-shot
budgets. MerLin performs the optical/noise calculations and native sampling;
Perceval defines the circuit, noise model and detectors. A custom classical SPSA
optimizer updates the phases, following the author implementation.

Changes from the paper:

- MerLin replaces the Perceval simulation backend. The circuit has 12 modes,
  four photons, one Clements block and 132 nominal phase parameters. Input Fock state is
  `[0,0,1,0,1,0,1,0,1,0,0,0]`.
- ECB 2004–2023 EUR/USD data replace the unavailable original financial dataset.
- Belenos-characteristic local simulations replace the Altair hardware experiment,
  with additional overlap controls. No QPU experiments are included.

The [notebook](notebook.ipynb) displays the actual circuit, compares native samples
with theory, applies recycling and runs 100 SPSA updates for three conditions
across all five seeds (0-4). Its six-mode/three-photon CPU example uses
transmission 0.5 and 6,392,000 initial shots in total; allow a few minutes.
This educational setting was selected after exploratory two-photon and
three-photon comparisons, not as a paper benchmark. Every paired seed is shown.
Historical Figure 2 results are loaded separately from curated files.

## How to Run

### Installation

Validated  with Python 3.13.16, MerLin 0.4.2, Perceval 1.3.1 and PyTorch 2.12.1. 

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r papers/error-qcbm/requirements.txt
```

Open [notebook.ipynb](notebook.ipynb) in your notebook editor using this environment.
The notebook and validated backend use CPU float64. 
The pinned dependencies include the headless notebook tools.

### Command line

```bash
# Run a small offline environment/gradient check, without training.
python implementation.py --paper error-qcbm

# List available options.
python implementation.py --paper error-qcbm --help

# Train three toy conditions for two updates at 2,000 initial shots per call.
python implementation.py --paper error-qcbm --config configs/toy_smoke.json

# Run the notebook's longer five-seed educational comparison.
python implementation.py --paper error-qcbm --config configs/toy_comparison.json
```

From inside `papers/error-qcbm/`, use `../../implementation.py`. In a flat
checkout, omit `papers/` from paths and use `../implementation.py` from the
folder. Full experiment settings are listed below; sampling and continuation
details are in [SAMPLING.md](lib/SAMPLING.md) and [TRAINING.md](lib/TRAINING.md).

### Source modules

The reproduction uses 15 Python modules, including the package initializer.
Related data preparation, optimization and validation functions share modules.

| Modules | Responsibility |
|---|---|
| `runner.py` | Shared CLI entry point and offline environment checks |
| `backend.py`, `noise.py`, `theory.py` | Optical model, noise profiles and independent physical references |
| `data.py` | Gaussian/financial targets and explicit source acquisition/preparation |
| `sampling.py`, `sampler_validation.py` | Native count accumulation and its environment-specific verification |
| `recycling.py` | Input validation and adaptation to official Perceval photon recycling |
| `spsa.py` | Distribution losses and the classical SPSA update |
| `training.py`, `resumable.py` | Finite-shot objectives, campaigns and checkpoint continuation |
| `artifacts.py` | Atomic JSON/file writes and objective-shot accounting |
| `results.py` | Curated results, summaries and figure rendering |
| `__init__.py` | Package initializer |
Full CPU training can take many hours per seed, particularly at 50M shots per call.

For headless notebook execution, run these commands inside the folder:

```bash
python -m ipykernel install --user --name error-qcbm --display-name "Python 3 (error-qcbm)"
python -m nbconvert --to notebook --execute notebook.ipynb --output notebook.executed --output-dir outdir/notebook_validation --ExecutePreprocessor.kernel_name=error-qcbm --ExecutePreprocessor.timeout=600
```

## Configuration

[cli.json](cli.json) defines options; JSON files in `configs/` select operations.

| Configuration | Operation |
|---|---|
| `defaults.json` / `smoke.json` | Small offline environment check |
| `toy_smoke.json` | Fast four-mode/two-photon smoke, three conditions, two updates |
| `toy_comparison.json` | Educational six-mode/three-photon T=0.5 comparison, 100 updates, all five seeds 0..4; notebook default |
| `gaussian.json` / `gaussian_smoke.json` | Gaussian full experiment / reduced run |
| `fx.json` / `fx_smoke.json` | ECB financial full experiment / reduced run; frozen CSV required |
| `belenos.json` | Four conditions using Belenos-characteristic and lossless models |
| `belenos_smoke.json` | Seed-zero, 20-update smoke at 50M shots per call; expensive |
| `belenos_profile_smoke.json` | Small offline native sampling with the bundled noise profile |
| `belenos_loss_only.json` / `belenos_overlap084.json` / `lossless_overlap084.json` | Additional loss/overlap controls |
| `belenos_vectorized.json` | Optimized Belenos run; full environment-matched sampler evidence required |
| `sampler_validation_smoke.json` / `sampler_validation_full.json` | Small sampler check / 50M equivalence audit |
| `prepare_gaussian.json` / `prepare_fx.json` / `prepare_fx_download.json` | Target generation / frozen CSV processing / explicit official acquisition |
| `plot_results.json` | Redraw curated figures without training or new photon shots |

Runs write configuration, logs and artifacts to `outdir/run_YYYYMMDD-HHMMSS/`.
Notebook outputs go to ignored `outdir/notebook_*/`; final images and compact
redraw data are in `results/`. Resumable experiments keep update-boundary state
in `training.state_dir`. Compatible runs resume; completed runs are skipped.

Configs overlay `configs/defaults.json`; explicit CLI flags take precedence.
`--data-root` names the shared parent, with `error-qcbm/` appended.
`--outdir` names the output base. The global `--seed` does not replace the
explicit `training.seeds` list or sampling/perturbation offsets.

## Data

The Gaussian target is the author's known 495-bin two-Gaussian probability
vector, generated locally rather than estimated from training samples:

```bash
python implementation.py --paper error-qcbm --config configs/prepare_gaussian.json
```

The financial adaptation uses [European Central Bank (ECB) daily USD-per-EUR rates](https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2004-01-01&endPeriod=2023-12-31&format=csvdata)
from 2004–2023. Log returns and the empirical histogram are calculated here.
Financial figures carry this source and processing notice. Preparation and
training metadata record the same source credit, [ECB terms URL](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html) and transformation notice.
Place the hash-matched frozen CSV under external `data/error-qcbm/fx/` before
full training; the CSV is not bundled. Offline preparation and explicit download
use `prepare_fx.json` and `prepare_fx_download.json`. Live acquisition timed out
during review; a changed response is not substituted automatically.

The bundled [Belenos profile](configs/noise/belenos_20261017.json) contains only
selected fields from a read-only Quandela Hub query: transmission T=.028 and
overlap input I=.729, with brightness=1, g2=0 and threshold detection. It requires
no Hub account or original response. I=.84 is an additional model-input ablation from the paper,
not a measured Belenos value. See [DATA_SOURCES.md](DATA_SOURCES.md) for sources,
hashes, acquisition instructions, transformations and modeling limits.

## Results and Analysis

The Gaussian and financial campaigns each completed 20 seeds across three
conditions, with 1,000 SPSA updates per run. Gaussian uses KL with 200k initial
shots per call; financial uses TVD with 60k. Belenos-characteristic experiments
use 50M per call. The Belenos family has 143 attempts, including three failed
originals and three separately recorded sampling-stream retries.

| Original paired comparison | Pairs | Recycling wins: final sampled loss |
|---|---:|---:|
| Gaussian | 20 | 20/20 |
| ECB financial adaptation | 20 | 20/20 |
| Belenos I=.729 | 19 | 19/19 |

These comparisons use the recorded finite-shot training objectives. They do
not establish universal mitigation improvement or advantage over classical
generative models.

![Figure 2 reproduction: Gaussian panels a–b, ECB financial panels c–d, and Belenos-characteristic simulation panels e–f](results/figure2_combined.png)

**Figure 2.** The six reproduction panels are arranged in three rows and two
columns: Gaussian (a,b), ECB financial adaptation (c,d), and Belenos-characteristic
local simulation (e,f). The combined view uses the same recorded values and
selection rules as the individual panel-pair exports.
All curves show finite-shot training losses. Bands in (a,c) are population
standard deviations across 20 original runs, not confidence intervals. Panel (b)
shows five runs per condition selected by lowest final sampled KL, separately
for lossless and recycling; these are not matched-best seeds. Panel (e) shows
three instances with different initializations, unlike the paper's hardware
repeats. Panel (f) uses seed 6 for every curve, selected by final sampled KL from
19 complete original noisy-recycling runs; this selected example does not
establish a population-level trend with indistinguishability.

Panel (d) displays 5,123 stored samples from the seed-0 lossless model, selected
independently of performance, with adjacent training bins grouped from 495 to 99.
Financial target shape differs from the paper because its original data were
unavailable. These are archived full-run figures, separate from the notebook
demonstration. Full selection and retry policies remain in
[`results/figure2_manifest.json`](results/figure2_manifest.json).

The result bundle includes `runs.csv` (run identities and endpoints), `curves.csv`
(recorded learning curves), `financial_histogram.csv` (display counts and bins),
and `summary.json` (aggregate statistics). Separate retries supplement failed
originals; they are not additional independent initializations.

The compact public CSV/JSON bundle redraws the figures without original raw runs:

```bash
python implementation.py --paper error-qcbm --config configs/plot_results.json
```

The reviewed fresh environment reproduced all PNG pixels, canonical SVG
geometry/text and the saved summary. Other environments may alter rendering.

## Discussion and Next Steps

- In the Gaussian experiment with relatively low photon loss, post-selection achieved a lower KL divergence between the trained circuit’s exact output distribution and the target distribution than recycling. In contrast, recycling achieved a lower KL divergence in the high-loss experiment, suggesting its potential usefulness under substantial photon loss. However, the experiments also differed in shot budget and photon indistinguishability, so the difference cannot be attributed solely to the loss rate. We additionally conducted a small-scale training test using MMD loss, but it did not cover the same low- and high-loss settings. Further matched experiments are therefore needed to establish whether this trend is independent of the choice of loss function.
- Photon indistinguishability alone may not be a reliable predictor of QCBM performance. Its effect should be assessed jointly with the target distribution, optimization procedure, and sampling budget.
- Alternative probabilistic models may offer additional approaches to recovering output distributions from photon-loss data. As a preliminary exploration, we can use an optimization algorithm to estimate the underlying distribution under a specified photon-loss model. Further investigation is needed to assess its reconstruction accuracy, computational cost, and scalability relative to recycling.


## Reproducibility Notes

- Training uses actual native shots under `torch.no_grad()`, followed by
  post-selection/recycling and a custom NumPy SPSA update. Recycling uses
  Perceval's official `photon_recycling()` function.
- In the inspected MerLin 0.4.2 training path, a gradient-enabled `shots` request
  is disabled with a warning and backward uses exact probabilities. Actual
  finite-shot autodiff is unsupported in that path; our SPSA protocol follows
  the author implementation.
- Initial phases, perturbation seeds and initial-shot budgets are matched across
  conditions. The same gain-calibration rule can give different numerical gains.
  Both perturbations, calibration and reporting consume shots, including vacuum.
- Full runs record fresh finite-shot loss every ten updates; plotted
  `Iteration Number` is update/10. An extra initial report remains in the data.
- Ordinary configs use stock native accumulation. The optimized adapter changes
  counts only and requires full source/environment-matched evidence; a small
  sampler smoke is not production qualification. See [sampling notes](lib/SAMPLING.md).
- Historical CPU training used Torch 2.6.0, float64 and one thread.
  Noisy 50M calls took about 18–20 seconds with the adapter in those measurements.
- The Belenos model omits nonzero g2, dark counts, mode-dependent loss, phase
  errors and drift. Its two global parameters do not describe a complete device.

## Testing

```bash
cd papers/error-qcbm
python -m pytest -q
```

In a flat checkout, enter `error-qcbm` instead. The fresh Windows environment
passed 149 tests, including optical controls, native counts, recycling, SPSA,
continuation, data preparation and result rendering. Tests use synthetic fixtures
and the public noise profile without the original Hub reply or financial CSV.
The bounded notebook was also executed and its saved count totals and phase
updates checked; it is not a full paper-scale training test.

## Citation and License

Cite the original paper, reference implementation and data source below.
The catalogue's root `LICENSE` and `THIRD_PARTY_NOTICES.md` remain applicable;
SDK notices are retained in [lib/SAMPLING.md](lib/SAMPLING.md#dependency-notices).
The pinned author repository provides no explicit code license; author-derived
redistribution terms remain to be confirmed before publication. Data sources
and transformations are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

```bibtex
@article{Salavrakos2025ErrorMitigatedQCBM,
  author = {Salavrakos, Alexia and Sedrakyan, Tigran and Mills, James and Mansfield, Shane and Mezher, Rawad},
  title = {Error-mitigated photonic quantum circuit {Born} machine},
  journal = {Physical Review A},
  volume = {111},
  pages = {L030401},
  year = {2025},
  doi = {10.1103/PhysRevA.111.L030401},
  eprint = {2405.02277},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2405.02277v3}
}

@misc{QuandelaPhotonicQCBMCode,
  author = {Sedrakyan, Tigran and Salavrakos, Alexia and Mills, James and Bertasi, Eric and Fabre, Marion},
  title = {Photonic {QCBM} reference implementation},
  howpublished = {Quandela GitHub repository},
  url = {https://github.com/Quandela/photonic-qcbm/tree/d593d209128e403644b9aee595c474ce551391ae},
  note = {Pinned commit d593d209128e403644b9aee595c474ce551391ae. Component contributions are described in its README. No code-license grant identified in the reviewed tree.}
}

@misc{ECBDailyUSDEUR,
  author = {{European Central Bank}},
  title = {Daily euro foreign exchange reference rate: {USD} per {EUR}},
  howpublished = {ECB Data Portal, EXR.D.USD.EUR.SP00.A},
  url = {https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2004-01-01&endPeriod=2023-12-31&format=csvdata},
  note = {Reproduction uses 2004--2023 rates; log returns and empirical histogram calculated by reproduction authors.}
}
```
