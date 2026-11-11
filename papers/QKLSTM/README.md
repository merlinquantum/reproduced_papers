# QK-LSTM for Climate Time-Series Forecasting — Reproduction

Status: scaffold only (from `papers/reproduction_template`); the runner still writes the template's placeholder
artifact. Implementation, results and deviations are added as the reproduction proceeds.

## Reference and Attribution

- Paper: Quantum Kernel-Based Long Short-term Memory for Climate Time-Series Forecasting (arXiv preprint, 2024)
- Authors: Hsu, Chen, Li, Lee, Chen
- DOI/ArXiv: <https://arxiv.org/abs/2412.08851>
- Original repository (if any): none (no code link in the paper or on the abstract page)
- License and attribution notes: the paper is CC BY 4.0. The method is introduced in the companion paper
  arXiv:2411.13225.

## Overview

The paper forecasts the daily Air Quality Index (AQI) of Bengaluru from 11 pollutant features with an LSTM whose
gate affine maps are replaced by quantum-kernel expansions (4-qubit fidelity kernel), compared with a classical LSTM.

- Data: CPCB daily city data via the Kaggle mirror "Air Quality Data in India (2015 - 2020)" v12 (CC0), packaged at
  `data/QKLSTM/air_quality_india/city_day.csv` (provenance and sha256 in `SOURCE.md` beside it).

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: the repo-level `implementation.py`. The CLI is entirely described in `cli.json`, so updating/adding arguments does not require editing Python code.

```bash
# From inside papers/QKLSTM
python ../../implementation.py --help

# From the repo root
python implementation.py --paper QKLSTM --help
```

Example overrides (see `cli.json` for the authoritative list):

- `--config PATH` Load an additional JSON config (merged over `defaults.json`).
- `--seed INT` Random seed applied via the generic runtime.
- `--dtype STR` Force a global torch/NumPy dtype (e.g., `float32`, `float64`).
- `--device STR` Device override (`cpu`, `cuda:0`, `mps`, etc.).
- `--log-level LEVEL` Logging verbosity (`DEBUG`, `INFO`, ...).
- `--outdir DIR` Base output directory (default in `defaults.json`).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--lr FLOAT` Override `training.lr`.

Example runs:

```bash
# From a JSON config (inside the project)
python ../../implementation.py --config configs/example.json

# Override some parameters inline
python ../../implementation.py --config configs/example.json --epochs 50 --lr 1e-3

# Equivalent from the repo root
python implementation.py --paper QKLSTM --config configs/example.json --epochs 50 --lr 1e-3
```

The script saves a snapshot of the resolved config alongside results and logs.

### Data location and shared code

- The shared data root defaults to `<repo>/data`; each paper is scoped to a subfolder named after the paper (here `data/QKLSTM/`). The runner resolves this via `runtime_lib.data_paths.paper_data_dir`, so you usually just pass `--paper` or set `dataset.root` to `null` and downloads land in the right place.
- Override the base root with `DATA_DIR=/abs/path` or `--data-root /abs/path` when calling `implementation.py`; the paper subfolder is appended automatically.
- The dataset is bundled: `data/QKLSTM/air_quality_india/city_day.csv` (see `SOURCE.md` there).
- Reusable helpers can live under `papers/shared/<paper>/...`; import them from your paper code to avoid duplication while keeping code on the Python path (e.g., `from papers.shared.<paper>.module import helper`).

### Output directory and generated files

At each run, a timestamped folder is created under the base `outdir` (default: `outdir`):

```
<outdir>/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration used for the run
└── done.txt               # Placeholder artifact (replace with real outputs)
```

Notes:
- Change the base output directory with `--outdir` or in `configs/example.json` (key `outdir`).
- Add your own artifacts in the training code (e.g. checkpoints, metrics, figures). The template ignores `outdir/` in Git by default.

## Configuration

Place configuration files in `configs/`.

- `defaults.json` defines the default parameter values.
- `cli.json` declares the CLI arguments, their types, and the config keys they mutate.
- The shared runtime automatically loads `configs/defaults.json`, `cli.json`, and calls `lib/runner.py::train_and_evaluate`, so keep that entry point in every project.
- Any config value named `dtype` is normalized into a `(label, torch.dtype)` pair at runtime via `runtime_lib.dtypes`, so you can inspect both the requested string and the resolved `torch.dtype` without duplicating conversion logic.
- `example.json` shows the structure for a specific experiment.
- Keys typically include: dataset, model, training, evaluation, logging.
- Precision control: add a top-level `"dtype"` entry (mirrors `"seed"`) to force the entire pipeline (datasets + models) to run in a specific torch dtype; individual models may still expose a `model.dtype` override if required.

**Placeholder guard:** If any config value still contains a `<<...>>` placeholder (e.g., `"teacher_path": "<<TEACHER_PATH>>"`), the shared runner aborts early with a clear error. Replace these placeholders with real paths/values before launching a run.

## Data

- **Source.** The paper uses daily city air-quality data from India's Central Pollution Control Board (CPCB,
  <https://cpcb.nic.in/>), Bengaluru, 2015-2020, accessed by the authors on 2020-11-04. No public historical
  download of that extract was found on the CPCB portal, so this reproduction uses the community mirror
  "Air Quality Data in India (2015 - 2020)" (Kaggle `rohanrao/air-quality-data-in-india`, version 12, last updated
  2020-07-28, CC0: Public Domain), which states CPCB as its origin. Only `city_day.csv` is kept, unmodified, at
  `data/QKLSTM/air_quality_india/city_day.csv`; provenance and SHA-256 are in `SOURCE.md` beside it.
- **Substitute status.** Same source institution and period, but a third-party extract: the paper's own file cannot
  be compared with it, so small differences in values or coverage cannot be excluded. Results are labelled
  accordingly.
- **File.** 2,574,056 bytes, 29,531 rows, 26 cities; columns City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2,
  O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket.
- **Bengaluru series.** 2,009 daily rows, 2015-01-01 to 2020-07-01, every calendar day present once.
  Missing fraction per column: PM2.5 7.3%, PM10 17.9%, NO 0.3%, NO2 0.3%, NOx 0.2%, NH3 10.1%, CO 0.6%, SO2 0.3%,
  O3 7.2%, Benzene 13.2%, Toluene 4.6%, Xylene 100%, AQI 4.9%. PM10 starts on 2015-11-07, PM2.5 on 2015-03-20 and
  AQI on 2015-03-21; the other columns start on 2015-01-01. 1,368 of the 2,009 days have all eleven features.
- **Features.** Xylene has no value for Bengaluru, which matches the paper's exclusion of it. The eleven features
  used are PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene and Toluene (the paper names nine of them; Benzene
  and Toluene are inferred as the remaining two). The target is AQI.
- **Preprocessing** (paper Eqs. 2-3, details reconstructed): per-feature outliers with |z| > 3 are removed and
  missing values are linearly interpolated in time; leading gaps are filled from the first valid value; days
  without an AQI value are not used as targets. The configuration keys are documented under Configuration.

## Results and Analysis

- Where results are stored, how to reproduce key figures/tables
- Any divergence from reported metrics and possible causes
- Post-hoc analyses (ablation, sensitivity, robustness)

## Extensions and Next Steps

- Potential model variations to explore
- Additional datasets or tasks
- Improved training strategies or evaluation metrics

## Reproducibility Notes

- Random seed control
- Precision/number format control via `dtype` (global or per-model) so experiments can target `float32`, `float64`, `bfloat16`, etc.
- Determinism settings (if applicable)
- Exact versions of libraries (consider `pip freeze > results/requirements.txt`)

## Testing

Run tests from inside the `papers/QKLSTM/` directory:

```bash
cd papers/QKLSTM
pytest -q
```

Notes:
- Tests are scoped to this folder and expect the current working directory to be `QKLSTM/`.
- If `pytest` is not installed: `pip install pytest`.
