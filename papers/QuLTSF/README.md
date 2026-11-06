# QuLTSF — Long-Term Time Series Forecasting with Quantum Machine Learning

## Reference and Attribution

- Paper: QuLTSF: Long-Term Time Series Forecasting with Quantum Machine Learning (arXiv, 2024)
- DOI/ArXiv: https://arxiv.org/abs/2412.13769
- Original repository: https://github.com/chariharasuthan/QuLTSF
- License and attribution notes: cite the paper and upstream repository when comparing against the original implementation.

## Overview

This folder is the initial MerLin-structured reproduction scaffold for QuLTSF.

Implemented so far:
- multivariate CSV forecasting pipeline with sliding windows,
- `qultsf_reference` model with classical projection, compact hidden block, and forecast decoder,
- `photonic_qultsf` model where the hidden block is replaced by a MerLin photonic VQC,
- `qultsf_original`, a local port of the original PennyLane hybrid implementation from the upstream QuLTSF repository.

Not complete yet:
- exact upstream Weather preprocessing,
- exact paper hyperparameters and table recreation,
- full baseline suite from the paper.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

```bash
# From the repo root
python implementation.py --paper QuLTSF --help

# From inside papers/QuLTSF
python ../../implementation.py --help
```

Example runs:

```bash
python implementation.py --paper QuLTSF --config configs/smoke_reference.json
python implementation.py --paper QuLTSF --config configs/smoke_photonic.json
python implementation.py --paper QuLTSF --config configs/qultsf_original_seq336_pred96.json
```

## Data

- Expected CSV location: `data/QuLTSF/weather.csv` or `data/time_series/weather.csv`.
- Current defaults target the canonical LTSF Weather benchmark schema:
  - timestamp column: `date`
  - task mode: benchmark-style `M`, `S`, or `MS`
  - default task mode in this repo: `MS`
  - default forecasting target: `OT` (the final benchmark target column)
- Why `OT`: this is not stated explicitly in the QuLTSF paper text, but it is the
  standard Autoformer/LTSF benchmark convention used by the upstream QuLTSF code
  path. The QuLTSF repository instructs users to download `weather.csv` from the
  Autoformer dataset bundle, and the corresponding THUML benchmark loader uses
  `target='OT'` by default for custom datasets such as Weather.
- Important nuance: the upstream QuLTSF Weather launch script actually runs with
  `features=M`, i.e. it jointly forecasts all 21 variables. The added
  `configs/qultsf_original_seq336_pred96.json` mirrors that upstream setup.
- In the shared loader, task modes mean:
  - `M`: use all non-timestamp variables as input and predict all of them
  - `S`: use only the target column as input and predict only that target
  - `MS`: use all non-timestamp variables as input and predict only the target
- Override the root with `DATA_DIR=/abs/path` or `--data-root /abs/path`.

## Outputs

Each run writes a timestamped folder under `outdir/` with:
- `config_snapshot.json`
- `history.json`
- `metrics.json`
- `metadata.json`
- `predictions.json`
- `<model_name>.pt`
- `done.txt`

## Next Steps

- tune the MerLin photonic replacement under the same evaluation protocol,
- add scripts for paper-style tables and plots.

## Provenance

- `qultsf_original` is ported from the upstream QuLTSF repository:
  - model source: `models/QuLTSF.py`
  - experiment script reference: `scripts/QuLTSF_seq_len_336.sh`
- The local port keeps the original architecture semantics while adapting them to
  this repository's runner/config layout.
