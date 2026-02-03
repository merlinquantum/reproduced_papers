# Limitations of Amplitude Encoding on Quantum Classification

## Reference and Attribution

- Paper: Limitations of Amplitude Encoding on Quantum Classification(2025)
- Authors: Wang *et al.*
- DOI/ArXiv: [2503.01545](https://arxiv.org/abs/2503.01545)


## Overview

### 🎯 Main goal 
>

### Main result

>


### Main contributions of the paper

> 

### Their framework

### Difference in framework

### Their results

### Our results

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: the paper-level `lib/runner.py`. The CLI is entirely described in `cli.json`, so updating/adding arguments does not require editing Python code.

```bash
# From inside papers/reproduction_template
python l../../implementation.py  --help

# From the repo root
python implementation.py --paper DQNN --help
```

Example overrides (see `cli.json` for the authoritative list):

- `--config CONFIG_NAME` Load an additional JSON config (merged over `defaults.json`). The config path is automatically handled by the code.

Example runs:

```bash
# From a JSON config (inside the project)
python ../../implementation.py  --config configs/defaults.json

# Override some parameters inline
python ../../implementation.py  --config configs/defaults.json --TODO 50 

# Equivalent from the repo root
python implementation.py --paper AA_study --config configs/defaults.json --TODO 50
```

## Project structure --> TODO
- `papers.DQNN.lib/runner.py` — The file to run for every experiment.
- `papers.DQNN.lib/` — core papers.DQNN.library modules used by scripts.
  - `torchmps/` — Repository to instanciate a MPS tensor module in Torch.
  - `ablation_exp.py`, `bond_dimension_exp.py`, `default_exp.py`- Files containing the function to run the corresponding experiment.
  - `boson_sampler.py` - The file containg the class managing the quantum layers.
  - `classical_utils.py`, `photonic_qt_utils.py` - Files containing utility functions.
  - `model.py` — The torch module implementing the quantum train algorithm.
- `configs/` — Experiment configs + CLI schema consumed by the shared runner. The available ones are below.
  - `defaults.json`, `cli.json`, `bond_dim_exp.json`, `ablation_exp.json`
- Other
  - `requirements.txt` — Python dependencies.
  - `tests/` - Unitary tests to make sure the papers.DQNN.library works correctly.
  - `utils/` — Containing the `utils.py` file used for plotting and repo utility functions.

## Results and Analysis

- The results are stored in the [results](results/) folder. Logs and figures will be saved in the [outdir](outdir/) directory.
- To reproduce the experiments, simply call these lines at the paper level:
 
 For just a basic training and evaluation:
 >``python3 ../../implementation.py  --config defaults.json``



## Extensions and Next Steps


## Testing

Run tests from inside the `papers/AA_study/` directory:

```bash
cd papers/AA_study
pytest -q
```
Notes:
- Tests are scoped to this template folder and expect the current working directory to be `DQNN/`.
- If `pytest` is not installed: `pip install pytest`.

## Acknowledgments