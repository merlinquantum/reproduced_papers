# MerLin Reproduced Papers

## About this repository


This repository contains implementations and resources for reproducing key quantum machine learning papers, with a focus on photonic and optical quantum computing.

It is part of the main MerLin project: [https://github.com/merlinquantum/merlin](https://github.com/merlinquantum/merlin)
and complements the online documentation available at:

[https://merlinquantum.ai/research/reproduced_papers.html](https://merlinquantum.ai/research/reproduced_papers.html)

Each paper reproduction is designed to be accessible, well-documented, and easy to extend. Contributions are welcome!

## How to contribute a reproduced paper

We encourage contributions of new quantum ML paper reproductions. Please follow the guidelines below:

### Mandatory structure for a reproduction

```
NAME/                     # Non-ambiguous acronym or fullname of the reproduced paper
├── .gitignore            # specific .gitignore rules for clean repository
├── implementation.py     # Main engine to train a model - the cli can accept parameters or config file
├── notebook.ipynb        # Interactive exploration of key concepts
├── README.md             # Paper overview and results overview
├── requirements.txt      # additional requirements for the scripts
├── configs/              # predefined configurations to train models
├── data/                 # Datasets and preprocessing if any
├── lib/                  # code used by implementation.py and notebook.ipynb - as a integrated library
├── models/               # Trained models 
├── results/              # Selected generated figures, tables, or outputs from trained models
├── tests/                # Validation tests
└── utils/                # additional commandline utilities for visualization, launch of multiple trainings, etc...
```

### Reproduction template (starter kit)

Use the ready-to-go template in `reproduction_template/` to bootstrap a new paper folder that follows the structure above.

Quick start:

```bash
# 1) Create your paper folder (replace NAME with a short, unambiguous id)
cp -R reproduction_template NAME

cd NAME

# 2) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) Run with the example config (JSON-only)
python implementation.py --config configs/example.json

# 4) See outputs (default base outdir is `outdir/`)
ls outdir

# 5) Run tests (from inside NAME/)
pytest -q
```

Then edit the placeholders in:
- `README.md` — paper reference/authors, reproduction details, CLI options, results analysis
- `configs/example.json` — dataset/model/training defaults (extend or add more configs)
- `implementation.py` and `lib/` — actual dataset/model/training logic

Notes:
- Configs are JSON-only in the template.
- Each run creates a timestamped folder under the base `outdir` (default `outdir/`): `run_YYYYMMDD-HHMMSS/` with `config_snapshot.json` and your artifacts.
- Tests are intended to be run from inside the paper folder (e.g., `cd NAME && PYTHONPATH=. pytest -q`).

### Submission process

1. **Propose** the paper in our [GitHub Discussions](https://github.com/merlinquantum/merlin/discussions)
2. **Implement** using the repository tools, following the structure above
3. **Validate** results against the original paper
4. **Document** in Jupyter notebook format
5. **Submit** a pull request with the complete reproduction folder

### Contribution requirements

- High-impact quantum ML papers (>50 citations preferred)
- Photonic/optical quantum computing focus
- Implementable with current repository features
- Clear experimental validation

### Recognition

Contributors are recognized in:
- Paper reproduction documentation
- MerLin project contributors list
- Academic citations in MerLin publications

## Code Style and Quality

This repository uses [Ruff](https://docs.astral.sh/ruff/) for consistent code formatting and linting across all paper implementations.

### Usage

**Check code style:**
```bash
ruff check .
```

**Format code:**
```bash
ruff format .
```

**Install pre-commit hooks (recommended):**
```bash
pip install pre-commit
pre-commit install
```

### Configuration

- Code style rules are defined in `pyproject.toml`
- GitHub Actions automatically check all PRs and pushes
- Pre-commit hooks run ruff automatically before commits