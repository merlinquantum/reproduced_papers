# How to contribute a reproduced paper

We encourage contributions of new quantum ML paper reproductions. Please follow the guidelines below:

## Mandatory structure for a reproduction

```
papers/NAME/            # Non-ambiguous acronym or fullname of the reproduced paper
├── .gitignore            # specific .gitignore rules for clean repository
├── notebook.ipynb        # Interactive exploration of key concepts
├── README.md             # Paper overview and results overview
├── requirements.txt      # additional requirements for the scripts
├── configs/              # defaults + experiment configs consumed by the repo root runner
├── cli.json              # CLI schema for the shared runner
├── lib/                  # code used by the shared runner and notebooks - as an integrated library (import shared data helpers from papers/shared/<paper>/)
├── models/               # Trained models
├── results/              # Selected generated figures, tables, or outputs from trained models
├── tests/                # Validation tests
└── utils/                # additional commandline utilities for visualization, launch of multiple trainings, etc...
```

## Reproduction template (starter kit)

Use the ready-to-go template in `papers/reproduction_template/` to bootstrap a new paper folder that follows the structure above.

Quick start:

```bash
# 1) Create your paper folder under papers/ (replace NAME with a short, unambiguous id)
cp -R papers/reproduction_template papers/NAME

cd papers/NAME

# 2) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional shared deps can go in the repo root, but each project keeps its own requirements.txt.

# 3) Run with the example config (JSON-only) via the repo-level runner
python ../../implementation.py --config configs/example.json

# 4) See outputs (default base outdir is `outdir/` inside NAME/)
ls outdir

# 5) Run tests (from inside papers/NAME/)
pytest -q
```

You can also run from the repository root:

```bash
python implementation.py --paper NAME --config configs/example.json
```

`--paper` (or `--paper-dir`) is mandatory so the shared runner knows which reproduction folder to load.

**Placeholder guard:** If any config value still contains a `<<...>>` placeholder (e.g., `"teacher_path": "<<TEACHER_PATH>>"`), the shared runner aborts early with a clear error. Replace these placeholders with real paths/values before launching a run.

Then edit the placeholders in:
- `README.md` — paper reference/authors, reproduction details, CLI options, results analysis
- `configs/example.json` — dataset/model/training defaults (extend or add more configs)
- `configs/defaults.json` + `cli.json` — default parameters plus the CLI schema consumed by the shared runner (every project must expose `lib.runner.train_and_evaluate`, which the runtime imports automatically)
- Any `dtype` entries in those configs (top-level or nested) are normalized at runtime into `(label, torch.dtype)` pairs via `runtime_lib.dtypes`, so projects can rely on validated torch dtypes without re-implementing alias logic.
- `lib/runner.py` and supporting modules inside `lib/` — dataset/model/training logic invoked by the shared runner
- `runtime_lib.config.load_config` / `.deep_update` handle JSON loading and overrides globally; the template already wires `lib.config` to these helpers so you must not add a custom `lib/config.py` (JSON is the only supported format).

> **Note:** Every reproduction has its own `requirements.txt`. Install the relevant file before running `implementation.py --paper ...` to ensure dependencies are available.

Notes:
- Configs are JSON-only in the template.
- Each run creates a timestamped folder under the base `outdir` (default `outdir/`): `run_YYYYMMDD-HHMMSS/` with `config_snapshot.json` and your artifacts.
- Tests are intended to be run from inside the paper folder (e.g., `cd NAME && PYTHONPATH=. pytest -q`).

## Submission process

1. **Propose** the paper in our [GitHub Discussions](https://github.com/merlinquantum/merlin/discussions)
2. **Implement** using the repository tools, following the structure above
3. **Validate** results against the original paper
4. **Document** in Jupyter notebook format
5. **Submit** a pull request with the complete reproduction folder
6. **Summarize** in a couple of lines the results of the reproduced paper in the table in the main README.

## Contribution requirements

- High-impact quantum ML papers (>50 citations preferred)
- Photonic/optical quantum computing focus
- Implementable with current repository features
- Clear experimental validation

## Recognition

Contributors are recognized in:
- Paper reproduction documentation
- MerLin project contributors list
- Academic citations in MerLin publications

# Code Style and Quality

This repository uses [Ruff](https://docs.astral.sh/ruff/) for consistent code formatting and linting across all paper implementations.

## Usage

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

## Configuration

- Code style rules are defined in `pyproject.toml`
- GitHub Actions automatically check all PRs and pushes
- Pre-commit hooks run ruff automatically before commits
