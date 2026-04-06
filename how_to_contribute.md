# How to contribute a reproduced paper

We encourage contributions of new quantum ML paper reproductions. Please follow the guidelines below:

## Mandatory structure for a reproduction

```
papers/NAME/            # Non-ambiguous acronym or fullname of the reproduced paper
├── .gitignore            [recommended] specific ignore rules for clean repository
├── notebook.ipynb        [required] interactive exploration of key concepts
├── README.md             [required] paper overview and results overview
├── requirements.txt      [required] additional requirements for the scripts
├── configs/              [required] defaults + experiment configs consumed by the repo root runner
├── cli.json              [required] CLI schema for the shared runner
├── lib/                  [required] code used by the shared runner and notebooks
│   └── runner.py         [required] entrypoint: lib.runner.train_and_evaluate(cfg, run_dir)
├── models/               [recommended] trained models
├── results/              [recommended] selected generated figures, tables, or outputs from trained models
├── tests/                [required] validation tests
└── utils/                [required] additional commandline utilities for visualization, launch of multiple trainings, etc...
```

`data/` is shared at repository root (not inside each paper): write datasets/artifacts under `data/<NAME>/`.

Nested multi-part papers are also supported when one publication is split into independent subprojects. In that case the container directory holds shared documentation only, and each part must follow the full reproduction structure on its own:

```text
papers/NAME/
├── README.md
├── .gitignore
├── requirements.txt
├── PART1/
│   ├── cli.json
│   ├── configs/defaults.json
│   ├── lib/runner.py
│   ├── notebook.ipynb
│   ├── tests/
│   └── utils/
└── PART2/
    └── ...
```

Example: `papers/fock_state_expressivity/VQC_classif/` and its sibling directories are validated as separate reproductions; the parent `papers/fock_state_expressivity/` is not expected to expose its own `cli.json` or `lib/runner.py`.
In this multi-part layout, a shared parent `.gitignore` is acceptable for all subprojects.

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

## Required conventions

- Keep the runnable entrypoint in `lib/runner.py` and expose `train_and_evaluate(cfg, run_dir)`.
- Do not add `runtime.json` or `runtime_entry.py`; the shared runtime now assumes `configs/defaults.json`, `cli.json`, and `lib/runner.py`.
- Keep `cli.json` project-specific only. Global flags like `--config` and `--outdir` are injected by `runtime_lib/global_cli.json`.
- Do not duplicate shared startup logs or seed wrappers in each project; shared runtime handles run banner/config logging and seeding.
- If you use `dtype` in configs, rely on runtime normalization (`dtype` keys arrive in project code as validated `(label, torch.dtype)` pairs).
- Use the shared data root convention: default is repo-level `data/` and each paper should store data under `data/<NAME>/` (override with `--data-root` or `DATA_DIR` when needed).

### Shared code under `papers/shared/`

- Put reusable, cross-paper logic in `papers/shared/<topic_or_paper>/` (especially dataset preparation/loading utilities reused by multiple reproductions).
- Keep each paper runnable on its own: paper-local modules in `papers/<NAME>/lib/` should import from `papers/shared/...` through thin wrappers (for example a paper-local `lib/data.py` forwarding to shared helpers).
- Avoid duplicating the same data/helper implementation across multiple papers when a shared module is a better fit.

Notes:
- Configs are JSON-only in the template.
- Each run creates a timestamped folder under the base `outdir` (default `outdir/`): `run_YYYYMMDD-HHMMSS/` with `config_snapshot.json` and your artifacts.
- Tests are intended to be run from inside the paper folder (e.g., `cd NAME && PYTHONPATH=. pytest -q`).

## Submission process

1. **Create a branch with an allowed prefix** (`release-`, `paper-`, `PAPER-`, `pml-`, or `PML-`) before opening a PR to `main`;
2. **Propose** the paper in our [GitHub Discussions](https://github.com/merlinquantum/merlin/discussions);
3. **Implement** using the repository tools, following the structure above;
4. **Reproduce with paper settings**: validate results in the original paper setup (datasets/splits/metrics/hyperparameters as closely as possible);
5. **Reproduce with MerLin settings**: run and validate the same study through the MerLin runtime and project conventions;
6. **Document** in Jupyter notebook format;
7. **Format and lint** before opening the PR: run `ruff format .` and `ruff check .` with latest stable ruff (fix reported issues);
8. **Submit** a pull request with the complete reproduction folder;
9. **Summarize** in a couple of lines the results of the reproduced paper in the table in the main README.

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

**Run the paper-structure validator manually:**
```bash
# Validate every paper under papers/
python3 scripts/check_paper_structure.py --all

# Validate one paper by passing any path inside it
python3 scripts/check_paper_structure.py papers/nearest_centroids_merlin/configs/defaults.json

# Validate one subproject inside a multi-part paper
python3 scripts/check_paper_structure.py papers/fock_state_expressivity/VQC_classif/configs/defaults.json

# Validate only the files currently staged for commit
python3 scripts/check_paper_structure.py $(git diff --cached --name-only)
```

The validator checks the repository paper layout described above, including required top-level files/directories, the presence of `lib/runner.py`, valid JSON configs, and accidental committed `results/run_*` artifacts that should live under `outdir/`.

## Configuration

- Code style rules are defined in `pyproject.toml`
- GitHub Actions automatically check all PRs and pushes
- Pre-commit hooks run ruff and the paper-structure validator automatically before commits
