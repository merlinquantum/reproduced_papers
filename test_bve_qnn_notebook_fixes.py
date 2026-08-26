"""Regression tests for the BVE-QNN notebook review fixes."""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOK = REPO_ROOT / "papers" / "bve_qnn" / "notebook.ipynb"


def _load_notebook():
    with NOTEBOOK.open(encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def test_notebook_exists():
    assert NOTEBOOK.is_file()


def test_notebook_has_h1_title():
    nb = _load_notebook()
    first_md = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            first_md = _cell_source(cell)
            break
    assert first_md is not None
    assert first_md.lstrip().startswith("# "), (
        "First markdown cell must start with an H1 heading"
    )


def test_notebook_has_prerequisites():
    nb = _load_notebook()
    first_md = _cell_source(nb["cells"][0])
    assert "sem_supervised_dataset.npz" in first_md
    assert "qnn_exp1_merlin_dualrail_depth32_step5000.pt" in first_md


def test_notebook_no_pip_noise():
    nb = _load_notebook()
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = out.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            assert "Requirement already satisfied" not in text, (
                "pip install output not cleared"
            )


def test_notebook_no_qnn_rebound():
    nb = _load_notebook()
    found_quantum_layer_cell = False
    for cell in nb["cells"]:
        src = _cell_source(cell)
        if "spec_mappings" in src and "quantum_layer" in src:
            found_quantum_layer_cell = True
            assert "qnn_check" in src
            break
    assert found_quantum_layer_cell, "Expected quantum_layer/spec_mappings cell missing"


def test_notebook_no_leaked_filename():
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "running_exp1" not in full


def test_notebook_no_duplicated_sentence():
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "trainables" not in full


def test_notebook_no_v3_label():
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "(v3," not in full


def test_notebook_train_from_scratch_flag():
    nb = _load_notebook()
    code = "".join(_cell_source(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "TRAIN_FROM_SCRATCH" in code, (
        "Training cell must use a TRAIN_FROM_SCRATCH flag"
    )


def test_notebook_weather_pde_has_period():
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "weather PDE." in full


def test_requirements_no_transitive_deps():
    req_path = REPO_ROOT / "papers" / "bve_qnn" / "requirements.txt"
    with req_path.open(encoding="utf-8") as f:
        lines = [
            ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")
        ]
    dep_names = [ln.split(">")[0].split("=")[0].split("<")[0] for ln in lines]
    for dep in ["perceval-quandela", "torch", "numpy"]:
        assert dep not in dep_names, (
            f"Transitive dependency '{dep}' should be removed — "
            "merlinquantum pulls it automatically"
        )


def test_readme_no_broken_latex():
    readme = REPO_ROOT / "papers" / "bve_qnn" / "README.md"
    with readme.open(encoding="utf-8") as f:
        text = f.read()
    assert "\\(" not in text and "\\)" not in text, (
        r"LaTeX \(...\) syntax does not render on GitHub — use $...$ instead"
    )


def test_neutral_atom_notebooks_exist():
    na_dir = REPO_ROOT / "papers" / "bve_qnn" / "notebooks" / "neutral_atom"
    assert (na_dir / "quantum_bve_step_by_step.ipynb").is_file(), (
        "Dataset generation notebook missing"
    )
    assert (na_dir / "running_exp1.ipynb").is_file(), (
        "Neutral-atom training notebook missing"
    )


def test_neutral_atom_config_exists():
    cfg = REPO_ROOT / "papers" / "bve_qnn" / "configs" / "neutral-atom.json"
    assert cfg.is_file(), "configs/neutral-atom.json missing"
    with cfg.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["model"]["name"] == "neutral_atom_qadence"


def test_generate_dataset_script_exists():
    script = REPO_ROOT / "papers" / "bve_qnn" / "utils" / "generate_dataset.py"
    assert script.is_file(), "utils/generate_dataset.py missing"


def test_no_cursor_coauthor_in_commits():
    base_ref = "upstream/main"
    if (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", base_ref],
            cwd=REPO_ROOT,
        ).returncode
        != 0
    ):
        base_ref = "origin/main"

    merge_base = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        capture_output=True,
        check=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()

    result = subprocess.run(
        ["git", "log", "--format=%B", f"{merge_base}..HEAD"],
        capture_output=True,
        check=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert "Co-authored-by: Cursor" not in result.stdout


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
