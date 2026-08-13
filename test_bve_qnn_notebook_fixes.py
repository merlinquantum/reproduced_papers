"""Tests for notebook fixes in PR #99 — run from reproduced_papers repo root."""

import json
import os

NOTEBOOK = os.path.join("papers", "bve_qnn", "notebook.ipynb")


def _load_notebook():
    with open(NOTEBOOK, encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def test_notebook_exists():
    assert os.path.isfile(NOTEBOOK)


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
    for cell in nb["cells"]:
        src = _cell_source(cell)
        if "spec_mappings" in src and "quantum_layer" in src:
            assert "qnn_check" in src
            break


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
    req_path = os.path.join("papers", "bve_qnn", "requirements.txt")
    with open(req_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    dep_names = [ln.split(">")[0].split("=")[0].split("<")[0] for ln in lines]
    for dep in ["perceval-quandela", "torch", "numpy"]:
        assert dep not in dep_names, (
            f"Transitive dependency '{dep}' should be removed — "
            "merlinquantum pulls it automatically"
        )


def test_readme_no_broken_latex():
    readme = os.path.join("papers", "bve_qnn", "README.md")
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    assert "\\(" not in text and "\\)" not in text, (
        r"LaTeX \(...\) syntax does not render on GitHub — use $...$ instead"
    )


def test_neutral_atom_notebooks_exist():
    na_dir = os.path.join("papers", "bve_qnn", "notebooks", "neutral_atom")
    assert os.path.isfile(os.path.join(na_dir, "quantum_bve_step_by_step.ipynb")), (
        "Dataset generation notebook missing"
    )
    assert os.path.isfile(os.path.join(na_dir, "running_exp1.ipynb")), (
        "Neutral-atom training notebook missing"
    )


def test_neutral_atom_config_exists():
    cfg = os.path.join("papers", "bve_qnn", "configs", "neutral-atom.json")
    assert os.path.isfile(cfg), "configs/neutral-atom.json missing"
    with open(cfg, encoding="utf-8") as f:
        data = json.load(f)
    assert data["model"]["name"] == "neutral_atom_qadence"


def test_generate_dataset_script_exists():
    script = os.path.join("papers", "bve_qnn", "utils", "generate_dataset.py")
    assert os.path.isfile(script), "utils/generate_dataset.py missing"


def test_no_cursor_coauthor_in_commits():
    import subprocess

    result = subprocess.run(
        ["git", "log", "--format=%B", "origin/main..HEAD"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert "Co-authored-by: Cursor" not in result.stdout


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
