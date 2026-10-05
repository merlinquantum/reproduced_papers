"""DataS1 task populations, download verification and residue encodings."""

import pytest
from common import BINDING_POSITIONS, IMMUNOGENIC_POSITIONS, REPO_ROOT
from lib import data as data_module
from lib.data import ALPHABET, load_task, one_hot_features


def _config(task, positions, **extra):
    return {"seed": 49, "dataset": {"task": task, "positions": positions}, **extra}


def test_binding_official_split_and_missing_assay_exclusion():
    data = load_task(_config("binding", BINDING_POSITIONS))
    assert data["train"]["bits"].shape == (150, 20)
    assert data["test"]["bits"].shape == (999, 20)
    assert data["train"]["targets"].sum().item() == 75
    assert not set(data["train"]["peptides"]) & set(data["test"]["peptides"])
    assert data["excluded_missing_assay"] == ["DLPLIIQRL"]


def test_explicit_runtime_data_root_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cfg = _config(
        "immunogenic", IMMUNOGENIC_POSITIONS, data_root=str(REPO_ROOT / "data")
    )
    data = load_task(cfg)
    assert data["train"]["bits"].shape == (300, 20)
    assert data["test"]["targets"].sum().item() == 500


def test_onehot_residue_control_preserves_amino_acid_identity():
    data = load_task(_config("binding", BINDING_POSITIONS))
    encoded = one_hot_features(data["train"]["bits"][:1])
    assert encoded.shape == (1, 80)
    assert encoded.sum().item() == 4
    peptide = data["train"]["peptides"][0]
    assert encoded.nonzero()[:, 1].tolist() == [
        i * 20 + ALPHABET.index(peptide[p - 1]) for i, p in enumerate(BINDING_POSITIONS)
    ]


def test_corrupted_local_dataset_is_rejected(tmp_path):
    supplement = tmp_path / "QCHIPP" / "paper_supplement"
    supplement.mkdir(parents=True)
    (supplement / data_module.DATASET_FILENAME).write_text(
        "Mutant_peptide\nAAAAAAAAA\n"
    )
    with pytest.raises(ValueError, match="SHA-256"):
        data_module.ensure_dataset(tmp_path)


def test_failed_download_raises_with_manual_instructions(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(data_module.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="README"):
        data_module.ensure_dataset(tmp_path)
