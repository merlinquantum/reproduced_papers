"""Readable model and experiment names used in figures, tables and the notebook."""

import json
from pathlib import Path

import pytest
from power_of_QNN.lib.display_names import (
    experiment_label,
    model_label,
    photonic_description,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _model_names() -> set[str]:
    names = set()
    for path in CONFIGS.glob("*.json"):
        config = json.loads(path.read_text())
        for section in (config, config.get("fisher", {}), config.get("training", {})):
            models = section.get("models", []) if isinstance(section, dict) else []
            names.update(model["name"] for model in models if "name" in model)
    return names


def test_every_configured_model_has_a_readable_name():
    names = _model_names()
    assert names, "no model names found in configs"
    for name in sorted(names):
        label = model_label(name, detailed=True)
        assert label and name not in label


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Q2_d40", "qubit QNN (paper)"),
        ("Q3_qubit_qnn_reps3", "qubit QNN, 3 uploads"),
        ("classical_42", "linear [4,2]"),
        ("A8_d40", "photonic 10-mode"),
        ("B2_6m_cluster", "photonic 6-mode"),
    ],
)
def test_short_labels(name, expected):
    assert model_label(name) == expected


def test_codes_for_the_same_circuit_share_a_description():
    # B6 and A8 are one circuit registered by two design searches.
    assert photonic_description("B6_10m_cluster") == photonic_description(
        "A8_10m_cluster_reps3"
    )
    assert (
        photonic_description("A8_d40") == "photonic 10-mode, cluster input, 3 uploads"
    )


def test_paper_depth_depends_on_encoding():
    assert photonic_description("P4_hard") == "photonic 8-mode, Fock input, 2 uploads"
    assert photonic_description("P4_easy").endswith("1 upload, easy encoding")


def test_unknown_names_fail_clearly():
    with pytest.raises(KeyError):
        model_label("Z9_unknown")
    with pytest.raises(KeyError):
        experiment_label("fair_unknown")


def test_planned_design_uses_the_recorded_mode_count():
    # The 8-mode diagnostic run reuses the code name photonic_hard.
    assert model_label("photonic_hard").startswith("photonic 6-mode")
    assert model_label("photonic_hard", modes=8).startswith("photonic 8-mode")


def test_ablation_encoding_is_not_called_the_paper_model():
    assert "RY" in model_label("easy_qnn_ry") and "(paper)" not in model_label(
        "easy_qnn_ry"
    )
