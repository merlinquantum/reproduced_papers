"""Configuration and import smoke tests for BP_QNN."""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_required_configs_are_explicit():
    for name in (
        "original_config",
        "fig3_gb",
        "fig4_gb",
        "fig3_merlin",
        "demo_fig3",
        "demo_fig3_merlin",
    ):
        config = json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())
        assert config["qubits"]
        assert config["layers"]
        assert config["samples"] > 0

    fig3 = json.loads((PROJECT_DIR / "configs" / "fig3_gb.json").read_text())
    fig4 = json.loads((PROJECT_DIR / "configs" / "fig4_gb.json").read_text())
    assert len(fig3["qubits"]) > 2
    assert fig3["layers_per_qubit"] == 10
    assert max(fig4["layers"]) == 500

    demo = json.loads((PROJECT_DIR / "configs" / "demo_fig3.json").read_text())
    assert demo["qubits"] == [2, 3, 4, 5, 6]
    assert demo["layers_per_qubit"] == 10
    assert demo["samples"] == 32

    photonic_demo = json.loads(
        (PROJECT_DIR / "configs" / "demo_fig3_merlin.json").read_text()
    )
    assert photonic_demo["qubits"] == [2, 3, 4]
    assert photonic_demo["theta_initializations"] == ["arcsin"]


def test_runner_imports_without_optional_quantum_backends():
    import sys

    sys.path.insert(0, str(PROJECT_DIR.parent.parent))
    from papers.BP_QNN.lib.runner import train_and_evaluate

    assert callable(train_and_evaluate)
