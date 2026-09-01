from __future__ import annotations

import json

from common import PROJECT_DIR, load_runtime_ready_config
from lib.runner import train_and_evaluate


def _base_cfg(tmp_path):
    cfg = load_runtime_ready_config()
    cfg["dataset"]["root"] = str(
        PROJECT_DIR.parent.parent / "data" / "variational_qml_ts_benchmark"
    )
    cfg["training"]["epochs"] = 2
    cfg["training"]["use_convergence"] = False
    return cfg


def test_mlp_smoke_writes_metrics(tmp_path):
    cfg = _base_cfg(tmp_path)
    run_dir = tmp_path / "mlp"
    train_and_evaluate(cfg, run_dir)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["model_name"] == "mlp"
    assert m["epochs"] == 2
    assert m["mse_test"] > 0
    assert (run_dir / "losses.csv").exists()


def test_quantum_qrnn_smoke(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["dataset"]["name"] = "henon_1000"
    cfg["model"]["name"] = "qrnn"
    cfg["model"]["params"].update(
        {"ansatz": "paper_no_reset", "num_qubits": 4, "hidden_size": 2, "bugfix": True}
    )
    run_dir = tmp_path / "qrnn"
    train_and_evaluate(cfg, run_dir)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["model_name"] == "qrnn"
    assert m["num_parameters"] > 0
