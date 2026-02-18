from __future__ import annotations

import json
from pathlib import Path

import pytest
from common import load_project_defaults


def _write_embeddings(embeddings_dir: Path, dim: int) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        {"embedding": [0.1] * dim, "label": 0, "sentence": "a"},
        {"embedding": [0.2] * dim, "label": 1, "sentence": "b"},
        {"embedding": [0.3] * dim, "label": 0, "sentence": "c"},
        {"embedding": [0.4] * dim, "label": 1, "sentence": "d"},
    ]
    splits = {
        "train": samples[:2],
        "eval": samples[2:3],
        "test": samples[3:4],
    }
    for split, rows in splits.items():
        payload = {
            "embeddings": [row["embedding"] for row in rows],
            "labels": [row["label"] for row in rows],
            "sentences": [row["sentence"] for row in rows],
            "embedding_dim": dim,
            "num_samples": len(rows),
        }
        path = embeddings_dir / f"{split}_embeddings.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_cfg(tmp_path: Path, model_name: str) -> dict:
    cfg = load_project_defaults()
    cfg["dataset"]["embeddings_dir"] = str(tmp_path / "embeddings")
    cfg["model"]["name"] = model_name
    cfg["model"]["embedding_dim"] = 4
    cfg["model"]["hidden_dim"] = 2
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["learning_rate"] = 1e-3
    return cfg


def test_merlin_model_smoke(tmp_path):
    try:
        import lib.merlin_llm_models  # noqa: F401
    except ImportError:
        pytest.skip("MerLin dependencies are not installed.")

    from lib import runner as qllm_runner

    _write_embeddings(Path(tmp_path) / "embeddings", dim=4)
    cfg = _build_cfg(tmp_path, "merlin-basic")
    cfg["model"]["quantum_modes"] = 2
    cfg["model"]["photons"] = 1

    run_dir = Path(tmp_path) / "run_merlin"
    run_dir.mkdir(parents=True, exist_ok=True)
    qllm_runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "metrics.json").exists() or (run_dir / "SKIPPED.txt").exists()


def test_torchquantum_model_smoke(tmp_path):
    try:
        import lib.torchquantum_models  # noqa: F401
    except ImportError:
        pytest.skip("TorchQuantum dependencies are not installed.")

    from lib import runner as qllm_runner

    _write_embeddings(Path(tmp_path) / "embeddings", dim=4)
    cfg = _build_cfg(tmp_path, "torchquantum")
    cfg["model"]["encoder_configs"] = [
        {"n_qubits": 2, "n_layers": 1, "connectivity": 1}
    ]
    cfg["model"]["pqc_config"] = [
        {"n_qubits": 2, "n_main_layers": 1, "connectivity": 1, "n_reuploading": 1}
    ]

    run_dir = Path(tmp_path) / "run_torchquantum"
    run_dir.mkdir(parents=True, exist_ok=True)
    qllm_runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "metrics.json").exists() or (run_dir / "SKIPPED.txt").exists()
