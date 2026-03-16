from __future__ import annotations

import pandas as pd
import torch
from common import load_runtime_ready_config
from lib.data import build_dataloaders
from lib.model import RNNRegressor


def test_rnn_regressor_matches_rnncell_unroll():
    torch.manual_seed(0)
    batch_size, seq_len, features = 2, 5, 3
    model = RNNRegressor(input_size=features, hidden_dim=4, layers=1, dropout=0.0)
    sample = torch.randn(batch_size, seq_len, features)

    rnn_outputs, _ = model.rnn(sample)
    final_hidden = rnn_outputs[:, -1]

    cell = torch.nn.RNNCell(features, model.rnn.hidden_size)
    cell.weight_ih.data.copy_(model.rnn.weight_ih_l0.data)
    cell.weight_hh.data.copy_(model.rnn.weight_hh_l0.data)
    cell.bias_ih.data.copy_(model.rnn.bias_ih_l0.data)
    cell.bias_hh.data.copy_(model.rnn.bias_hh_l0.data)

    hidden = torch.zeros(batch_size, model.rnn.hidden_size)
    for t in range(seq_len):
        hidden = cell(sample[:, t, :], hidden)

    assert torch.allclose(hidden, final_hidden, atol=1e-6)

    preds = model(sample)
    expected = model.head(final_hidden).squeeze(-1)
    assert torch.allclose(preds, expected, atol=1e-6)


def test_build_dataloaders_parses_alias_columns(tmp_path):
    rows = []
    for idx in range(20):
        rows.append(
            {
                "timestamp": f"2020-01-01 {idx:02d}:00:00",
                "Temperature (C)": 10.0 + idx * 0.5,
                "Humidity": 0.4 + idx * 0.01,
            }
        )
    csv_path = tmp_path / "synthetic_weather.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    cfg = load_runtime_ready_config()
    dataset_cfg = cfg["dataset"]
    dataset_cfg["path"] = str(csv_path)
    dataset_cfg["kaggle_dataset"] = None
    dataset_cfg["preprocess"] = None
    dataset_cfg["target_column"] = "temperature"
    dataset_cfg["feature_columns"] = ["temperature", "humidity"]
    dataset_cfg["sequence_length"] = 4
    dataset_cfg["prediction_horizon"] = 1
    dataset_cfg["batch_size"] = 2
    dataset_cfg["max_rows"] = None
    dataset_cfg["train_ratio"] = 0.5
    dataset_cfg["val_ratio"] = 0.25

    train_loader, val_loader, test_loader, metadata = build_dataloaders(cfg)

    assert metadata["input_size"] == 2
    assert metadata["splits"]["train"] > 0
    assert metadata["splits"]["val"] > 0
    assert metadata["splits"]["test"] > 0

    batch_sequences, batch_targets = next(iter(train_loader))
    assert batch_sequences.shape[-1] == 2
    assert batch_targets.ndim == 1
    assert len(val_loader.dataset) + len(test_loader.dataset) + len(
        train_loader.dataset
    ) == sum(metadata["splits"].values())


def _has_merlin() -> bool:
    try:
        import importlib

        import merlin as ml

        if not hasattr(ml, "QuantumLayer"):
            return False
        if not hasattr(ml, "MeasurementStrategy"):
            return False
        # Photonic QRNN requires partial measurement support (Merlin v0.3+).
        if not hasattr(ml.MeasurementStrategy, "partial"):
            return False
        importlib.import_module("merlin.core.partial_measurement")
        importlib.import_module("merlin.core.state_vector")
        return True
    except Exception:
        return False


def test_photonic_qrnn_supports_batched_forward_and_backward():
    if not _has_merlin():
        return

    from lib.photonic_qrnn import PhotonicQRNNConfig, PhotonicQRNNRegressor

    torch.manual_seed(0)

    cfg = PhotonicQRNNConfig(
        kd=1,
        kh=1,
        depth=1,
        shots=0,
        dtype=torch.float32,
        measurement_space="dual_rail",
    )
    model = PhotonicQRNNRegressor(input_size=2, config=cfg)

    x = torch.randn(2, 3, 2)
    y = model(x)
    assert y.shape == (2,)

    loss = y.sum()
    loss.backward()
    assert model.cell.readout.weight.grad is not None


def test_photonic_qrnn_pads_missing_features_with_zeros():
    if not _has_merlin():
        return

    from lib.photonic_qrnn import PhotonicQRNNConfig, PhotonicQRNNRegressor

    torch.manual_seed(0)

    cfg = PhotonicQRNNConfig(
        kd=1,
        kh=1,
        depth=1,
        shots=0,
        dtype=torch.float32,
        measurement_space="dual_rail",
    )
    # Dataset provides only 1 feature; model will pad to 2*kd=2.
    model = PhotonicQRNNRegressor(input_size=1, config=cfg)

    x = torch.randn(2, 3, 1)
    y = model(x)
    assert y.shape == (2,)

    loss = y.sum()
    loss.backward()
    assert model.cell.readout.weight.grad is not None


def test_photonic_training_per_sample_steps_with_batched_loader():
    if not _has_merlin():
        return

    from lib.photonic_qrnn import PhotonicQRNNConfig, PhotonicQRNNRegressor
    from lib.training import train_one_epoch

    torch.manual_seed(0)

    cfg = PhotonicQRNNConfig(
        kd=1,
        kh=1,
        depth=1,
        shots=0,
        dtype=torch.float32,
        measurement_space="dual_rail",
    )
    model = PhotonicQRNNRegressor(input_size=2, config=cfg)

    x = torch.randn(2, 3, 2)
    y = torch.randn(2)

    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    class _CountingAdam(torch.optim.Adam):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.step_calls = 0

        def step(self, closure=None):  # type: ignore[override]
            self.step_calls += 1
            return super().step(closure=closure)

    opt = _CountingAdam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()
    device = torch.device("cpu")

    train_one_epoch(model, loader, crit, opt, device)
    # Default is per-sample stepping for photonic models when batch_size > 1.
    assert opt.step_calls == 2


def _has_pennylane() -> bool:
    try:
        import pennylane as _  # noqa: F401

        return True
    except Exception:
        return False


def test_gate_pqrnn_forward_and_backward():
    if not _has_pennylane():
        return

    from lib.gate_pqrnn import GatePQRNNConfig, GatePQRNNRegressor

    torch.manual_seed(0)

    cfg = GatePQRNNConfig(
        n_data=2,
        n_hidden=1,
        depth=1,
        entangling="nn",
        dtype=torch.float64,
    )
    model = GatePQRNNRegressor(input_size=2, config=cfg)

    x = torch.randn(2, 3, 2, dtype=torch.float64)
    y = model(x)
    assert y.shape == (2,)

    loss = y.sum()
    loss.backward()
    assert model.cell.readout.weight.grad is not None


def test_feature_normalization_minmax_signed_keeps_range():
    from lib.data import build_dataloaders

    cfg = {
        "dtype": "float64",
        "dataset": {
            "generator": {
                "name": "sin",
                "params": {
                    "frequency": 0.2,
                    "amplitude": 1.0,
                    "phase": 0.0,
                    "t_max": 10,
                    "n_points": 51,
                    "noise_std": 0.0,
                },
                "feature_dim": 1,
            },
            "sequence_length": 6,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "batch_size": 8,
            "shuffle": False,
            "feature_normalization": "minmax_-1_1",
        },
    }

    train_loader, _val_loader, _test_loader, _metadata = build_dataloaders(cfg)
    x, _y = next(iter(train_loader))
    assert torch.isfinite(x).all()
    assert x.min().item() >= -1.0001
    assert x.max().item() <= 1.0001


def _has_perceval_and_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        import perceval  # noqa: F401

        return True
    except Exception:
        return False


def test_qrb_circuit_png_export(tmp_path):
    if not _has_perceval_and_matplotlib():
        return

    from lib.perceval_export import export_qrb_circuit_svg

    svg = export_qrb_circuit_svg(run_dir=tmp_path, kd=1, kh=0)
    assert svg is not None
    assert svg.exists()
    assert svg.stat().st_size > 0
    assert "<svg" in svg.read_text(encoding="utf-8")
