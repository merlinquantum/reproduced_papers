"""The data pipeline must follow upstream ``Dataset_Custom`` exactly."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest
import torch

from papers.shared.QuLTSF.data import build_dataloader, build_dataset, load_series


def _cfg(n_rows=200, seq_len=12, pred_len=4, features_mode="M", n_channels=3):
    return {
        "dataset": {
            "source": "synthetic",
            "name": "synthetic",
            "synthetic": {"n_rows": n_rows, "n_channels": n_channels, "seed": 0},
            "time_column": "date",
            "target": "OT",
            "features_mode": features_mode,
            "sequence_length": seq_len,
            "prediction_horizon": pred_len,
            "batch_size": 8,
            "train_ratio": 0.7,
            "test_ratio": 0.2,
            "val_drop_last": True,
            "test_drop_last": False,
        }
    }


def test_borders_follow_upstream_formula():
    cfg = _cfg(n_rows=200, seq_len=12)
    series = load_series(cfg)
    num_train, num_test = int(200 * 0.7), int(200 * 0.2)
    num_val = 200 - num_train - num_test
    assert series.border_starts == (0, num_train - 12, 200 - num_test - 12)
    assert series.border_ends == (num_train, num_train + num_val, 200)


def test_window_counts_and_contents():
    cfg = _cfg(n_rows=200, seq_len=12, pred_len=4)
    series = load_series(cfg)
    test = build_dataset(series, cfg, "test", torch.float64)
    assert len(test) == (200 - series.border_starts[2]) - 12 - 4 + 1
    inputs, targets = test[0]
    start = series.border_starts[2]
    np.testing.assert_allclose(inputs.numpy(), series.values[start : start + 12])
    np.testing.assert_allclose(targets.numpy(), series.values[start + 12 : start + 16])
    # The first test target row is the first row after the validation split.
    assert start + 12 == series.border_ends[1]


def test_scaler_is_fitted_on_training_rows_only():
    series = load_series(_cfg(n_rows=300))
    train = series.values[: series.border_ends[0]]
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-12)
    assert not np.allclose(series.values.mean(axis=0), 0.0, atol=1e-3)


@pytest.mark.parametrize(("mode", "channels"), [("M", 3), ("MS", 3), ("S", 1)])
def test_feature_modes_put_target_last(mode, channels):
    series = load_series(_cfg(features_mode=mode))
    assert len(series.columns) == channels
    assert series.columns[-1] == "OT"


def test_loaders_follow_upstream_batching():
    cfg = _cfg(n_rows=400)
    series = load_series(cfg)
    train = build_dataloader(series, cfg, "train", torch.float32)
    test = build_dataloader(series, cfg, "test", torch.float32)
    assert train.drop_last and not test.drop_last
    assert sum(batch[0].shape[0] for batch in test) == len(test.dataset)


def test_window_batches_match_torch_dataloader_exactly():
    """Same batches, same order and same global-RNG use as DataLoader."""
    from torch.utils.data import DataLoader

    cfg = _cfg(n_rows=600, seq_len=12, pred_len=4)
    series = load_series(cfg)
    train_set = build_dataset(series, cfg, "train", torch.float32)
    val_set = build_dataset(series, cfg, "val", torch.float32)

    def passes(make_train, make_val):
        torch.manual_seed(2021)
        out = []
        train, val = make_train(), make_val()
        for _ in range(3):  # epochs: a shuffled train pass, then a val pass
            out.extend(torch.cat([x.flatten(), y.flatten()]) for x, y in train)
            out.extend(torch.cat([x.flatten(), y.flatten()]) for x, y in val)
        out.append(torch.rand(3))  # the global RNG must end in the same state
        return out

    reference = passes(
        lambda: DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True),
        lambda: DataLoader(val_set, batch_size=8, shuffle=False, drop_last=True),
    )
    fast = passes(
        lambda: build_dataloader(series, cfg, "train", torch.float32),
        lambda: build_dataloader(series, cfg, "val", torch.float32),
    )
    assert len(reference) == len(fast)
    for expected, actual in zip(reference, fast, strict=True):
        torch.testing.assert_close(actual, expected)


def test_missing_dataset_file_fails_clearly(tmp_path):
    cfg = _cfg()
    cfg["dataset"].update(source="csv", path="missing.csv")
    cfg["data_root"] = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_series(cfg)
