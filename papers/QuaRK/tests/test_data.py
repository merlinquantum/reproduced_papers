from __future__ import annotations

import numpy as np
from common import load_runtime_ready_config
from QuaRK.lib.data import build_dataloader, generate_dataset


def test_build_dataloader_first_batch_shape():
    cfg = load_runtime_ready_config()
    windows, labels = next(iter(build_dataloader(cfg, split="train")))
    assert windows.shape == (256, 25, 3), windows.shape
    assert labels.shape == (4, 256), labels.shape


def test_forecast_label_is_next_point_and_split_is_disjoint():
    cfg = load_runtime_ready_config()
    dataset = generate_dataset(cfg["data"], 12345)
    u = np.asarray(dataset.meta["u_readout"])
    # The code's F1 reads the last point; the forecast variant applies the same u to
    # the point after the window, which lies in the gap and is not part of any window.
    np.testing.assert_allclose(dataset.labels[0], dataset.windows[:, -1, :] @ u)
    assert not np.allclose(dataset.labels[3], dataset.labels[0])
    assert set(dataset.train_idx).isdisjoint(dataset.test_idx)
    assert len(dataset.train_idx) + len(dataset.test_idx) == dataset.windows.shape[0]
