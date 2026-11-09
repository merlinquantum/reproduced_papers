"""Time-series data loading, min-max scaling and sliding-window splitting.

Ported from the original benchmark repository
(``utils/handling_data.py`` in tobias-fllnr/VariationalQMLTimeSeriesBenchmark).

The three chaotic datasets (Mackey-Glass 1D, Hénon 2D, Lorenz 3D) each contain
1000 points.  Every column is independently min-max scaled to ``[0, 1]`` and the
series is cut into overlapping windows

    x = [x_t, ..., x_{t+l-1}]   ->   y = x_{t+l+k-1}

with sequence length ``l`` and prediction step ``k``.  The first 60% of the
windows are used for training, the next 20% for validation and the final 20%
for testing, matching the paper (Section IV).

Note on normalisation
----------------------
The original code computes the per-column min/max over the *entire* 1000-point
series, i.e. including the validation and test segments.  This is a (mild) form
of leakage but it is applied identically to every model, so it does not bias the
quantum-vs-classical comparison.  We reproduce this behaviour faithfully and
document it in ``BUGS.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Number of feature dimensions per dataset label.
DATASET_DIM = {"mackey": 1, "henon": 2, "lorenz": 3}


def dataset_dim(data_label: str) -> int:
    """Return the feature dimension implied by a dataset label."""
    for prefix, dim in DATASET_DIM.items():
        if data_label.startswith(prefix):
            return dim
    raise ValueError(f"Unknown dataset label: {data_label!r}")


class DataHandling:
    """Load, scale and window one chaotic time-series dataset."""

    def __init__(
        self,
        data_label: str,
        seq_length: int,
        prediction_step: int,
        data_root: str | Path = "data/variational_qml_ts_benchmark",
        data_length: int = 1000,
    ) -> None:
        self.data_label = data_label
        self.seq_length = seq_length
        self.prediction_step = prediction_step
        self.data_length = data_length
        self.validation_size = 0.2
        self.test_size = 0.2

        base = data_label.split("_")[0]  # "henon_1000" -> "henon"
        self.file_path = Path(data_root) / f"{base}_{data_length}.csv"
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        self.data, self.min_values, self.max_values = self._load_data()

    def _load_data(self) -> tuple[pd.DataFrame, list, list]:
        data = pd.read_csv(self.file_path)
        data = data.head(self.data_length)
        min_values = [data[c].min() for c in data.columns]
        max_values = [data[c].max() for c in data.columns]
        return data, min_values, max_values

    def transform(self) -> pd.DataFrame:
        data = self.data.copy()
        for i, column in enumerate(self.data.columns):
            data[column] = (data[column] - self.min_values[i]) / (
                self.max_values[i] - self.min_values[i]
            )
        return data

    def get_training_and_test_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return (train_x, train_y, val_x, val_y, test_x, test_y) tensors."""
        data = self.transform()
        x, y = [], []
        for i in range(len(data) - self.seq_length - self.prediction_step):
            x.append(data.iloc[i : i + self.seq_length].values)
            y.append(data.iloc[i + self.seq_length + self.prediction_step - 1].values)

        split_val = int(len(x) * (1 - self.test_size - self.validation_size))
        split_test = int(len(x) * (1 - self.test_size))
        x, y = np.array(x), np.array(y)

        def t(a):
            return torch.tensor(a, dtype=torch.float32)

        return (
            t(x[:split_val]),
            t(y[:split_val]),
            t(x[split_val:split_test]),
            t(y[split_val:split_test]),
            t(x[split_test:]),
            t(y[split_test:]),
        )
