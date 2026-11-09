"""Forecasting models: QuLTSF, the LTSF-Linear baselines and classical controls.

Every model maps ``x [batch, L, channels]`` to ``[batch, T, channels]`` and is
channel-independent: the same weights are applied to each channel's length-L
history (the LTSF-Linear and QuLTSF convention, ``individual=False``).

Models
------
``qultsf``
    Paper Sec. 4 / upstream ``models/QuLTSF.py``: ``Linear(L -> 2**N)``,
    amplitude embedding on N qubits, K StronglyEntanglingLayers, ``<Z_i>`` on
    every qubit, ``Linear(N -> T)``. ``model.params.circuit`` selects the
    paper's trainable circuit, a frozen random circuit, or the circuit-free
    ``identity`` control. ``model.params.simulator`` selects the batched
    PyTorch simulator (default) or the PennyLane reference port.
``linear``, ``nlinear``, ``dlinear``
    Ports of upstream ``models/Linear.py``, ``NLinear.py`` and ``DLinear.py``
    (moving-average kernel 25, shared across channels).
``persistence``
    Repeats the last observed value over the horizon. No parameters.
``rank_bottleneck``
    ``Linear(L -> r) -> Linear(r -> T)``: QuLTSF's skeleton with the quantum
    layer and the 2**N expansion removed, a rank-r linear forecaster.
``mlp``
    ``Linear(L -> h) -> GELU -> Linear(h -> T)``, sized to a parameter budget.
``qultsf_merlin``
    Phase 4 photonic adaptation (not the paper's model): the qubit hidden
    layer replaced by an amplitude-encoded n-photon state in m modes, a
    MerLin interferometer and per-mode threshold readout; see
    :mod:`lib.qultsf_merlin`. MerLin is imported only when this model is built.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantum_simulator import AmplitudeSELayer


def _channel_independent(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply ``module`` (``[N, L] -> [N, T]``) to every channel of ``x``."""
    batch, seq_len, channels = x.shape
    rows = x.permute(0, 2, 1).reshape(batch * channels, seq_len)
    out = module(rows)
    return out.reshape(batch, channels, -1).permute(0, 2, 1)


class QuLTSF(nn.Module):
    """Hybrid QuLTSF model (paper Eq. 2-4).

    Parameters
    ----------
    seq_len : int
        Lookback length L.
    pred_len : int
        Forecast horizon T.
    num_qubits : int
        Number of qubits N (``2**N`` amplitudes).
    num_layers : int
        Number of StronglyEntanglingLayers K.
    circuit : str
        ``"trainable"``, ``"frozen"`` or ``"identity"``; see
        :class:`lib.quantum_simulator.AmplitudeSELayer`.
    dtype : torch.dtype
        Parameter dtype.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_qubits: int,
        num_layers: int,
        circuit: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        # Creation order matches upstream (input layer, quantum weights, output
        # layer) so a given seed draws the same initial values.
        self.input_classical_layer = nn.Linear(seq_len, 2**num_qubits, dtype=dtype)
        self.hidden_quantum_layer = AmplitudeSELayer(
            num_qubits, num_layers, circuit=circuit, dtype=dtype
        )
        self.output_classical_layer = nn.Linear(num_qubits, pred_len, dtype=dtype)

    def _rows(self, rows: torch.Tensor) -> torch.Tensor:
        hidden = self.input_classical_layer(rows)
        hidden = self.hidden_quantum_layer(hidden)
        return self.output_classical_layer(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _channel_independent(self._rows, x)


class PennyLaneQuLTSF(nn.Module):
    """QuLTSF with the upstream PennyLane QNode as the hidden layer.

    Reference artifact for checking :class:`QuLTSF`; slow, CPU only.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_qubits: int,
        num_layers: int,
        qml_device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        from .original_qultsf import OriginalQuLTSFConfig, OriginalQuLTSFModel

        self.model = OriginalQuLTSFModel(
            OriginalQuLTSFConfig(
                seq_len=seq_len,
                pred_len=pred_len,
                num_qubits=num_qubits,
                qml_device=qml_device,
                num_layers=num_layers,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Linear(nn.Module):
    """Upstream ``Linear``: one ``Linear(L -> T)`` shared across channels."""

    def __init__(self, seq_len: int, pred_len: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class NLinear(nn.Module):
    """Upstream ``NLinear``: subtract the last value, Linear, add it back."""

    def __init__(self, seq_len: int, pred_len: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_last = x[:, -1:, :].detach()
        out = self.Linear((x - seq_last).permute(0, 2, 1)).permute(0, 2, 1)
        return out + seq_last


class MovingAverage(nn.Module):
    """Upstream ``moving_avg`` with edge replication padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """Upstream ``DLinear`` (``individual=False``, kernel 25)."""

    def __init__(
        self, seq_len: int, pred_len: int, dtype: torch.dtype, kernel_size: int = 25
    ) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len, dtype=dtype)
        self.Linear_Trend = nn.Linear(seq_len, pred_len, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_avg(x)
        seasonal = x - trend
        out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)) + self.Linear_Trend(
            trend.permute(0, 2, 1)
        )
        return out.permute(0, 2, 1)


class Persistence(nn.Module):
    """Repeat the last observed value over the horizon."""

    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:, :].expand(-1, self.pred_len, -1)


class RankBottleneck(nn.Module):
    """``Linear(L -> r) -> Linear(r -> T)``, a rank-r linear forecaster."""

    def __init__(
        self, seq_len: int, pred_len: int, rank: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(seq_len, rank, dtype=dtype)
        self.decoder = nn.Linear(rank, pred_len, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _channel_independent(lambda rows: self.decoder(self.encoder(rows)), x)


class MLP(nn.Module):
    """``Linear(L -> h) -> GELU -> Linear(h -> T)``."""

    def __init__(
        self, seq_len: int, pred_len: int, hidden: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, pred_len, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _channel_independent(self.net, x)


def mlp_hidden_for_budget(seq_len: int, pred_len: int, parameter_budget: int) -> int:
    """Return the hidden width whose MLP parameter count is closest to a budget.

    The MLP has ``h (L + T + 1) + T`` parameters.
    """
    return max(1, round((parameter_budget - pred_len) / (seq_len + pred_len + 1)))


def qultsf_parameter_count(
    seq_len: int, pred_len: int, num_qubits: int, num_layers: int
) -> int:
    """Return the trainable parameter count of the paper's QuLTSF model."""
    dim = 2**num_qubits
    return (
        seq_len * dim
        + dim
        + 3 * num_qubits * num_layers
        + num_qubits * pred_len
        + pred_len
    )


def _param(params: dict, key: str):
    if key not in params:
        raise KeyError(f"Missing required config key model.params.{key}")
    return params[key]


def build_model(cfg: dict, dtype: torch.dtype) -> nn.Module:
    """Build the model named by ``cfg["model"]["name"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    dtype : torch.dtype
        Parameter dtype.

    Returns
    -------
    torch.nn.Module
        The model, on CPU.

    Raises
    ------
    KeyError
        If a required ``model.params`` key is missing.
    ValueError
        If the model name or simulator is unknown.
    """
    seq_len = int(cfg["dataset"]["sequence_length"])
    pred_len = int(cfg["dataset"]["prediction_horizon"])
    name = str(cfg["model"]["name"])
    params = cfg["model"].get("params", {})

    if name == "qultsf":
        simulator = _param(params, "simulator")
        num_qubits = int(_param(params, "num_qubits"))
        num_layers = int(_param(params, "num_layers"))
        if simulator == "torch":
            return QuLTSF(
                seq_len,
                pred_len,
                num_qubits,
                num_layers,
                circuit=str(_param(params, "circuit")),
                dtype=dtype,
            )
        if simulator == "pennylane":
            if _param(params, "circuit") != "trainable":
                raise ValueError(
                    "The PennyLane reference only supports circuit='trainable'"
                )
            return PennyLaneQuLTSF(
                seq_len,
                pred_len,
                num_qubits,
                num_layers,
                qml_device=str(_param(params, "qml_device")),
                dtype=dtype,
            )
        raise ValueError(
            f"model.params.simulator must be 'torch' or 'pennylane', got {simulator!r}"
        )
    if name == "qultsf_merlin":
        from .qultsf_merlin import QuLTSFMerlin

        return QuLTSFMerlin(
            seq_len,
            pred_len,
            n_modes=int(_param(params, "n_modes")),
            n_photons=int(_param(params, "n_photons")),
            interferometer=str(_param(params, "interferometer")),
            dtype=dtype,
        )
    if name == "linear":
        return Linear(seq_len, pred_len, dtype)
    if name == "nlinear":
        return NLinear(seq_len, pred_len, dtype)
    if name == "dlinear":
        return DLinear(
            seq_len, pred_len, dtype, kernel_size=int(_param(params, "kernel_size"))
        )
    if name == "persistence":
        return Persistence(pred_len)
    if name == "rank_bottleneck":
        return RankBottleneck(seq_len, pred_len, int(_param(params, "rank")), dtype)
    if name == "mlp":
        if "hidden" in params:
            hidden = int(params["hidden"])
        else:
            budget = qultsf_parameter_count(
                seq_len,
                pred_len,
                int(_param(params, "match_num_qubits")),
                int(_param(params, "match_num_layers")),
            )
            hidden = mlp_hidden_for_budget(seq_len, pred_len, budget)
        return MLP(seq_len, pred_len, hidden, dtype)
    raise ValueError(f"Unknown model.name {name!r}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "DLinear",
    "Linear",
    "MLP",
    "NLinear",
    "Persistence",
    "PennyLaneQuLTSF",
    "QuLTSF",
    "RankBottleneck",
    "build_model",
    "count_trainable_parameters",
    "mlp_hidden_for_budget",
    "qultsf_parameter_count",
]
