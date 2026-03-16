from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from .angle_encoding import apply_input_encoding

logger = logging.getLogger(__name__)


@dataclass
class GatePQRNNConfig:
    """Gate-based pQRNN config (PennyLane implementation).

    This implements the pQRNN described in papers/QRNN/QRNN.md:
    - Data register D: n_data qubits (angle encoding with RY)
    - History register H: n_hidden qubits (recurrent state)
    - Ansatz: hardware-efficient layers with RX-RZ-RX per qubit + entangling IsingZZ
    - Partial measurement readout: use p(1) on the first qubit of D

    Notes
    -----
    - We propagate the hidden state as a density matrix rho_H (exact, no branching).
    - D is reset every time step by re-preparing |0>^n_data (handled implicitly).
    """

    n_data: int
    n_hidden: int
    depth: int = 1
    entangling: str = "cb"  # cb|nn|all
    entangling_wrap: bool = True
    dtype: torch.dtype | None = None
    input_encoding: str = "identity"  # identity|arccos


def _zero_density_matrix(n_qubits: int, *, complex_dtype: torch.dtype) -> torch.Tensor:
    dim = 2**n_qubits
    rho = torch.zeros(dim, dim, dtype=complex_dtype)
    rho[0, 0] = 1.0 + 0.0j
    return rho


class GatePQRNNCell(nn.Module):
    def __init__(self, input_size: int, config: GatePQRNNConfig) -> None:
        super().__init__()

        self.input_size = int(input_size)
        self.config = config

        self._dtype = config.dtype if config.dtype is not None else torch.float64
        self._complex_dtype = (
            torch.complex64 if self._dtype == torch.float32 else torch.complex128
        )

        if int(config.depth) <= 0:
            raise ValueError("depth must be positive")
        if int(config.n_data) <= 0:
            raise ValueError("n_data must be positive")
        if int(config.n_hidden) < 0:
            raise ValueError("n_hidden must be non-negative")

        self.n_data = int(config.n_data)
        self.n_hidden = int(config.n_hidden)
        self.n_total = self.n_data + self.n_hidden

        if self.input_size > self.n_data:
            raise ValueError(
                f"Gate pQRNN expects input_size<=n_data={self.n_data}, got {self.input_size}"
            )
        if self.input_size < self.n_data:
            logger.warning(
                "Gate pQRNN received input_size=%d but requires n_data=%d; will pad inputs with zeros.",
                self.input_size,
                self.n_data,
            )

        entangling = str(config.entangling).strip().lower()
        if entangling not in {"cb", "nn", "all"}:
            raise ValueError("entangling must be one of: 'cb', 'nn', 'all'")
        self.entangling = entangling
        self.entangling_wrap = bool(config.entangling_wrap)

        # Trainable parameters.
        depth = int(config.depth)

        # Single-qubit rotations: RX-RZ-RX per qubit per layer.
        self.rot = nn.Parameter(
            0.01 * torch.randn(depth, self.n_total, 3, dtype=self._dtype)
        )

        # IsingZZ entanglers per layer.
        edges = self._edges()
        self._n_edges = len(edges)
        self.zz = nn.Parameter(
            0.01 * torch.randn(depth, self._n_edges, dtype=self._dtype)
        )

        # Readout: y_t = Linear(p1)
        self.readout = nn.Linear(1, 1, dtype=self._dtype)

        # Initial hidden density matrix.
        if self.n_hidden > 0:
            rho_h0 = _zero_density_matrix(
                self.n_hidden, complex_dtype=self._complex_dtype
            )
        else:
            rho_h0 = torch.ones(1, 1, dtype=self._complex_dtype)
        self.register_buffer("_rho_h0", rho_h0)

        # Lazy PennyLane init.
        self._qnode = None

    def _edges(self) -> list[tuple[int, int]]:
        if self.n_total <= 1:
            return []
        if self.entangling == "all":
            return [
                (i, j) for i in range(self.n_total) for j in range(i + 1, self.n_total)
            ]
        # Nearest-neighbor chain on the full register.
        edges = [(i, i + 1) for i in range(self.n_total - 1)]
        # CB connectivity (paper): chain + optional wrap-around (0, N-1).
        if self.entangling == "cb" and self.entangling_wrap and self.n_total > 2:
            edges.append((0, self.n_total - 1))
        return edges

    def _pad_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.shape[-1] == self.n_data:
            return x_batch
        if x_batch.shape[-1] > self.n_data:
            raise ValueError(
                f"Gate pQRNN expects at most {self.n_data} features, got {int(x_batch.shape[-1])}"
            )
        pad = x_batch.new_zeros(
            (x_batch.shape[0], self.n_data - int(x_batch.shape[-1]))
        )
        return torch.cat([x_batch, pad], dim=-1)

    def _build_qnode(self):
        try:
            import pennylane as qml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "gate_pqrnn model requires PennyLane. Install it (e.g. pip install pennylane)."
            ) from exc

        dev = qml.device("default.mixed", wires=self.n_total)

        d_wires = list(range(self.n_data))
        h_wires = list(range(self.n_data, self.n_total))
        edges = self._edges()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(
            x_angles: torch.Tensor,
            rho_full: torch.Tensor,
            rot: torch.Tensor,
            zz: torch.Tensor,
        ):
            # Prepare the full-system density matrix on all wires.
            qml.QubitDensityMatrix(rho_full, wires=list(range(self.n_total)))

            # Angle encoding on D.
            for i, w in enumerate(d_wires):
                qml.RY(x_angles[i], wires=w)

            # Ansatz.
            for layer in range(rot.shape[0]):
                for w in range(self.n_total):
                    qml.RX(rot[layer, w, 0], wires=w)
                    qml.RZ(rot[layer, w, 1], wires=w)
                    qml.RX(rot[layer, w, 2], wires=w)

                for e_idx, (a, b) in enumerate(edges):
                    # Paper gate: Rzz(theta) = exp(i theta Z⊗Z)
                    # PennyLane: IsingZZ(phi) = exp(-i phi/2 Z⊗Z)
                    # => phi = -2 * theta
                    qml.IsingZZ(-2.0 * zz[layer, e_idx], wires=[a, b])

            exp_z = qml.expval(qml.PauliZ(d_wires[0]))
            rho_h_out = (
                qml.density_matrix(wires=h_wires)
                if self.n_hidden > 0
                else qml.density_matrix(wires=[])
            )
            return exp_z, rho_h_out

        self._qnode = circuit

    def forward(
        self, x: torch.Tensor, rho_h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x_batch = x.unsqueeze(0)
        elif x.ndim == 2:
            x_batch = x
        else:
            raise ValueError(f"Expected x to be 1D or 2D, got {tuple(x.shape)}")

        if x_batch.shape[0] != 1:
            raise ValueError("Gate pQRNN cell expects batch_size==1")

        x_batch = self._pad_features(x_batch)
        x_angles = x_batch[0].to(dtype=self._dtype)
        x_angles = apply_input_encoding(x_angles, self.config.input_encoding)

        if self._qnode is None:
            self._build_qnode()
        assert self._qnode is not None

        # Build the full density matrix: |0><0|_D ⊗ rho_H.
        dim_d = 2**self.n_data
        rho_d0 = torch.zeros(
            dim_d, dim_d, dtype=self._complex_dtype, device=rho_h.device
        )
        rho_d0[0, 0] = 1.0 + 0.0j
        rho_full = torch.kron(rho_d0, rho_h.to(dtype=self._complex_dtype))

        exp_z, rho_h_out = self._qnode(x_angles, rho_full, self.rot, self.zz)

        # Convert <Z> to p(1) for the first data qubit: p1 = (1 - <Z>)/2.
        p1 = (1.0 - exp_z) / 2.0
        y = self.readout(p1.view(1, 1))

        if self.n_hidden == 0:
            # Keep a trivial density matrix.
            rho_h_next = rho_h
        else:
            rho_h_next = rho_h_out.to(dtype=self._complex_dtype)

        return y, rho_h_next


class GatePQRNNRegressor(nn.Module):
    """Unrolled gate-based pQRNN regressor.

    Returns a scalar prediction based on the last timestep (forecasting).
    """

    def __init__(self, input_size: int, config: GatePQRNNConfig) -> None:
        super().__init__()
        self.cell = GatePQRNNCell(input_size, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape (batch, seq_len, feat), got {tuple(x.shape)}"
            )

        batch, seq_len, feat = x.shape
        if feat > self.cell.n_data:
            raise ValueError(
                f"Expected feature dim <= n_data={self.cell.n_data}, got {feat}"
            )

        outputs: list[torch.Tensor] = []
        for b in range(batch):
            rho_h = self.cell._rho_h0
            y_b = None
            for t in range(seq_len):
                y_b, rho_h = self.cell(x[b, t, :], rho_h)
            assert y_b is not None
            outputs.append(y_b)

        y = torch.cat(outputs, dim=0)
        return y.squeeze(-1)


__all__ = ["GatePQRNNConfig", "GatePQRNNCell", "GatePQRNNRegressor"]
