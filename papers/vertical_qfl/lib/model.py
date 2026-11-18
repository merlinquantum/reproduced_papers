"""Vertical quantum federated learning model (Ballester et al., ICEA '25).

Every party (client or server) runs an 8-qubit data re-uploading circuit. The
clients each see a disjoint slice of the 256 pixels; the server concatenates the
clients' expectation values, re-uploads them in its own circuit and reads out the
ten class scores. Circuit details that the paper leaves open are taken from the
authors' released code (see LOG.md, "Gaps resolved").
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import statevector as sv

N_BLOCKS = 5
DEPTH_PER_BLOCK = 100
CLIENT_INPUTS = 32
CLIENT_OUTPUTS = 8
N_CLASSES = 10

ENCODINGS = {
    # one pixel per wire per block, alternating RX / RY, 4 blocks -> 32 inputs
    "client": CLIENT_INPUTS,
    # RX layer then RY layer per block (16 values), 4 blocks -> 64 inputs
    "server_wide": 64,
    # the same 8 values re-uploaded with RX in each of the 4 blocks
    "server_narrow": CLIENT_OUTPUTS,
}
READOUTS = {"client": "z8", "server_wide": "zx5", "server_narrow": "zx5"}


class ReuploadingCircuit(nn.Module):
    """One party's 8-qubit data re-uploading circuit.

    Parameters
    ----------
    encoding : str
        One of ``"client"``, ``"server_wide"``, ``"server_narrow"``.
    n_blocks : int
        Number of blocks D; data is encoded after each of the first D-1 blocks.
    depth_per_block : int
        Trainable layers per block L. Each layer is an RX layer, an RY layer and a
        cyclic CZ ring, giving ``D * L * 2 * 8`` trainable angles.
    dtype : torch.dtype
        Real dtype of the angles; the state uses the matching complex dtype.
    """

    def __init__(
        self,
        encoding: str,
        n_blocks: int = N_BLOCKS,
        depth_per_block: int = DEPTH_PER_BLOCK,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if encoding not in ENCODINGS:
            raise ValueError(
                f"Unknown encoding {encoding!r}; expected {list(ENCODINGS)}"
            )
        self.encoding = encoding
        self.n_blocks = n_blocks
        self.depth_per_block = depth_per_block
        self.n_inputs = ENCODINGS[encoding]
        self.readout = READOUTS[encoding]
        # Angles initialised uniformly in [0, pi), as in the authors' code.
        self.angles = nn.Parameter(
            torch.rand(n_blocks * depth_per_block, 2, sv.N_WIRES, dtype=dtype) * math.pi
        )
        self.register_buffer("cz_signs", sv.cz_ring_signs())

    @property
    def n_outputs(self) -> int:
        return CLIENT_OUTPUTS if self.readout == "z8" else N_CLASSES

    def _encoding_gates(self, block: int, x: torch.Tensor) -> torch.Tensor:
        """Per-sample single-qubit gates, shape ``(batch, 8, 2, 2)``."""
        wires = sv.N_WIRES
        if self.encoding == "client":
            values = x[:, block * wires : (block + 1) * wires]
            return sv.rx_matrix(values) if block % 2 == 0 else sv.ry_matrix(values)
        if self.encoding == "server_wide":
            first = x[:, 2 * block * wires : (2 * block + 1) * wires]
            second = x[:, (2 * block + 1) * wires : (2 * block + 2) * wires]
            return sv.ry_matrix(second) @ sv.rx_matrix(first)
        return sv.rx_matrix(x[:, :wires])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, n_inputs)`` real inputs to expectation values."""
        if x.shape[1] != self.n_inputs:
            raise ValueError(
                f"{self.encoding} circuit expects {self.n_inputs} inputs, got {x.shape[1]}"
            )
        x = x.to(self.angles.dtype)
        state = sv.zero_state(
            x.shape[0], sv.complex_dtype_for(self.angles.dtype), x.device
        )
        depth = self.depth_per_block
        for block in range(self.n_blocks - 1):
            layers = self.angles[block * depth : (block + 1) * depth]
            state = sv.run_trainable_layers(
                state, layers, self.cz_signs, cz_after_last=True
            )
            state = sv.apply_wire_gates(state, self._encoding_gates(block, x))
            state = sv.apply_cz_ring(state, self.cz_signs)
        final_layers = self.angles[(self.n_blocks - 1) * depth :]
        state = sv.run_trainable_layers(
            state, final_layers, self.cz_signs, cz_after_last=False
        )
        if self.readout == "z8":
            return sv.expectation_z(state, list(range(sv.N_WIRES)))
        # Ten class scores: <Z> then <X> on wires 0-4 (paper: 10 classes from 8 qubits).
        wires = list(range(5))
        return torch.cat(
            [sv.expectation_z(state, wires), sv.expectation_x(state, wires)], dim=1
        )


class VerticalQuantumFederatedModel(nn.Module):
    """Clients on disjoint pixel slices plus a quantum server.

    Parameters
    ----------
    client_slices : list of int
        For each client, the index of the 32-column slice of the (permuted) pixel
        vector it sees. ``[0..7]`` is the 8-client model; repeating an index
        reproduces the paper's duplicated-client controls.
    server_encoding : str
        ``"server_wide"`` (64 inputs) or ``"server_narrow"`` (8 inputs).
    softmax_output : bool
        The authors' code applies softmax to the server output and then
        ``CrossEntropyLoss`` (softmax twice). ``True`` keeps that behaviour.
    """

    def __init__(
        self,
        client_slices: list[int],
        server_encoding: str,
        softmax_output: bool,
        n_blocks: int = N_BLOCKS,
        depth_per_block: int = DEPTH_PER_BLOCK,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if ENCODINGS[server_encoding] != CLIENT_OUTPUTS * len(client_slices):
            # The narrow server of the 1-client experiment takes the 8 outputs of that client.
            raise ValueError(
                f"Server encoding {server_encoding!r} takes {ENCODINGS[server_encoding]} inputs "
                f"but {len(client_slices)} clients give {CLIENT_OUTPUTS * len(client_slices)}"
            )
        self.client_slices = list(client_slices)
        self.softmax_output = softmax_output
        self.clients = nn.ModuleList(
            ReuploadingCircuit("client", n_blocks, depth_per_block, dtype)
            for _ in client_slices
        )
        self.server = ReuploadingCircuit(
            server_encoding, n_blocks, depth_per_block, dtype
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Class scores for ``(batch, 256)`` angle-encoded, permuted pixels."""
        outputs = []
        for client, slice_index in zip(self.clients, self.client_slices):
            start = slice_index * CLIENT_INPUTS
            outputs.append(client(pixels[:, start : start + CLIENT_INPUTS]))
        scores = self.server(torch.cat(outputs, dim=1))
        return torch.softmax(scores, dim=1) if self.softmax_output else scores

    def trainable_parameter_counts(self) -> dict[str, int]:
        """Trainable angles per party and in total."""
        counts = {f"client_{i}": c.angles.numel() for i, c in enumerate(self.clients)}
        counts["server"] = self.server.angles.numel()
        counts["total"] = sum(counts.values())
        return counts
