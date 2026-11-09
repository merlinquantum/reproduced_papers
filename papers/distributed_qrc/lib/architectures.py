"""The four architectures of Table I and the configurations of Tables IV-VII (and C1-C12).

* SRSR: one reservoir of N neurons, one ridge (n = 10 kernel qubits).
* MRSR: k reservoirs of n neurons each, outputs concatenated (Eq. 4), one ridge
  (10 qubits).
* SRMR: one reservoir of N neurons, m ridge instances, each fed a contiguous
  chunk of the features.
* MRMR: k reservoirs, reservoir i feeding ridge instance i.

For SRMR and MRMR, kernel instances use 5 qubits when the total neuron count is
10, 15 or 25 and 10 qubits otherwise (Sec. II-B).
"""

from __future__ import annotations

from dataclasses import dataclass

ARCHITECTURES = ("srsr", "mrsr", "srmr", "mrmr")
SINGLE_READOUT_QUBITS = 10
SMALL_TOTALS = (10, 15, 25)


@dataclass(frozen=True)
class CellSpec:
    """Structure of one architecture configuration (one row of Tables IV-VII)."""

    architecture: str
    label: str
    reservoir_sizes: tuple[int, ...]
    n_instances: int
    kernel_qubits: int

    @property
    def total_neurons(self) -> int:
        return sum(self.reservoir_sizes)


def _parse_kxn(size: str) -> tuple[int, int]:
    count, neurons = size.lower().split("x")
    return int(count), int(neurons)


def build_cell_spec(architecture: str, size: int | str | list[int]) -> CellSpec:
    """Build a :class:`CellSpec` from a config entry.

    ``size`` is ``N`` for SRSR, ``"kxn"`` for MRSR and MRMR, and ``[N, m]`` for SRMR.
    """

    if architecture == "srsr":
        n = int(size)
        return CellSpec("srsr", str(n), (n,), 1, SINGLE_READOUT_QUBITS)
    if architecture == "mrsr":
        k, n = _parse_kxn(str(size))
        return CellSpec("mrsr", f"{k}x{n}", (n,) * k, 1, SINGLE_READOUT_QUBITS)
    if architecture == "srmr":
        n, m = (int(v) for v in size)
        qubits = 5 if n in SMALL_TOTALS else SINGLE_READOUT_QUBITS
        return CellSpec("srmr", f"{n}/{m}", (n,), m, qubits)
    if architecture == "mrmr":
        k, n = _parse_kxn(str(size))
        qubits = 5 if k * n in SMALL_TOTALS else SINGLE_READOUT_QUBITS
        return CellSpec("mrmr", f"{k}x{n}", (n,) * k, k, qubits)
    raise ValueError(
        f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
    )
