from __future__ import annotations

from dataclasses import dataclass

import perceval as pcvl


@dataclass(frozen=True)
class QRBCircuitSpec:
    """Specification of a QRB circuit.

    Notes
    -----
    - Total system: 2*m modes in dual-rail (m logical qubits / photons).
    - Measured subsystem (D register): first 2*k modes.
    """

    m: int
    k: int


def _mzi_with_params(prefix: str, idx: int) -> pcvl.Circuit:
    """Return a 2-mode MZI with two trainable phase shifters.

    Parameter names start with `prefix` so Merlin can bind them via trainable prefixes.
    """

    return (
        pcvl.Circuit(2)
        .add((0, 1), pcvl.BS())
        .add(0, pcvl.PS(pcvl.P(f"{prefix}a_{idx}")))
        .add((0, 1), pcvl.BS())
        .add(0, pcvl.PS(pcvl.P(f"{prefix}b_{idx}")))
    )


def _generic_interferometer(
    n_modes: int, *, prefix: str, stage: int
) -> pcvl.GenericInterferometer:
    """Create a rectangular GenericInterferometer with trainable parameters."""

    # The idx passed by Perceval is local to this interferometer.
    # We offset by stage to keep parameter names unique across blocks.
    offset = stage * (n_modes * (n_modes - 1) // 2)
    return pcvl.GenericInterferometer(
        n_modes,
        fun_gen=lambda idx: _mzi_with_params(prefix, offset + idx),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )


def build_qrb_circuit(spec: QRBCircuitSpec) -> pcvl.Circuit:
    """Build a QRB circuit compatible with Merlin's QuantumLayer.

    Layout
    ------
    - D register: modes [0 .. 2*k-1]
    - H register: modes [2*k .. 2*m-1]

    Circuit
    -------
    - W1 on D: interferometer(2k) with prefix "phi"
    - Encoding column on D: phase shifters (2k) with prefix "in" (expects dim(x)=2k)
    - W2 on D: interferometer(2k) with prefix "psi"
    - V on D+H: interferometer(2m) with prefix "theta"

    Notes
    -----
    This is a minimal, implementation-oriented circuit builder. It is designed so that:
    - input parameters (angle encoding) are exactly the 2k phase shifters named in_*;
    - trainable parameters are all interferometer parameters with prefixes phi/psi/theta.
    """

    m = int(spec.m)
    k = int(spec.k)
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > m:
        raise ValueError(f"k must be <= m, got k={k} m={m}")

    n_modes_total = 2 * m
    n_modes_d = 2 * k

    circuit = pcvl.Circuit(n_modes_total)

    # Sandwich interferometer on the measured subsystem (D)
    #
    # Note: we wrap each logical block in a named sub-circuit so that Perceval
    # renderers display a clear high-level structure: (phi) -> (encoding) -> (psi)
    # on the first 2*k modes, then (theta) on all 2*m modes.
    phi_block = pcvl.Circuit(n_modes_d, name="phi")
    phi_block.add(0, _generic_interferometer(n_modes_d, prefix="phi", stage=0))
    circuit.add(0, phi_block)

    encoding_block = pcvl.Circuit(n_modes_d, name="encoding")
    # Encoding PS column: dim(x)=2k
    for mode in range(n_modes_d):
        encoding_block.add(mode, pcvl.PS(pcvl.P(f"in_{mode}")))
    circuit.add(0, encoding_block)

    psi_block = pcvl.Circuit(n_modes_d, name="psi")
    psi_block.add(0, _generic_interferometer(n_modes_d, prefix="psi", stage=1))
    circuit.add(0, psi_block)

    # Global interferometer on all modes
    theta_block = pcvl.Circuit(n_modes_total, name="theta")
    theta_block.add(0, _generic_interferometer(n_modes_total, prefix="theta", stage=2))
    circuit.add(0, theta_block)

    return circuit
