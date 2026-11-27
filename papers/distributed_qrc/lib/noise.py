"""Device noise models rebuilt from the paper's calibration medians (App. B).

The paper simulates ibm_brisbane, ibm_marrakesh and IonQ Aria-1 noise from QPU
calibration data but does not state how the noise model was built (see the deviations in `docs/reproduction_details.md`). This
module uses only the numbers printed in App. B:

* every native single-qubit gate is followed by a single-qubit depolarizing
  channel with average gate infidelity ``error_1q``;
* every two-qubit gate is followed by a two-qubit depolarizing channel with
  average gate infidelity ``error_2q``;
* measurement is a symmetric bit flip with probability ``readout_error``.

Randomized-benchmarking errors already include T1/T2 decoherence during the
gate, so relaxation is not added on top, and idle errors are neglected.

Gate counts per logical gate (applied identically on every device):

* ``RX(theta)``: 2 native single-qubit gates (RZ.SX.RZ.SX.RZ on IBM);
* ``RZ`` and the phase gate ``P``: virtual, error free;
* ``H``: 1 native single-qubit gate (RZ.SX.RZ);
* ``CX``: 1 native two-qubit gate plus 1 native single-qubit gate on each qubit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NATIVE_1Q_PER_RX = 2
NATIVE_1Q_PER_H = 1
NATIVE_1Q_PER_CX_QUBIT = 1


@dataclass(frozen=True)
class DeviceNoise:
    """Median error rates of one device (App. B)."""

    name: str
    error_1q: float
    error_2q: float
    readout_error: float

    @property
    def is_ideal(self) -> bool:
        return self.error_1q == 0 and self.error_2q == 0 and self.readout_error == 0

    @property
    def bloch_shrink_1q(self) -> float:
        """Bloch-vector contraction of one noisy native single-qubit gate.

        A d-dimensional depolarizing channel with average infidelity r has
        parameter ``lambda = 1 - d r / (d - 1)``; for d = 2 that is ``1 - 2 r``.
        """

        return 1.0 - 2.0 * self.error_1q

    @property
    def depolarizing_2q(self) -> float:
        """Two-qubit depolarizing parameter ``lambda = 1 - 4 r / 3``."""

        return 1.0 - 4.0 * self.error_2q / 3.0


IDEAL = DeviceNoise("ideal", 0.0, 0.0, 0.0)


def device_from_config(
    name: str, devices_cfg: dict[str, Any], scale: float = 1.0
) -> DeviceNoise:
    """Build a :class:`DeviceNoise` from ``cfg["noise"]["devices"][name]``.

    ``scale`` multiplies every error rate (gap diagnosis of the noise strength;
    1.0 reproduces the App. B medians).
    """

    if name == "ideal":
        return IDEAL
    if name not in devices_cfg:
        raise KeyError(f"Unknown noise device '{name}'; known: {sorted(devices_cfg)}")
    entry = devices_cfg[name]
    for key in ("error_1q", "error_2q", "readout_error"):
        if key not in entry:
            raise KeyError(f"Missing 'noise.devices.{name}.{key}'")
    rates = [scale * float(entry[k]) for k in ("error_1q", "error_2q", "readout_error")]
    if not all(0.0 <= r < 0.5 for r in rates):
        raise ValueError(f"Scaled error rates {rates} of '{name}' must lie in [0, 0.5)")
    return DeviceNoise(name, *rates)


# Table III of the paper: how components are spread over the three QPU noise
# models. A single reservoir or a single ridge instance runs on IBM Brisbane.
PAPER_DEVICE_MAPPING = {
    1: ["ibm_brisbane"],
    2: ["ibm_marrakesh", "ibm_brisbane"],
    3: ["ibm_marrakesh", "ibm_brisbane", "ionq_aria1"],
    5: ["ibm_marrakesh", "ibm_marrakesh", "ibm_brisbane", "ibm_brisbane", "ionq_aria1"],
}


def devices_for_components(count: int, mapping: str) -> list[str]:
    """Device names for ``count`` reservoirs or ridge instances.

    ``mapping`` is ``"paper"`` (Table III) or a single device name used for every
    component (the same-device noise side check).
    """

    if mapping == "paper":
        if count not in PAPER_DEVICE_MAPPING:
            raise ValueError(f"Table III has no mapping for {count} components")
        return list(PAPER_DEVICE_MAPPING[count])
    return [mapping] * count
