"""Qiskit feature maps and ansatzes with the author defaults.

Mirrors ``src/data/data_encoding.py`` and ``src/models/ansatzes.py`` of the
author repository (commit 53bc59a):

- ZFeatureMap / ZZFeatureMap: ``reps=2``.
- RealAmplitudes / EfficientSU2: ``reps=3``, entanglement from the grid.
- TwoLocal: RY rotations, CX entanglement, ``reps=3``.
- PauliTwoDesign: ``reps=3``, ``seed=3``; no entanglement option.

Entanglement options: ZZFeatureMap and TwoLocal take full, linear, circular,
pairwise or sca; RealAmplitudes and EfficientSU2 take full, linear, circular
or sca.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    EfficientSU2,
    PauliTwoDesign,
    RealAmplitudes,
    TwoLocal,
    ZFeatureMap,
    ZZFeatureMap,
)

FEATURE_MAPS = ("ZFeatureMap", "ZZFeatureMap")
ANSATZES = ("RealAmplitudes", "EfficientSU2", "TwoLocal", "PauliTwoDesign")
TWOLOCAL_ENTANGLEMENTS = ("full", "linear", "circular", "pairwise", "sca")
OTHER_ENTANGLEMENTS = ("full", "linear", "circular", "sca")


def build_feature_map(
    name: str, n_qubits: int, entanglement: str | None, reps: int = 2
) -> QuantumCircuit:
    """Return the data-encoding circuit.

    Parameters
    ----------
    name : str
        ``ZFeatureMap`` or ``ZZFeatureMap``.
    n_qubits : int
        One qubit per input feature.
    entanglement : str | None
        ZZFeatureMap entanglement; must be None for ZFeatureMap.
    reps : int
        Feature-map repetitions. Default value is 2 (author default).
    """

    if name == "ZFeatureMap":
        if entanglement is not None:
            raise ValueError("ZFeatureMap has no entanglement option")
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    if name == "ZZFeatureMap":
        if entanglement not in TWOLOCAL_ENTANGLEMENTS:
            raise ValueError(
                f"ZZFeatureMap entanglement must be one of {TWOLOCAL_ENTANGLEMENTS}"
            )
        return ZZFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )
    raise ValueError(f"unknown feature map {name!r}")


def build_ansatz(
    name: str, n_qubits: int, entanglement: str | None, reps: int = 3
) -> QuantumCircuit:
    """Return the trainable circuit.

    Parameters
    ----------
    name : str
        One of ``RealAmplitudes``, ``EfficientSU2``, ``TwoLocal``,
        ``PauliTwoDesign``.
    n_qubits : int
        Number of qubits.
    entanglement : str | None
        Entanglement strategy; must be None for PauliTwoDesign.
    reps : int
        Rotation/entanglement repetitions. Default value is 3 (author default).
    """

    if name == "PauliTwoDesign":
        if entanglement is not None:
            raise ValueError("PauliTwoDesign has no entanglement option")
        return PauliTwoDesign(num_qubits=n_qubits, reps=reps, seed=3)
    if name == "TwoLocal":
        if entanglement not in TWOLOCAL_ENTANGLEMENTS:
            raise ValueError(
                f"TwoLocal entanglement must be one of {TWOLOCAL_ENTANGLEMENTS}"
            )
        return TwoLocal(
            num_qubits=n_qubits,
            rotation_blocks=["ry"],
            entanglement_blocks=["cx"],
            entanglement=entanglement,
            reps=reps,
        )
    if entanglement not in OTHER_ENTANGLEMENTS:
        raise ValueError(f"{name} entanglement must be one of {OTHER_ENTANGLEMENTS}")
    if name == "RealAmplitudes":
        return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    if name == "EfficientSU2":
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    raise ValueError(f"unknown ansatz {name!r}")


def paper_grid() -> list[dict]:
    """Enumerate the 84 circuit configurations of the author ``get_configurations``.

    Returns
    -------
    list of dict
        Keys ``feature_map``, ``feature_map_entanglement``, ``ansatz``,
        ``ansatz_entanglement``, in the author's enumeration order
        (EfficientSU2, PauliTwoDesign, RealAmplitudes, TwoLocal), so the list
        position equals the ``index`` column of the released result CSVs.
    """

    configs = []
    for ansatz in ("EfficientSU2", "PauliTwoDesign", "RealAmplitudes", "TwoLocal"):
        ansatz_entanglements = (
            [None]
            if ansatz == "PauliTwoDesign"
            else TWOLOCAL_ENTANGLEMENTS
            if ansatz == "TwoLocal"
            else OTHER_ENTANGLEMENTS
        )
        for feature_map in FEATURE_MAPS:
            for ansatz_entanglement in ansatz_entanglements:
                map_entanglements = (
                    [None] if feature_map == "ZFeatureMap" else TWOLOCAL_ENTANGLEMENTS
                )
                for map_entanglement in map_entanglements:
                    configs.append(
                        {
                            "feature_map": feature_map,
                            "feature_map_entanglement": map_entanglement,
                            "ansatz": ansatz,
                            "ansatz_entanglement": ansatz_entanglement,
                        }
                    )
    return configs
