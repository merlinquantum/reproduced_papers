from __future__ import annotations

import pytest
import torch

# Importing `common` ensures the repo root and the paper directory are on
# `sys.path`, so `from lib...` imports work when running pytest from the repo root.
from common import PROJECT_DIR  # noqa: F401


def _has_merlin() -> bool:
    try:
        import importlib

        import merlin as ml

        if not hasattr(ml, "MeasurementStrategy"):
            return False
        if not hasattr(ml.MeasurementStrategy, "partial"):
            return False
        importlib.import_module("merlin.core.partial_measurement")
        importlib.import_module("merlin.core.state_vector")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_merlin(), reason="merlin is not available")


def test_build_qrb_circuit_has_expected_io_params():
    from lib.qrb_circuit import QRBCircuitSpec, build_qrb_circuit

    spec = QRBCircuitSpec(m=3, k=1)
    circuit = build_qrb_circuit(spec)
    assert circuit.m == 2 * spec.m

    names = [getattr(p, "name", "") for p in circuit.get_parameters()]

    # Exactly 2k input phase shifters: in_0..in_{2k-1}
    in_params = [n for n in names if n.startswith("in_")]
    assert sorted(in_params) == ["in_0", "in_1"]

    # Interferometer parameters are trainable and should exist.
    assert any(n.startswith("phi") for n in names)
    assert any(n.startswith("psi") for n in names)
    assert any(n.startswith("theta") for n in names)


def test_partial_measurement_to_dual_rail_probabilities():
    import merlin as ml
    from lib.qrb_readout import (
        DualRailReadoutSpec,
        partial_measurement_to_dual_rail_probabilities,
    )
    from merlin.core.partial_measurement import (
        PartialMeasurement,
        PartialMeasurementBranch,
    )
    from merlin.core.state_vector import StateVector

    k = 2
    # Dummy conditional state on 2 modes with 1 photon.
    amps = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
    dummy_state = StateVector(tensor=amps, n_modes=2, n_photons=1)

    b0 = PartialMeasurementBranch(
        outcome=(1, 0, 1, 0),
        probability=torch.tensor(0.7),
        amplitudes=dummy_state,
    )
    b1 = PartialMeasurementBranch(
        outcome=(1, 0, 0, 1),
        probability=torch.tensor(0.3),
        amplitudes=dummy_state,
    )

    pm = PartialMeasurement(
        branches=(b0, b1),
        measured_modes=(0, 1, 2, 3),
        unmeasured_modes=(4, 5),
    )

    probs = partial_measurement_to_dual_rail_probabilities(
        pm, spec=DualRailReadoutSpec(k=k)
    )
    assert probs.shape == (1, 2**k)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0], dtype=probs.dtype))

    comb = ml.Combinadics("dual_rail", n=k, m=2 * k)
    idx0 = comb.fock_to_index((1, 0, 1, 0))
    idx1 = comb.fock_to_index((1, 0, 0, 1))

    assert torch.isclose(probs[0, idx0], torch.tensor(0.7, dtype=probs.dtype))
    assert torch.isclose(probs[0, idx1], torch.tensor(0.3, dtype=probs.dtype))


def test_project_fock_amplitudes_to_dual_rail_one_qubit():
    from lib.qrb_readout import project_fock_amplitudes_to_dual_rail
    from merlin.core.state_vector import StateVector

    # n_qubits=1 => 2 modes, 1 photon, dual-rail size 2.
    # Put amplitude on |10> (photon in the first rail).
    state = StateVector(
        tensor=torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128),
        n_modes=2,
        n_photons=1,
    )

    projected = project_fock_amplitudes_to_dual_rail(state, n_qubits=1)
    assert projected.shape == (2,)
    norm = projected.abs().pow(2).sum()
    assert torch.allclose(norm, torch.tensor(1.0, dtype=norm.dtype))
