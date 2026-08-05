from __future__ import annotations

from lib.structured_circuits import (
    butterfly_param_count,
    butterfly_spec,
    make_butterfly_mzi_circuit,
)


def test_butterfly_spec_matches_radix_two_schedule() -> None:
    spec = butterfly_spec(8)
    assert spec.n_stages == 3
    assert spec.pairings == [
        [(0, 1), (2, 3), (4, 5), (6, 7)],
        [(0, 2), (1, 3), (4, 6), (5, 7)],
        [(0, 4), (1, 5), (2, 6), (3, 7)],
    ]


def test_make_butterfly_mzi_circuit_builds_expected_parameter_count() -> None:
    circuit = make_butterfly_mzi_circuit(8, prefix="T")
    assert circuit.m == 8
    assert len(circuit.get_parameters()) == butterfly_param_count(8)
