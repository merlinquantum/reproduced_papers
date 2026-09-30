from __future__ import annotations

import math
import sys

import torch
from common import PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib import mutations as mut  # noqa: E402
from lib.circuit_ir import (  # noqa: E402
    BLOCK_ANSATZ_MIX,
    BLOCK_ANSATZ_PHASE,
    build_original_ir,
    set_weights,
)
from lib.mutation_testing import (  # noqa: E402
    STATUS_KILLED,
    STATUS_SURVIVED,
    SuiteSpec,
    evaluate_mutant,
    summarise,
)
from lib.photonic_qnn import PhotonicQnnClassifier, PhotonicQnnSpec  # noqa: E402

N_MODES, N_LAYERS = 4, 3


def _trained_ir():
    weights = [0.3 * (index + 1) for index in range(N_MODES * N_LAYERS)]
    return set_weights(
        build_original_ir(n_modes=N_MODES, n_features=N_MODES, n_layers=N_LAYERS),
        weights,
    )


def _spec():
    return PhotonicQnnSpec(
        n_features=N_MODES,
        n_classes=3,
        n_modes=N_MODES,
        n_photons=2,
        n_layers=N_LAYERS,
        encoding_scale=math.pi,
        input_state=[1, 0, 1, 0],
    )


def test_operator_counts_follow_the_declared_formulas():
    """Mutant counts must match the structural formulas recorded in LOG.md."""
    ir = _trained_ir()
    counts = {
        "APC": len(mut.generate_apc(ir)),
        "DFC": len(mut.generate_dfc(N_MODES)),
        "APGC": len(mut.generate_apgc(ir)),
        "LS": len(mut.generate_ls(ir)),
        "ILS": len(mut.generate_ils(ir)),
        "ALA": len(mut.generate_ala(ir)),
        "ALD": len(mut.generate_ald(ir)),
    }
    assert counts["APC"] == 5 * N_MODES * N_LAYERS
    assert counts["DFC"] == 6 * 16  # C(4,2) pairs x 4x4 operation combinations
    assert counts["APGC"] == (2 * N_MODES - 2) * N_LAYERS
    assert counts["LS"] == N_LAYERS * (N_LAYERS - 1) // 2
    assert counts["ILS"] == N_LAYERS
    assert counts["ALA"] == 2 * N_LAYERS
    assert counts["ALD"] == 3 * N_LAYERS


def test_apc_layer_granularity_is_smaller_than_gate_granularity():
    ir = _trained_ir()
    assert len(mut.generate_apc(ir, granularity="layer")) == 5 * N_LAYERS
    assert len(mut.generate_apc(ir, granularity="gate")) == 5 * N_MODES * N_LAYERS


def test_ald_removes_the_expected_components():
    ir = _trained_ir()
    by_id = {mutant.mutant_id: mutant for mutant in mut.generate_ald(ir)}
    whole = by_id["ALD_L1_layer"].ir
    phases = by_id["ALD_L1_phases"].ir
    mixing = by_id["ALD_L1_mixing"].ir
    assert whole.positions(layer=1) == []
    assert phases.positions(block=BLOCK_ANSATZ_PHASE, layer=1) == []
    assert phases.positions(block=BLOCK_ANSATZ_MIX, layer=1) != []
    assert mixing.positions(block=BLOCK_ANSATZ_MIX, layer=1) == []


def test_ls_permutes_weights_without_changing_structure():
    ir = _trained_ir()
    mutant = mut.generate_ls(ir)[0].ir
    assert sorted(mutant.weights()) == sorted(ir.weights())
    assert mutant.weights() != ir.weights()
    assert len(mutant.elements) == len(ir.elements)


def test_apgc_turns_a_phase_shifter_into_a_beam_splitter():
    ir = _trained_ir()
    mutant = mut.generate_apgc(ir)[0].ir
    kinds_before = [element.kind for element in ir.elements]
    kinds_after = [element.kind for element in mutant.elements]
    assert kinds_before.count("bs") + 1 == kinds_after.count("bs")
    assert kinds_before.count("ps") - 1 == kinds_after.count("ps")


def test_ala_keeps_weights_of_the_source_layer_when_copying():
    ir = _trained_ir()
    by_id = {mutant.mutant_id: mutant for mutant in mut.generate_ala(ir)}
    copied = by_id["ALA_after_L0_copy_previous"].ir
    assert len(copied.weights()) == len(ir.weights()) + N_MODES
    assert copied.weights()[:N_MODES] == copied.weights()[N_MODES : 2 * N_MODES]


def test_dfc_leaves_the_circuit_untouched():
    mutants = mut.generate_dfc(N_MODES)
    assert all(mutant.ir is None for mutant in mutants)
    assert all(mutant.feature_transform is not None for mutant in mutants)
    x = torch.full((2, N_MODES), 0.25)
    mutated = mut.apply_feature_transform(x, ((0, "one_minus"), (1, "sign_flip")))
    assert torch.allclose(mutated[:, 0], torch.full((2,), 0.75))
    assert torch.allclose(mutated[:, 1], torch.full((2,), -0.25))
    assert torch.allclose(mutated[:, 2], x[:, 2])


def _suite(model, x_candidates):
    """Return a SuiteSpec whose analytic suite is the model's correct samples."""
    model.eval()
    with torch.no_grad():
        analytic = model(x_candidates).argmax(dim=1)
        sampled = model(x_candidates, shots=1024).argmax(dim=1)
    labels = analytic.clone()
    return SuiteSpec(
        x_candidates=x_candidates,
        labels=labels,
        shot_positions=torch.nonzero(sampled == labels, as_tuple=True)[0].tolist(),
        analytic_positions=torch.nonzero(analytic == labels, as_tuple=True)[0].tolist(),
    )


def test_null_control_is_never_killed_analytically():
    """An exact copy of the original circuit must survive the analytic protocol.

    This is the invariant that the mixed-protocol bug violated: judging an
    analytic prediction against a shot-selected suite let sampling noise kill
    an unmutated circuit.
    """
    ir = _trained_ir()
    spec = _spec()
    model = PhotonicQnnClassifier(spec, ir)
    suite = _suite(model, torch.rand(10, N_MODES))
    outcomes = [
        evaluate_mutant(mutant, spec, model, suite, shots=1024)
        for mutant in mut.generate_control_mutants(ir, count=4)
    ]
    assert all(outcome.status_analytic == STATUS_SURVIVED for outcome in outcomes)
    assert all(outcome.n_changed_analytic == 0 for outcome in outcomes)


def test_zeroing_every_phase_changes_the_circuit_behaviour():
    """Wiping every learned weight must alter at least one suite prediction."""
    ir = _trained_ir()
    spec = _spec()
    model = PhotonicQnnClassifier(spec, ir)
    suite = _suite(model, torch.rand(12, N_MODES))
    wiped = set_weights(ir, [0.0] * len(ir.weights()))
    mutant = mut.Mutant(
        operator="APC",
        mutant_id="APC_all_zero",
        description="all phases zeroed",
        ir=wiped,
    )
    outcome = evaluate_mutant(mutant, spec, model, suite, shots=1024)
    assert outcome.status_analytic == STATUS_KILLED
    assert outcome.n_suite_analytic == 12


def test_summarise_excludes_incompetent_mutants_from_the_score():
    from lib.mutation_testing import MutantOutcome

    outcomes = [
        MutantOutcome("a", "APC", "", STATUS_KILLED, STATUS_KILLED, 1, 4, [0, 1, 2, 0]),
        MutantOutcome(
            "b", "APC", "", STATUS_SURVIVED, STATUS_SURVIVED, 0, 4, [0, 1, 2, 1]
        ),
        MutantOutcome("c", "APC", "", "incompetent", "incompetent", 0, 4, []),
    ]
    summary = summarise(outcomes, family="unit")
    assert summary["per_operator"]["APC"]["mutation_score"] == 0.5
    assert summary["total_incompetent"] == 1
    assert summary["total_generated"] == 3
