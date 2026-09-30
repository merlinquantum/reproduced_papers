"""Mutation operators: the paper's seven QML operators, plus the prior baseline.

Iterations 3 and 4 of the reproduction plan.

The paper defines its operators for gate-model QNNs. Their photonic
counterparts are implemented here on the component-level IR of
:mod:`lib.circuit_ir`. Every mapping decision, and every place where the paper
is ambiguous, is recorded in ``LOG.md`` under "Mutation Operator Mapping".

Summary of the mapping:

======  =============================================================
Op      Photonic counterpart
======  =============================================================
APC     mutate the value of trainable phase shifters of one ansatz layer
DFC     mutate a pair of input features (data mutation, circuit untouched)
APGC    replace a trainable phase shifter (phase effect) by a trainable
        beam-splitter angle (amplitude effect), same parameter value
LS      swap the trainable phase columns of two ansatz layers
ILS     swap the two blocks inside one layer (mixing before phases)
ALA     insert an extra layer, weights random or copied from a neighbour
ALD     delete a layer: whole layer, phase column only, or mixing only
======  =============================================================

The baseline family reproduces the prior operators of Muskit / QMutPy at the
photonic component level: add a component, delete a component, change a
component for another of the same arity.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations, product

import torch
from lib.circuit_ir import (
    BALANCED_BS_THETA,
    BLOCK_ANSATZ_MIX,
    BLOCK_ANSATZ_PHASE,
    CircuitIR,
    Element,
    nearest_neighbour_pairs,
)

logger = logging.getLogger(__name__)

NEW_OPERATORS = ("APC", "DFC", "APGC", "LS", "ILS", "ALA", "ALD")
BASELINE_OPERATORS = ("ADD", "DELETE", "CHANGE")
CONTROL_OPERATOR = "CONTROL"


@dataclass(frozen=True)
class Mutant:
    """One generated mutant.

    Attributes
    ----------
    operator : str
        Operator that produced it.
    mutant_id : str
        Unique, deterministic identifier.
    description : str
        Human-readable description of the injected fault.
    ir : CircuitIR | None
        Mutated circuit, or ``None`` when the fault is in the data (DFC) and
        the original circuit is reused.
    feature_transform : tuple | None
        For DFC: ``((feature_i, op_name_i), (feature_j, op_name_j))``.
    """

    operator: str
    mutant_id: str
    description: str
    ir: CircuitIR | None = None
    feature_transform: tuple | None = None


# --------------------------------------------------------------------------
# Directed value sets
# --------------------------------------------------------------------------
# The paper's directed scheme deliberately avoids random draws: "we use known
# radians such as pi/2 and pi to ensure that the phase is shifted", and skips
# theta = 2*pi because it is covered by theta = 0. These five operations are
# the four categories of Section IV-C-1 (zeroing, sign flip, addition,
# scaling) instantiated with those known radians. Five operations per phase
# shifter is also what Table I implies (APC total = 5 * n_qubits * n_layers).
APC_OPERATIONS: dict[str, Callable[[float], float]] = {
    "zero": lambda angle: 0.0,
    "sign_flip": lambda angle: -angle,
    "add_half_pi": lambda angle: angle + math.pi / 2,
    "add_pi": lambda angle: angle + math.pi,
    "scale_two": lambda angle: 2.0 * angle,
}

# DFC operates on the scaled feature (in [0, 1]) that drives an encoding
# phase. With encoding_scale = pi, "add_half" is exactly a +pi/2 phase shift,
# so the directed-radian logic carries over to the data side.
DFC_OPERATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "add_half": lambda value: value + 0.5,
    "multiply_two": lambda value: value * 2.0,
    "sign_flip": lambda value: -value,
    "one_minus": lambda value: 1.0 - value,
}

# Component inventory available to the prior "add" and "change" operators.
# The photonic equivalent of "an available gate set": two phase shifters with
# known radians, and the balanced beam splitter.
BASELINE_PS_ANGLES = (math.pi / 2, math.pi)
BASELINE_BS_ANGLES = (math.pi / 4, math.pi)


def apply_feature_transform(x: torch.Tensor, feature_transform: tuple) -> torch.Tensor:
    """Return a copy of ``x`` with one DFC feature-pair mutation applied.

    Parameters
    ----------
    x : torch.Tensor
        Scaled features of shape ``(batch, n_features)``.
    feature_transform : tuple
        ``((feature_i, op_name_i), (feature_j, op_name_j))``.

    Returns
    -------
    torch.Tensor
        Mutated features.
    """

    mutated = x.clone()
    for feature_index, operation_name in feature_transform:
        mutated[:, feature_index] = DFC_OPERATIONS[operation_name](
            mutated[:, feature_index]
        )
    return mutated


# --------------------------------------------------------------------------
# The seven QML operators
# --------------------------------------------------------------------------


def generate_apc(ir: CircuitIR, *, granularity: str = "gate") -> list[Mutant]:
    """Ansatz Parameter Change: mutate learned phase values.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.
    granularity : str
        ``"gate"`` mutates one phase shifter at a time (consistent with the
        mutant counts of Table I); ``"layer"`` mutates a whole phase column at
        once (consistent with the prose of Section IV-C-1). The paper's text
        and its counts disagree, so both are implemented. Default value is
        ``"gate"``.

    Returns
    -------
    list[Mutant]
        Generated mutants.

    Raises
    ------
    ValueError
        If ``granularity`` is not ``"gate"`` or ``"layer"``.
    """

    if granularity not in {"gate", "layer"}:
        raise ValueError(f"Unknown APC granularity: {granularity!r}")

    mutants: list[Mutant] = []
    for layer in ir.layer_indices():
        phase_positions = ir.positions(block=BLOCK_ANSATZ_PHASE, layer=layer)
        targets = (
            [[position] for position in phase_positions]
            if granularity == "gate"
            else [phase_positions]
        )
        for target in targets:
            for operation_name, operation in APC_OPERATIONS.items():
                mutated = ir.copy()
                for position in target:
                    element = mutated.elements[position]
                    mutated.elements[position] = element.with_value(
                        operation(float(element.value))
                    )
                label = (
                    f"m{mutated.elements[target[0]].modes[0]}"
                    if granularity == "gate"
                    else "column"
                )
                mutants.append(
                    Mutant(
                        operator="APC",
                        mutant_id=f"APC_L{layer}_{label}_{operation_name}",
                        description=(
                            f"layer {layer} {label}: phase -> {operation_name}"
                        ),
                        ir=mutated,
                    )
                )
    return mutants


def generate_dfc(n_features: int) -> list[Mutant]:
    """Data-sample Feature Change: mutate a pair of input features.

    All unordered feature pairs are combined with all ordered pairs of the
    four feature operations, which is the paper's "one parameter value in the
    pair is mutated through one operation and the other parameter value either
    the same or different operation".

    Parameters
    ----------
    n_features : int
        Number of encoded features.

    Returns
    -------
    list[Mutant]
        Generated mutants; each carries a ``feature_transform`` instead of an
        IR, because DFC leaves the circuit untouched.
    """

    mutants: list[Mutant] = []
    for first, second in combinations(range(n_features), 2):
        for op_first, op_second in product(DFC_OPERATIONS, repeat=2):
            transform = ((first, op_first), (second, op_second))
            mutants.append(
                Mutant(
                    operator="DFC",
                    mutant_id=f"DFC_f{first}{op_first}_f{second}{op_second}",
                    description=(
                        f"feature {first} -> {op_first}, feature {second} -> {op_second}"
                    ),
                    feature_transform=transform,
                )
            )
    return mutants


def generate_apgc(ir: CircuitIR) -> list[Mutant]:
    """Ansatz Parameterized Gate Change: phase element -> mixing element.

    The paper changes a parameterized gate for one with a *different effect*
    on the qubit (RX/RY change amplitude, RZ changes phase), keeping the
    parameter value. Linear optics has exactly one natural one-mode
    parameterized element, the phase shifter, whose effect is a pure phase.
    The element with a different effect at the same parameter value is the
    two-mode beam-splitter angle, which changes amplitudes. Each trainable
    phase shifter on mode ``m`` is therefore replaced by a trainable
    beam-splitter angle on ``(m, m+1)`` or ``(m-1, m)``, whichever exists.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.

    Returns
    -------
    list[Mutant]
        Generated mutants.
    """

    mutants: list[Mutant] = []
    for layer in ir.layer_indices():
        for position in ir.positions(block=BLOCK_ANSATZ_PHASE, layer=layer):
            element = ir.elements[position]
            mode = element.modes[0]
            candidate_pairs = []
            if mode + 1 < ir.n_modes:
                candidate_pairs.append((mode, mode + 1))
            if mode - 1 >= 0:
                candidate_pairs.append((mode - 1, mode))
            for pair in candidate_pairs:
                mutated = ir.copy()
                mutated.elements[position] = replace(element, kind="bs", modes=pair)
                mutants.append(
                    Mutant(
                        operator="APGC",
                        mutant_id=f"APGC_L{layer}_m{mode}_bs{pair[0]}{pair[1]}",
                        description=(
                            f"layer {layer}: phase shifter on mode {mode} -> "
                            f"beam-splitter angle on modes {pair}"
                        ),
                        ir=mutated,
                    )
                )
    return mutants


def generate_ls(ir: CircuitIR) -> list[Mutant]:
    """Layer Shuffling: swap two ansatz layers.

    Because every mixing column of this ansatz is the same fixed balanced
    mesh, swapping two whole layers, swapping only their phase columns, and
    swapping only their mixing columns all produce the same circuit. Only one
    variant is generated; the other two would be redundant mutants, exactly
    the kind the paper's directed scheme sets out to avoid. The suppressed
    count is reported by :func:`suppressed_redundant_counts`.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.

    Returns
    -------
    list[Mutant]
        Generated mutants, one per unordered layer pair.
    """

    mutants: list[Mutant] = []
    layers = ir.layer_indices()
    for first, second in combinations(layers, 2):
        mutated = ir.copy()
        first_positions = ir.positions(block=BLOCK_ANSATZ_PHASE, layer=first)
        second_positions = ir.positions(block=BLOCK_ANSATZ_PHASE, layer=second)
        first_values = [ir.elements[p].value for p in first_positions]
        second_values = [ir.elements[p].value for p in second_positions]
        for position, value in zip(first_positions, second_values, strict=True):
            mutated.elements[position] = mutated.elements[position].with_value(value)
        for position, value in zip(second_positions, first_values, strict=True):
            mutated.elements[position] = mutated.elements[position].with_value(value)
        mutants.append(
            Mutant(
                operator="LS",
                mutant_id=f"LS_L{first}_L{second}",
                description=f"swap ansatz layers {first} and {second}",
                ir=mutated,
            )
        )
    return mutants


def generate_ils(ir: CircuitIR) -> list[Mutant]:
    """In-Layer Shuffling: swap the two blocks inside one layer.

    A photonic layer has two blocks, the trainable phase column and the fixed
    mixing column, so there is exactly one non-identity permutation per layer:
    put the mixing column first.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.

    Returns
    -------
    list[Mutant]
        One mutant per ansatz layer.
    """

    mutants: list[Mutant] = []
    for layer in ir.layer_indices():
        phase_positions = ir.positions(block=BLOCK_ANSATZ_PHASE, layer=layer)
        mix_positions = ir.positions(block=BLOCK_ANSATZ_MIX, layer=layer)
        if not phase_positions or not mix_positions:
            continue
        phases = [ir.elements[p] for p in phase_positions]
        mixes = [ir.elements[p] for p in mix_positions]
        start = min(phase_positions + mix_positions)
        end = max(phase_positions + mix_positions)
        mutated = ir.copy()
        mutated.elements[start : end + 1] = mixes + phases
        mutants.append(
            Mutant(
                operator="ILS",
                mutant_id=f"ILS_L{layer}",
                description=f"layer {layer}: mixing column before phase column",
                ir=mutated,
            )
        )
    return mutants


def generate_ala(ir: CircuitIR, *, seed: int = 0) -> list[Mutant]:
    """Ansatz Layer Addition: insert an extra layer after an existing one.

    The paper requires the inserted layer's parameters to be set so that the
    mutant stays competent, using "both random initialization of the weights
    and copying of another layer's weights". Both are generated. The random
    initialisation is drawn from a generator seeded per insertion position so
    that the mutant set is reproducible.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.
    seed : int
        Base seed for the random-initialisation variant. Default value is 0.

    Returns
    -------
    list[Mutant]
        Generated mutants.
    """

    mutants: list[Mutant] = []
    for layer in ir.layer_indices():
        mix_positions = ir.positions(block=BLOCK_ANSATZ_MIX, layer=layer)
        phase_positions = ir.positions(block=BLOCK_ANSATZ_PHASE, layer=layer)
        if not mix_positions or not phase_positions:
            continue
        insert_at = max(mix_positions) + 1
        source_values = [float(ir.elements[p].value) for p in phase_positions]

        generator = torch.Generator().manual_seed(seed + 1000 * (layer + 1))
        random_values = (
            (torch.rand(len(source_values), generator=generator) * 2 * math.pi)
            - math.pi
        ).tolist()

        for strategy, values in (
            ("copy_previous", source_values),
            ("random", random_values),
        ):
            new_elements = [
                Element(
                    kind="ps",
                    modes=(mode,),
                    role="weight",
                    value=float(value),
                    feature_index=None,
                    block=BLOCK_ANSATZ_PHASE,
                    layer=layer,
                )
                for mode, value in enumerate(values)
            ] + [
                Element(
                    kind="bs",
                    modes=pair,
                    role="fixed",
                    value=BALANCED_BS_THETA,
                    feature_index=None,
                    block=BLOCK_ANSATZ_MIX,
                    layer=layer,
                )
                for pair in nearest_neighbour_pairs(ir.n_modes)
            ]
            mutated = ir.copy()
            mutated.elements[insert_at:insert_at] = new_elements
            mutants.append(
                Mutant(
                    operator="ALA",
                    mutant_id=f"ALA_after_L{layer}_{strategy}",
                    description=(
                        f"insert an extra layer after layer {layer}, "
                        f"weights: {strategy}"
                    ),
                    ir=mutated,
                )
            )
    return mutants


def generate_ald(ir: CircuitIR) -> list[Mutant]:
    """Ansatz Layer Deletion: delete a layer, its phases, or its mixing.

    The paper targets "the entire layer, the rotation gates only, and the
    entangling gates only".

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.

    Returns
    -------
    list[Mutant]
        Generated mutants.
    """

    mutants: list[Mutant] = []
    for layer in ir.layer_indices():
        phase_positions = ir.positions(block=BLOCK_ANSATZ_PHASE, layer=layer)
        mix_positions = ir.positions(block=BLOCK_ANSATZ_MIX, layer=layer)
        variants = {
            "layer": phase_positions + mix_positions,
            "phases": phase_positions,
            "mixing": mix_positions,
        }
        for variant, positions in variants.items():
            if not positions:
                continue
            mutated = ir.copy()
            for position in sorted(positions, reverse=True):
                del mutated.elements[position]
            mutants.append(
                Mutant(
                    operator="ALD",
                    mutant_id=f"ALD_L{layer}_{variant}",
                    description=f"delete {variant} of layer {layer}",
                    ir=mutated,
                )
            )
    return mutants


def _generate_flat_control_mutants(ir: CircuitIR, *, count: int = 12) -> list[Mutant]:
    """Generate null mutants: exact copies of the original circuit.

    A mutant is declared killed when a suite prediction changes. With a finite
    shot budget the prediction can change from sampling noise alone, which
    would inflate every mutation score. These identity mutants measure that
    noise floor: each is evaluated through the same path, with its own
    deterministic sampling seed, so their kill rate is the false-kill rate of
    the whole procedure. The paper does not report such a control.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.
    count : int
        Number of independent identity replicates. Default value is 12.

    Returns
    -------
    list[Mutant]
        Identity mutants.
    """

    return [
        Mutant(
            operator=CONTROL_OPERATOR,
            mutant_id=f"CONTROL_{replicate:02d}",
            description=(
                "unmutated circuit, re-evaluated with an independent sampling "
                "seed (shot-noise control)"
            ),
            ir=ir.copy(),
        )
        for replicate in range(count)
    ]


def suppressed_redundant_counts(ir) -> dict[str, int]:
    """Return mutants the directed scheme declines to generate."""
    if isinstance(ir, dict):
        total_ls = 0
        for layer_ir in ir.values():
            n_layers = len(layer_ir.layer_indices())
            total_ls += 2 * (n_layers * (n_layers - 1) // 2)
        return {"LS": total_ls}

    n_layers = len(ir.layer_indices())
    layer_pairs = n_layers * (n_layers - 1) // 2
    return {
        "LS": 2 * layer_pairs,
    }


def _generate_flat_new_mutants(
    ir: CircuitIR,
    *,
    n_features: int,
    apc_granularity: str = "gate",
    ala_seed: int = 0,
) -> list[Mutant]:
    """Generate the full mutant set of the paper's seven operators.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.
    n_features : int
        Number of encoded features, used by DFC.
    apc_granularity : str
        See :func:`generate_apc`. Default value is ``"gate"``.
    ala_seed : int
        Seed for the ALA random-initialisation variant. Default value is 0.

    Returns
    -------
    list[Mutant]
        All generated mutants, grouped by operator in the order of the paper.
    """

    mutants = [
        *generate_apc(ir, granularity=apc_granularity),
        *generate_dfc(n_features),
        *generate_apgc(ir),
        *generate_ls(ir),
        *generate_ils(ir),
        *generate_ala(ir, seed=ala_seed),
        *generate_ald(ir),
    ]
    logger.info(
        "MUTANTS_GENERATED | family=new | total=%d | per_operator=%s",
        len(mutants),
        {
            operator: sum(1 for m in mutants if m.operator == operator)
            for operator in NEW_OPERATORS
        },
    )
    return mutants


def generate_new_operator_mutants(
    trained_ir, n_features: int, apc_granularity: str = "gate", ala_seed: int = 42
) -> list:
    if isinstance(trained_ir, CircuitIR):
        return _generate_flat_new_mutants(
            trained_ir,
            n_features=n_features,
            apc_granularity=apc_granularity,
            ala_seed=ala_seed,
        )

    if isinstance(trained_ir, list):
        return _generate_flat_new_mutants(
            trained_ir,
            n_features=n_features,
            apc_granularity=apc_granularity,
            ala_seed=ala_seed,
        )

    elif isinstance(trained_ir, dict):
        qcnn_mutants = []
        for layer_idx, layer_ir in trained_ir.items():
            layer_mutants = _generate_flat_new_mutants(
                layer_ir,
                n_features=n_features,
                apc_granularity=apc_granularity,
                ala_seed=ala_seed,
            )
            for mutant in layer_mutants:
                new_ir = None
                if mutant.ir is not None:
                    new_ir = trained_ir.copy()
                    new_ir[layer_idx] = mutant.ir

                qcnn_mutants.append(
                    replace(
                        mutant, ir=new_ir, mutant_id=f"L{layer_idx}_{mutant.mutant_id}"
                    )
                )
        return qcnn_mutants

    raise TypeError(f"Unsupporter representation format : {type(trained_ir)}")


def baseline_gate_set(n_modes: int) -> list[Element]:
    """Return the insertable component inventory for the prior operators.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.

    Returns
    -------
    list[Element]
        Phase shifters with known radians on every mode, plus balanced beam
        splitters on every nearest-neighbour pair.
    """

    elements: list[Element] = []
    for angle in BASELINE_PS_ANGLES:
        for mode in range(n_modes):
            elements.append(
                Element(
                    kind="ps",
                    modes=(mode,),
                    role="fixed",
                    value=angle,
                    feature_index=None,
                    block="inserted",
                    layer=None,
                )
            )
    for pair in nearest_neighbour_pairs(n_modes):
        elements.append(
            Element(
                kind="bs",
                modes=pair,
                role="fixed",
                value=BALANCED_BS_THETA,
                feature_index=None,
                block="inserted",
                layer=None,
            )
        )
    return elements


def generate_baseline_add(ir: CircuitIR) -> list[Mutant]:
    """Prior "gate addition": insert one component at every position.

    This is the exhaustive enumeration that the paper identifies as the main
    source of mutant blow-up.
    """

    mutants: list[Mutant] = []
    inventory = baseline_gate_set(ir.n_modes)
    for slot in range(len(ir.elements) + 1):
        for index, element in enumerate(inventory):
            mutated = ir.copy()
            mutated.elements.insert(slot, element)
            mutants.append(
                Mutant(
                    operator="ADD",
                    mutant_id=f"ADD_s{slot}_g{index}",
                    description=(
                        f"insert {element.kind} on modes {element.modes} at "
                        f"position {slot}"
                    ),
                    ir=mutated,
                )
            )
    return mutants


def generate_baseline_delete(ir: CircuitIR) -> list[Mutant]:
    """Prior "gate removal": delete one component."""

    mutants: list[Mutant] = []
    for position, element in enumerate(ir.elements):
        mutated = ir.copy()
        del mutated.elements[position]
        mutants.append(
            Mutant(
                operator="DELETE",
                mutant_id=f"DELETE_p{position}",
                description=(
                    f"delete {element.kind} on modes {element.modes} ({element.block})"
                ),
                ir=mutated,
            )
        )
    return mutants


def generate_baseline_change(ir: CircuitIR) -> list[Mutant]:
    """Prior "gate replacement": swap a component for one of equal arity."""

    mutants: list[Mutant] = []
    for position, element in enumerate(ir.elements):
        angles = BASELINE_PS_ANGLES if element.kind == "ps" else BASELINE_BS_ANGLES
        for angle in angles:
            mutated = ir.copy()
            mutated.elements[position] = replace(
                element, role="fixed", value=angle, feature_index=None
            )
            mutants.append(
                Mutant(
                    operator="CHANGE",
                    mutant_id=f"CHANGE_p{position}_a{angle:.4f}",
                    description=(
                        f"replace {element.kind} on modes {element.modes} "
                        f"({element.role}) by a fixed {element.kind} at "
                        f"{angle:.4f} rad"
                    ),
                    ir=mutated,
                )
            )
    return mutants


def _generate_flat_baseline_mutants(ir: CircuitIR) -> list[Mutant]:
    """Generate the full prior-operator mutant set.

    Parameters
    ----------
    ir : CircuitIR
        Trained original circuit.

    Returns
    -------
    list[Mutant]
        All baseline mutants.
    """

    mutants = [
        *generate_baseline_add(ir),
        *generate_baseline_delete(ir),
        *generate_baseline_change(ir),
    ]
    logger.info(
        "MUTANTS_GENERATED | family=baseline | total=%d | per_operator=%s",
        len(mutants),
        {
            operator: sum(1 for m in mutants if m.operator == operator)
            for operator in BASELINE_OPERATORS
        },
    )
    return mutants


def generate_baseline_mutants(trained_ir) -> list:
    if isinstance(trained_ir, CircuitIR):
        return _generate_flat_baseline_mutants(trained_ir)

    if isinstance(trained_ir, list):
        return _generate_flat_baseline_mutants(trained_ir)

    elif isinstance(trained_ir, dict):
        qcnn_mutants = []
        for layer_idx, layer_ir in trained_ir.items():
            layer_mutants = _generate_flat_baseline_mutants(layer_ir)
            for mutant in layer_mutants:
                new_ir = trained_ir.copy()
                new_ir[layer_idx] = mutant.ir

                qcnn_mutants.append(
                    replace(
                        mutant, ir=new_ir, mutant_id=f"L{layer_idx}_{mutant.mutant_id}"
                    )
                )
        return qcnn_mutants

    raise TypeError(f"Unsupported representation type : {type(trained_ir)}")


def generate_control_mutants(trained_ir, count: int = 12) -> list:
    if isinstance(trained_ir, CircuitIR):
        return _generate_flat_control_mutants(trained_ir, count=count)

    if isinstance(trained_ir, list):
        return _generate_flat_control_mutants(trained_ir, count=count)

    elif isinstance(trained_ir, dict):
        qcnn_mutants = []
        for i in range(count):
            mutant_dict = {idx: layer_ir.copy() for idx, layer_ir in trained_ir.items()}
            qcnn_mutants.append(
                Mutant(
                    operator=CONTROL_OPERATOR,
                    mutant_id=f"CONTROL_QCNN_{i:02d}",
                    description="unmutated QCNN circuit, re-evaluated with an independent sampling seed (shot-noise control)",
                    ir=mutant_dict,
                )
            )
        return qcnn_mutants

    raise TypeError(f"Unsupported representation type : {type(trained_ir)}")
