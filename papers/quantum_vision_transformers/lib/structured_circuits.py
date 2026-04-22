from __future__ import annotations

import math
from dataclasses import dataclass

import perceval as pcvl


@dataclass(frozen=True)
class ButterflySpec:
    n_modes: int
    n_stages: int
    pairings: list[list[tuple[int, int]]]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _butterfly_pairings(n_modes: int) -> list[list[tuple[int, int]]]:
    """Logical radix-2 butterfly pairings stage by stage."""
    if not _is_power_of_two(n_modes):
        raise ValueError(
            f"Butterfly layout requires n_modes to be a power of 2, got {n_modes}."
        )

    n_stages = int(math.log2(n_modes))
    stages: list[list[tuple[int, int]]] = []

    for stage in range(n_stages):
        stride = 2**stage
        block = 2 ** (stage + 1)
        pairs: list[tuple[int, int]] = []

        for block_start in range(0, n_modes, block):
            for offset in range(stride):
                left = block_start + offset
                right = left + stride
                pairs.append((left, right))

        stages.append(pairs)

    return stages


def butterfly_spec(n_modes: int) -> ButterflySpec:
    pairings = _butterfly_pairings(n_modes)
    return ButterflySpec(
        n_modes=n_modes,
        n_stages=len(pairings),
        pairings=pairings,
    )


def make_mzi_block(
    *,
    phi_inner: float | pcvl.Parameter,
    phi_outer: float | pcvl.Parameter,
) -> pcvl.Circuit:
    c = pcvl.Circuit(2)
    c.add((0, 1), pcvl.BS())
    c.add(0, pcvl.PS(phi_inner))
    c.add((0, 1), pcvl.BS())
    c.add(0, pcvl.PS(phi_outer))
    return c


def _add_pair_block(
    circuit: pcvl.Circuit,
    *,
    left: int,
    right: int,
    component: pcvl.Circuit,
) -> None:
    """Apply a 2-mode component to a logical pair while respecting Perceval adjacency rules."""
    if right <= left:
        raise ValueError(f"Expected left < right, got ({left}, {right}).")

    if component.m != 2:
        raise ValueError(f"Expected a 2-mode component, got m={component.m}.")

    if right == left + 1:
        circuit.add(left, component)
        return

    span = right - left
    # On the local window [left + 1, ..., right], rotate the right mode to the front
    # so the logical pair (left, right) becomes physically adjacent at (left, left + 1).
    bring_right_next_to_left = [span - 1] + list(range(span - 1))
    restore_original_order = list(range(1, span)) + [0]

    circuit.add(left + 1, pcvl.PERM(bring_right_next_to_left))
    circuit.add(left, component)
    circuit.add(left + 1, pcvl.PERM(restore_original_order))


def make_butterfly_mzi_circuit(
    n_modes: int,
    *,
    prefix: str,
    trainable_inner: bool = True,
    trainable_outer: bool = True,
    fixed_inner: float = 0.0,
    fixed_outer: float = 0.0,
) -> pcvl.Circuit:
    """Build a radix-2 butterfly of MZIs, using local permutations for non-adjacent pairs."""
    spec = butterfly_spec(n_modes)
    circuit = pcvl.Circuit(n_modes)

    for stage_idx, stage_pairs in enumerate(spec.pairings):
        for pair_idx, (left, right) in enumerate(stage_pairs):
            if trainable_inner:
                phi_inner = pcvl.P(f"{prefix}_s{stage_idx}_p{pair_idx}_li")
            else:
                phi_inner = fixed_inner

            if trainable_outer:
                phi_outer = pcvl.P(f"{prefix}_s{stage_idx}_p{pair_idx}_lo")
            else:
                phi_outer = fixed_outer

            mzi = make_mzi_block(phi_inner=phi_inner, phi_outer=phi_outer)
            _add_pair_block(circuit, left=left, right=right, component=mzi)

    return circuit

def butterfly_param_count(
    n_modes: int,
    *,
    trainable_inner: bool = True,
    trainable_outer: bool = True,
) -> int:
    if not _is_power_of_two(n_modes):
        raise ValueError(
            f"Butterfly layout requires n_modes to be a power of 2, got {n_modes}."
        )

    n_stages = int(math.log2(n_modes))
    n_blocks = (n_modes // 2) * n_stages
    phases_per_block = int(trainable_inner) + int(trainable_outer)
    return n_blocks * phases_per_block
