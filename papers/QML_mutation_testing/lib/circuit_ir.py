"""Component-level intermediate representation of the photonic QNN circuit.

Iteration 2 of the reproduction plan. Mutation testing needs to *edit* a
circuit: delete one component, insert one, swap two blocks, change a
component's type. MerLin's ``CircuitBuilder`` is a write-only declarative
API, so the circuit is represented here as an explicit ordered list of
primitive elements which can be copied and rewritten, and only then compiled
to a Perceval circuit.

This is the one place where the reproduction deliberately steps below the
high-level MerLin API, for the reason the photonic policy allows: the paper's
contribution *is* structural circuit manipulation, which the high-level
builder cannot express.

Only two primitive element kinds are needed, and they mirror exactly what
``CircuitBuilder.to_pcvl_circuit`` emits:

- ``ps``: a phase shifter on one mode -> ``pcvl.PS``
- ``bs``: a beam splitter on a mode pair -> ``pcvl.BS``

Provenance tags on each element (``block``, ``layer``) are what the mutation
operators use to find "the rotation column of layer 3" or "the entangler of
layer 0".
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace

import perceval as pcvl

# 50:50 splitting in Perceval's convention: reflectivity is cos^2(theta / 2),
# so theta = pi / 2 is the balanced beam splitter. This is also what
# ``EntanglingBlock(trainable=False)`` compiles to (a bare ``pcvl.BS()``).
BALANCED_BS_THETA = math.pi / 2

BLOCK_FEATURE_MIX = "fm_mix"
BLOCK_FEATURE_ENCODE = "fm_encode"
BLOCK_ANSATZ_PHASE = "ansatz_phase"
BLOCK_ANSATZ_MIX = "ansatz_mix"


@dataclass(frozen=True)
class Element:
    """One primitive optical component.

    Attributes
    ----------
    kind : str
        ``"ps"`` (phase shifter, one mode) or ``"bs"`` (beam splitter, two
        adjacent modes).
    modes : tuple[int, ...]
        Target mode for ``ps``, ordered mode pair for ``bs``.
    role : str
        ``"fixed"`` (constant angle), ``"weight"`` (learned parameter), or
        ``"encoding"`` (angle supplied by an input feature).
    value : float | None
        Angle for ``role="fixed"``, or the learned angle for
        ``role="weight"``. Unused for ``role="encoding"``.
    feature_index : int | None
        Index of the input feature driving an ``"encoding"`` element.
    block : str
        Provenance tag: one of the ``BLOCK_*`` constants.
    layer : int | None
        Ansatz layer index for ansatz elements, ``None`` for the feature map.
    """

    kind: str
    modes: tuple[int, ...]
    role: str
    value: float | None
    feature_index: int | None
    block: str
    layer: int | None

    def with_value(self, value: float) -> Element:
        """Return a copy of this element carrying a new angle."""

        return replace(self, value=value)


@dataclass
class CircuitIR:
    """An ordered, editable list of optical elements.

    Attributes
    ----------
    n_modes : int
        Number of optical modes.
    elements : list[Element]
        Components in propagation order.
    """

    n_modes: int
    elements: list[Element]

    def copy(self) -> CircuitIR:
        """Return a deep copy that can be mutated independently."""

        return CircuitIR(n_modes=self.n_modes, elements=deepcopy(self.elements))

    # -- structure queries used by the mutation operators -------------------

    def layer_indices(self) -> list[int]:
        """Return the sorted ansatz layer indices present in the circuit."""

        return sorted(
            {element.layer for element in self.elements if element.layer is not None}
        )

    def positions(
        self, *, block: str | None = None, layer: int | None = None
    ) -> list[int]:
        """Return positions of elements matching a block tag and/or layer.

        Parameters
        ----------
        block : str | None
            Block tag to match, or ``None`` for any. Default value is None.
        layer : int | None
            Ansatz layer to match, or ``None`` for any. Default value is None.

        Returns
        -------
        list[int]
            Matching positions, in circuit order.
        """

        return [
            index
            for index, element in enumerate(self.elements)
            if (block is None or element.block == block)
            and (layer is None or element.layer == layer)
        ]

    def weight_positions(self) -> list[int]:
        """Return positions of learned-weight elements, in circuit order."""

        return [
            index
            for index, element in enumerate(self.elements)
            if element.role == "weight"
        ]

    def weights(self) -> list[float | None]:
        """Return the learned angles in circuit order.

        Entries are ``None`` for weights that have not been assigned yet (the
        freshly built original circuit, whose angles MerLin initialises).
        """

        return [
            None
            if self.elements[index].value is None
            else float(self.elements[index].value)
            for index in self.weight_positions()
        ]

    def weights_initialised(self) -> bool:
        """Return whether every weight angle carries a concrete value."""

        values = self.weights()
        return bool(values) and all(value is not None for value in values)

    def encoding_feature_order(self) -> list[int]:
        """Return the feature indices consumed by encoding elements, in order.

        Mutants may drop an encoding element (prior "delete" or "change"
        operators), so this list is not always ``range(n_features)``. The model
        selects exactly these feature columns when building the phase vector.
        """

        return [
            element.feature_index
            for element in self.elements
            if element.role == "encoding"
        ]

    def n_encoded_features(self) -> int:
        """Return the number of distinct input features consumed."""

        return len(
            {
                element.feature_index
                for element in self.elements
                if element.role == "encoding"
            }
        )

    def signature(self) -> tuple:
        """Return a hashable structural+numeric fingerprint of the circuit.

        Two mutants with equal signatures are identical circuits and therefore
        redundant. The directed generation scheme of the paper is supposed to
        avoid producing them; this makes the claim measurable.
        """

        return tuple(
            (
                element.kind,
                element.modes,
                element.role,
                None if element.value is None else round(float(element.value), 12),
                element.feature_index,
            )
            for element in self.elements
        )

    # -- compilation -------------------------------------------------------

    def to_pcvl(self) -> tuple[pcvl.Circuit, list[str], list[str]]:
        """Compile to a Perceval circuit with named free parameters.

        Weight and encoding angles become named Perceval parameters so that
        MerLin can bind them; fixed angles are baked in. Names are zero-padded
        so that any lexicographic ordering MerLin may apply coincides with
        circuit order.

        Returns
        -------
        tuple[pcvl.Circuit, list[str], list[str]]
            The circuit, the ordered weight parameter names, and the ordered
            encoding parameter names.
        """

        circuit = pcvl.Circuit(self.n_modes)
        weight_names: list[str] = []
        encoding_names: list[str] = []

        for element in self.elements:
            if element.kind == "ps":
                if element.role == "fixed":
                    angle: object = element.value
                elif element.role == "weight":
                    name = f"theta{len(weight_names):04d}"
                    weight_names.append(name)
                    angle = pcvl.P(name)
                else:
                    name = f"px{len(encoding_names):04d}"
                    encoding_names.append(name)
                    angle = pcvl.P(name)
                circuit.add(element.modes[0], pcvl.PS(angle))

            elif element.kind == "bs":
                if element.role == "fixed":
                    theta: object = element.value
                elif element.role == "weight":
                    name = f"theta{len(weight_names):04d}"
                    weight_names.append(name)
                    theta = pcvl.P(name)
                else:
                    name = f"px{len(encoding_names):04d}"
                    encoding_names.append(name)
                    theta = pcvl.P(name)
                circuit.add(element.modes, pcvl.BS(theta=theta))

            else:
                raise ValueError(f"Unknown element kind: {element.kind!r}")

        return circuit, weight_names, encoding_names


def nearest_neighbour_pairs(n_modes: int) -> list[tuple[int, int]]:
    """Return the nearest-neighbour mode pairs of a planar mesh column."""

    return [(mode, mode + 1) for mode in range(n_modes - 1)]


def build_original_ir(
    *,
    n_modes: int,
    n_features: int,
    n_layers: int,
) -> CircuitIR:
    """Build the IR of the paper's ZFeatureMap + RealAmplitudes analogue.

    The layout matches ``lib.photonic_qnn`` exactly:

    1. a fixed balanced beam-splitter column (the ``H`` column of ZFeatureMap);
    2. one encoding phase shifter per feature (``P(2*x_i)``);
    3. ``n_layers`` times: a trainable phase column (``RY`` column of
       RealAmplitudes) followed by a fixed balanced mixing column (the CNOT
       entangler).

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    n_features : int
        Number of encoded features; must not exceed ``n_modes``.
    n_layers : int
        Number of ansatz layers.

    Returns
    -------
    CircuitIR
        Circuit whose weight angles are unassigned (``None``); MerLin
        initialises them when the layer is built.

    Raises
    ------
    ValueError
        If more features than modes are requested.
    """

    if n_features > n_modes:
        raise ValueError(f"Cannot encode {n_features} features on {n_modes} modes.")

    elements: list[Element] = []

    for pair in nearest_neighbour_pairs(n_modes):
        elements.append(
            Element(
                kind="bs",
                modes=pair,
                role="fixed",
                value=BALANCED_BS_THETA,
                feature_index=None,
                block=BLOCK_FEATURE_MIX,
                layer=None,
            )
        )
    for feature in range(n_features):
        elements.append(
            Element(
                kind="ps",
                modes=(feature,),
                role="encoding",
                value=None,
                feature_index=feature,
                block=BLOCK_FEATURE_ENCODE,
                layer=None,
            )
        )
    for layer in range(n_layers):
        for mode in range(n_modes):
            elements.append(
                Element(
                    kind="ps",
                    modes=(mode,),
                    role="weight",
                    value=None,
                    feature_index=None,
                    block=BLOCK_ANSATZ_PHASE,
                    layer=layer,
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
                    block=BLOCK_ANSATZ_MIX,
                    layer=layer,
                )
            )

    return CircuitIR(n_modes=n_modes, elements=elements)


def set_weights(ir: CircuitIR, weights: list[float]) -> CircuitIR:
    """Return a copy of ``ir`` with its weight angles replaced in order.

    Parameters
    ----------
    ir : CircuitIR
        Circuit whose weights should be replaced.
    weights : list[float]
        New angles, in circuit order.

    Returns
    -------
    CircuitIR
        Updated copy.

    Raises
    ------
    ValueError
        If the number of supplied weights does not match the circuit.
    """

    positions = ir.weight_positions()
    if len(positions) != len(weights):
        raise ValueError(
            f"Circuit has {len(positions)} weights but {len(weights)} were given."
        )
    updated = ir.copy()
    for position, weight in zip(positions, weights, strict=True):
        updated.elements[position] = updated.elements[position].with_value(
            float(weight)
        )
    return updated
