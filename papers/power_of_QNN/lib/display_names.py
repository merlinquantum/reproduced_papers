"""Readable names for models and experiments in figures, tables and documentation.

Configs, run directories and result arrays keep their short code names (``Q2_d40``,
``B2_6m_cluster``, ``fair_iris12_d40``...), because the recorded evidence is keyed on them.
Everything a reader sees goes through :func:`model_label` and :func:`experiment_label`.

Photonic code names follow the design searches of Phase 04 (LOG.md Amendments 2-5):
``P0`` = the planned design, ``P1``-``P7`` = search 1 (circuit layout), ``T`` = search 2
(input states and encodings), ``A`` = search 3 (chip size and input), ``B`` = search 4
(capacity and loss together). The same circuit can carry two codes: B6 is A8, B4 is A7,
A2 is T2a (hard encoding) and A1 is P4 (hard encoding).
"""

from __future__ import annotations

import re

_FIXED_LABELS = {
    "qnn": "qubit QNN (paper)",
    "Q2": "qubit QNN (paper)",
    "Q3": "qubit QNN, 3 uploads",
    "easy_qnn": "easy QNN (paper)",
    "easy": "easy QNN (paper)",
    "easy_qnn_ry": "easy QNN, RY angle encoding (ablation A-03)",
    "classical_41112": "classical [4,1,1,1,2] (paper)",
    "classical_4442": "classical [4,4,4,2] (paper)",
    "classical_4622": "classical [4,6,2,2]",
    "classical_42": "linear [4,2]",
    "classical_d28_array": "classical [4,4,2,2] (published curve)",
    # Classical references of the input-size scaling runs (Figs. 8 and 9).
    "classical_6_110_2": "classical [6,110,2]",
    "classical_6111112": "classical [6,1,1,1,1,1,2]",
    "classical_811111112": "classical [8,1,1,1,1,1,1,1,2]",
    "classical_101412": "classical [10,1,4,1,2]",
    "classical_6722": "classical [6,7,2,2]",
    "classical_882": "classical [8,8,2]",
    "classical_108142": "classical [10,8,1,4,2]",
    "T2b": "qubit QNN as 1 photon in 16 modes (QLOQ)",
    "T2c": "photonic feed-forward",
}

# (modes, input state, data uploads or None for the paper's depth, layout, readout note)
_PHOTONIC_DESIGNS = {
    "P0": (6, "Fock", None, "encode then mesh", "planned design"),
    "P1": (6, "Fock", None, "encode then mesh", "dead phases removed"),
    "P2": (10, "Fock", None, "encode then mesh", None),
    "P3": (6, "Fock", None, "interleaved", None),
    "P4": (8, "Fock", None, "interleaved", None),
    "P5": (6, "cluster", None, "interleaved", None),
    "P6": (6, "GHZ", None, "interleaved", None),
    "P7": (6, "Fock", None, "interleaved", "photon-number parity readout"),
    "T2a": (8, "cluster", None, "interleaved", None),
    "A1": (8, "Fock", 2, "interleaved", None),
    "A2": (8, "cluster", 2, "interleaved", None),
    "A3": (10, "Fock", 2, "interleaved", None),
    "A4": (10, "cluster", 2, "interleaved", None),
    "A5": (12, "Fock", 2, "interleaved", None),
    "A6": (12, "cluster", 2, "interleaved", None),
    "A7": (8, "cluster", 3, "interleaved", None),
    "A8": (10, "cluster", 3, "interleaved", None),
    "B1": (6, "Fock", 3, "interleaved", None),
    "B2": (6, "cluster", 3, "interleaved", None),
    "B3": (8, "Fock", 3, "interleaved", None),
    "B4": (8, "cluster", 3, "interleaved", None),
    "B5": (10, "Fock", 3, "interleaved", None),
    "B6": (10, "cluster", 3, "interleaved", None),
}

# The two photonic designs carried into the fair comparison get a short name.
_SHORT_PHOTONIC = {
    "A8": "photonic 10-mode",
    "B6": "photonic 10-mode",
    "B2": "photonic 6-mode",
}

_EXPERIMENT_LABELS = {
    "fair_iris01_d8": "Iris 0/1, training loss (d = 8)",
    "fair_iris01_d40": "Iris 0/1, held-out (d = 40)",
    "fair_iris12_d40": "Iris 1/2, held-out (d = 40)",
    "fair_bcancer_d40": "breast cancer, held-out (d = 40)",
    "fair_mnist35_d40": "MNIST 3 vs 5, held-out (d = 40)",
}


# Phase 04 planned design (encode then mesh, Fock input, 2 extra modes), optionally at s inputs.
_PLANNED = re.compile(r"photonic_(hard|easy)(?:_s(\d+))?(_unbunched)?$")
# Paper qubit models at s qubits (Fig. 9 scaling runs).
_SCALED_QUBIT = re.compile(r"(qnn|easy_qnn)_s(\d+)$")


def _scaled_label(name: str, modes: int | None) -> str | None:
    """Label for the planned photonic design and the paper models at other input sizes."""
    planned = _PLANNED.match(name)
    if planned:
        encoding, inputs, unbunched = planned.groups()
        # The name only encodes the input size; a run that widened the chip (the 8-mode
        # diagnostic reuses ``photonic_hard``) must pass its recorded mode count.
        modes = modes if modes is not None else int(inputs or 4) + 2
        label = f"photonic {modes}-mode, Fock input, encode then mesh, planned design"
        if unbunched:
            label += ", collision-free postselection"
        return label + (", easy encoding" if encoding == "easy" else "")
    scaled = _SCALED_QUBIT.match(name)
    if scaled:
        family, qubits = scaled.groups()
        return f"{_FIXED_LABELS[family]}, {qubits} qubits"
    return None


def _code(name: str) -> str:
    """Return the design or model code at the start of a run-level model name."""
    if name in _FIXED_LABELS:
        return name
    match = re.match(r"(T2[abc]|[PAB]\d|Q[23]|easy)", name)
    if match is None:
        raise KeyError(f"No display name for model {name!r}")
    return match.group(1)


def photonic_description(name: str) -> str:
    """Return the full description of a photonic design candidate.

    Parameters
    ----------
    name : str
        Code name such as ``"A8_10m_cluster_reps3"`` or ``"P4_hard"``.

    Returns
    -------
    str
        For example ``"photonic 10-mode, cluster input, 3 uploads"``.
    """
    code = _code(name)
    if code not in _PHOTONIC_DESIGNS:
        easy = ", easy encoding" if name.endswith("_easy") else ""
        return _FIXED_LABELS[code] + easy
    modes, input_state, uploads, layout, note = _PHOTONIC_DESIGNS[code]
    if uploads is None:
        # The paper's depth: the hard encoding is uploaded twice, the easy one once.
        uploads = 1 if name.endswith("_easy") else 2
    parts = [f"photonic {modes}-mode", f"{input_state} input"]
    parts.append(
        "encode then mesh"
        if layout == "encode then mesh"
        else f"{uploads} upload{'s' if uploads > 1 else ''}"
    )
    if note:
        parts.append(note)
    if name.endswith("_easy"):
        parts.append("easy encoding")
    return ", ".join(parts)


def model_label(name: str, detailed: bool = False, modes: int | None = None) -> str:
    """Return the readable name of a model.

    Parameters
    ----------
    name : str
        Code name used in configs and run artifacts (``"Q2_d40"``, ``"B2_6m_cluster"``...).
    detailed : bool
        Spell out every photonic design (modes, input, uploads) instead of the short names
        of the two designs used in the fair comparison. Default value is False.
    modes : int | None
        Optical mode count recorded by the run (``entry["hardware"]["modes"]``). Needed
        for the planned photonic design, whose code name does not fix the chip size.
        Default value is None.

    Returns
    -------
    str
        Readable model name.

    Raises
    ------
    KeyError
        If the code name is unknown.
    """
    scaled = _scaled_label(name, modes)
    if scaled is not None:
        return scaled
    code = _code(name)
    if code in _PHOTONIC_DESIGNS:
        if not detailed and code in _SHORT_PHOTONIC:
            return _SHORT_PHOTONIC[code]
        return photonic_description(name)
    if code in ("T2b", "T2c"):
        return photonic_description(name)
    return _FIXED_LABELS[code]


def experiment_label(name: str) -> str:
    """Return the readable name of a fair-comparison experiment (``fair_*`` config stem)."""
    return _EXPERIMENT_LABELS[name]
