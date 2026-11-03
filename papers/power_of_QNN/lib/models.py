"""Build the paper's models from their JSON config entries."""

from __future__ import annotations

from .classical import FeedForwardNetwork
from .quantum import QubitQNN

MODEL_TYPES = ("qubit", "classical", "photonic", "qloq", "feedforward")


def build_model(model_cfg: dict):
    """Return a model object with ``probabilities(x, theta)`` and ``d``.

    Parameters
    ----------
    model_cfg : dict
        ``{"type": "qubit", "n_qubits", "feature_map", "feature_map_reps",
        "ansatz_reps", "feature_entanglement", "ansatz_entanglement",
        "hadamard_every_rep"}`` or ``{"type": "classical", "sizes", "activation",
        "head", "bias"}`` or ``{"type": "photonic", "n_inputs", "encoding", "d",
        "detection", "extra_modes", "mesh_order", "layout", "input_kind", "readout",
        "data_reps"}`` (``data_reps: null`` = the paper's depth)
        (MerLin adaptation, Phase 04), ``{"type": "qloq", "n_qubits", "feature_map",
        "feature_map_reps", "ansatz_reps"}`` (QLOQ single-photon encoding) or
        ``{"type": "feedforward", "encoding"}`` (feed-forward block). Every key is
        required.

    Raises
    ------
    ValueError
        For an unknown model type.
    """
    model_type = model_cfg["type"]
    if model_type == "qubit":
        return QubitQNN(
            n_qubits=model_cfg["n_qubits"],
            feature_map=model_cfg["feature_map"],
            feature_map_reps=model_cfg["feature_map_reps"],
            ansatz_reps=model_cfg["ansatz_reps"],
            feature_entanglement=model_cfg["feature_entanglement"],
            ansatz_entanglement=model_cfg["ansatz_entanglement"],
            hadamard_every_rep=model_cfg["hadamard_every_rep"],
        )
    if model_type == "classical":
        return FeedForwardNetwork(
            sizes=model_cfg["sizes"],
            activation=model_cfg["activation"],
            head=model_cfg["head"],
            bias=model_cfg["bias"],
        )
    if model_type == "photonic":
        # Imported here so the qubit and classical experiments do not need MerLin.
        from .photonic_merlin import PhotonicEncodingModel

        return PhotonicEncodingModel(
            n_inputs=model_cfg["n_inputs"],
            encoding=model_cfg["encoding"],
            d=model_cfg["d"],
            detection=model_cfg["detection"],
            extra_modes=model_cfg["extra_modes"],
            mesh_order=model_cfg["mesh_order"],
            layout=model_cfg["layout"],
            input_kind=model_cfg["input_kind"],
            readout=model_cfg["readout"],
            data_reps=model_cfg["data_reps"],
        )
    if model_type == "qloq":
        from .qloq_merlin import QloqQubitQNN

        return QloqQubitQNN(
            n_qubits=model_cfg["n_qubits"],
            feature_map=model_cfg["feature_map"],
            feature_map_reps=model_cfg["feature_map_reps"],
            ansatz_reps=model_cfg["ansatz_reps"],
        )
    if model_type == "feedforward":
        from .feedforward_merlin import FeedForwardPhotonicModel

        return FeedForwardPhotonicModel(encoding=model_cfg["encoding"])
    raise ValueError(
        f"Unknown model type {model_type!r}; expected one of {MODEL_TYPES}"
    )


def describe_model(model_cfg: dict, model) -> dict:
    """Return a JSON-friendly description of a built model."""
    description = {
        "name": model_cfg["name"],
        "type": model_cfg["type"],
        "d": model.d,
        "input_size": model.input_size,
    }
    if hasattr(model, "describe"):
        description["hardware"] = model.describe()
    return description
