"""Guided Quantum Compressor (GQC) model: encoder -> quantum/classical block
-> head, plus a decoder used only for the reconstruction loss at training
time.

The "quantum block" is selected via ``cfg["model"]["backend"]``:

- ``"gate"`` (default): the paper's 6-qubit PennyLane VQC (:mod:`.vqc_gate`).
- ``"photonic"``: the MerLin photonic adaptation (:mod:`.vqc_photonic`,
  Phase 4 of the reproduction workflow).
- ``"classical"``: a parameter-matched classical dense block
  (:mod:`.vqc_classical`), used as the C5 fair-baseline ablation to isolate
  whether the quantum block contributes beyond a same-shaped classical layer.

All three share the identical autoencoder, head, training loop, calibration,
and MoE code, so any resulting metric differences are attributable to the
quantum/classical block itself.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .autoencoder import Decoder, Encoder
from .head import Head
from .vqc_classical import ClassicalVQCReplacement
from .vqc_gate import VQCLayer
from .vqc_photonic import PhotonicVQCLayer

_BACKENDS = {"gate", "photonic", "classical"}


def _build_backend(backend: str, n_qubits: int, model_cfg: dict[str, Any]) -> nn.Module:
    if backend == "gate":
        n_layers = int(model_cfg.get("vqc", {}).get("n_layers", 6))
        return VQCLayer(n_qubits=n_qubits, n_layers=n_layers)
    if backend == "photonic":
        photonic_cfg = model_cfg.get("photonic", {})
        n_photons = int(photonic_cfg.get("n_photons", 3))
        readout = str(photonic_cfg.get("readout", "fixed"))
        return PhotonicVQCLayer(n_qubits=n_qubits, n_photons=n_photons, readout=readout)
    if backend == "classical":
        n_layers = int(model_cfg.get("vqc", {}).get("n_layers", 6))
        return ClassicalVQCReplacement(n_qubits=n_qubits, n_layers=n_layers)
    raise ValueError(
        f"Unknown model.backend={backend!r}; expected one of {sorted(_BACKENDS)}"
    )


class GQCModel(nn.Module):
    """Combined GQC hybrid model.

    ``forward`` returns ``(p_hat, x_reconstructed)`` where
    ``x_reconstructed = decoder(encoder(x))``. The decoder is only needed for
    the reconstruction loss at train time; :meth:`predict_proba` skips it for
    faster inference.
    """

    def __init__(self, input_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        n_qubits = int(model_cfg.get("n_qubits", 6))
        ae_cfg = model_cfg.get("autoencoder", {})
        hidden_dims = list(ae_cfg.get("hidden_dims", [256, 128, 64]))
        head_cfg = model_cfg.get("head", {})
        head_hidden_dim = int(head_cfg.get("hidden_dim", 8))
        backend = str(model_cfg.get("backend", "gate"))

        self.encoder = Encoder(input_dim, hidden_dims, n_qubits)
        self.decoder = Decoder(input_dim, hidden_dims, n_qubits)
        self.vqc = _build_backend(backend, n_qubits, model_cfg)
        self.head = Head(hidden_dim=head_hidden_dim)
        self.backend = backend

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        vqc_out = self.vqc(z)
        p_hat = self.head(vqc_out)
        x_reconstructed = self.decoder(z)
        return p_hat, x_reconstructed

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only forward pass that skips the decoder."""
        z = self.encoder(x)
        vqc_out = self.vqc(z)
        return self.head(vqc_out)


__all__ = ["GQCModel"]
