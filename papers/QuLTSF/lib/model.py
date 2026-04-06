from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from runtime_lib.dtypes import dtype_torch

from .original_qultsf import OriginalQuLTSFConfig, OriginalQuLTSFModel


@dataclass(frozen=True)
class QuLTSFModelConfig:
    input_size: int
    output_size: int
    sequence_length: int
    prediction_horizon: int
    latent_dim: int = 8
    hidden_dim: int = 32
    dtype: torch.dtype | None = None
    shots: int = 0


class ReferenceQuantumBlock(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhotonicQuantumBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dtype: torch.dtype,
        shots: int,
    ) -> None:
        super().__init__()
        from papers.QLSTM.lib.photonic_quantum_cell import make_photonic_vqc_2n_modes

        self.layer = make_photonic_vqc_2n_modes(
            n_inputs=latent_dim,
            out_size=hidden_dim,
            shots=shots,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class QuLTSFCore(nn.Module):
    def __init__(self, config: QuLTSFModelConfig, *, photonic: bool) -> None:
        super().__init__()
        dtype = config.dtype or torch.float32
        input_dim = config.input_size * config.sequence_length
        output_dim = config.output_size * config.prediction_horizon

        self.output_size = config.output_size
        self.prediction_horizon = config.prediction_horizon

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, config.latent_dim, dtype=dtype),
            nn.Tanh(),
        )
        if photonic:
            self.quantum_block = PhotonicQuantumBlock(
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                dtype=dtype,
                shots=config.shots,
            )
        else:
            self.quantum_block = ReferenceQuantumBlock(
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                dtype=dtype,
            )

        self.decoder = nn.Linear(config.hidden_dim, output_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        hidden = self.quantum_block(encoded)
        decoded = self.decoder(hidden)
        return decoded.view(x.shape[0], self.prediction_horizon, self.output_size)


def build_model(cfg: dict, metadata: dict) -> nn.Module:
    model_cfg = cfg.get("model", {})
    params = model_cfg.get("params", {})
    name = str(model_cfg.get("name", "qultsf_reference")).strip().lower()
    dtype = dtype_torch(cfg.get("dtype")) or torch.float32
    config = QuLTSFModelConfig(
        input_size=int(metadata["input_size"]),
        output_size=int(metadata["output_size"]),
        sequence_length=int(metadata["sequence_length"]),
        prediction_horizon=int(metadata["prediction_horizon"]),
        latent_dim=int(params.get("latent_dim", 8)),
        hidden_dim=int(params.get("hidden_dim", 32)),
        dtype=dtype,
        shots=int(params.get("shots", 0)),
    )

    if name == "qultsf_reference":
        return QuLTSFCore(config, photonic=False)
    if name == "photonic_qultsf":
        return QuLTSFCore(config, photonic=True)
    if name == "qultsf_original":
        return OriginalQuLTSFModel(
            OriginalQuLTSFConfig(
                seq_len=int(metadata["sequence_length"]),
                pred_len=int(metadata["prediction_horizon"]),
                num_qubits=int(params.get("num_qubits", 10)),
                qml_device=str(
                    params.get("qml_device", params.get("QML_device", "default.qubit"))
                ),
                num_layers=int(params.get("num_layers", 3)),
                dtype=dtype,
            )
        )
    raise ValueError(f"Unknown model.name '{name}'")
