from __future__ import annotations

import logging

import merlin
import perceval as pcvl
import torch
import torch.nn as nn


def bs_layer_on_pairs(m: int) -> pcvl.Circuit:
    """First layer of parallel beam splitters on pairs of modes."""
    circuit = pcvl.Circuit(m)
    for k in range(0, m, 2):  # (0,1), (2,3), ...
        if k + 1 < m:
            circuit.add(k, pcvl.BS())
    return circuit


def ps_encode_inputs(m: int, n_inputs: int) -> pcvl.Circuit:
    """Phase shifters that encode classical inputs on the first n modes."""
    circuit = pcvl.Circuit(m)
    for i in range(n_inputs):
        circuit.add(i, pcvl.PS(pcvl.P(f"px{i}")))
    return circuit


def universal_interferometer(m: int, tag: str) -> pcvl.GenericInterferometer:
    """Trainable rectangular mesh interferometer with theta_* parameters."""
    return pcvl.GenericInterferometer(
        m,
        lambda idx: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_{tag}_in_{idx}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_{tag}_out_{idx}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )


class MerlinPhotonicGate(nn.Module):
    """Wrap a QuantumLayer and map probability simplex outputs to feature space."""

    def __init__(
        self,
        qlayer: merlin.QuantumLayer,
        target_size: int,
        shots: int = 0,
        *,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.quantum_layer = qlayer
        self.shots = int(shots)
        self._dtype = dtype if dtype is not None else torch.float32
        if qlayer.output_size != target_size:
            # Learned linear projection decouples features (LexGrouping keeps them on the simplex).
            self.post = nn.Linear(
                qlayer.output_size, target_size, bias=True, dtype=self._dtype
            )
        else:
            self.post = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shots = self.shots if self.shots > 0 else None
        # Cast d'entrée vers le dtype interne pour éviter les mismatches (ex: double vs float)
        out = self.quantum_layer(x.to(self._dtype), shots=shots)
        if isinstance(self.post, nn.Linear):
            out = self.post(out.to(self._dtype))
        else:
            out = out.to(self._dtype)
        return out


def make_photonic_vqc_2n_modes(
    n_inputs: int,
    out_size: int,
    shots: int = 0,
    dtype: torch.dtype | None = None,
) -> MerlinPhotonicGate:
    """Build a 2n-mode photonic VQC wrapped in a LexGrouping mapper."""

    m = 2 * n_inputs
    circuit = (
        bs_layer_on_pairs(m)
        // ps_encode_inputs(m, n_inputs)
        // universal_interferometer(m, "U")
    )

    qlayer = merlin.QuantumLayer(
        input_size=n_inputs,
        circuit=circuit,
        trainable_parameters=["theta_"],
        input_parameters=["px"],
        input_state=[1, 0] * n_inputs,
        computation_space=merlin.ComputationSpace.UNBUNCHED,
        measurement_strategy=merlin.MeasurementStrategy.MODE_EXPECTATIONS,
        dtype=dtype,
    )

    return MerlinPhotonicGate(qlayer, out_size, shots=shots, dtype=dtype)


class PhotonicQLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        shots: int = 0,
        use_photonic_head: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_photonic_head = use_photonic_head

        # Par défaut float64 pour cohérence avec couches MerLin
        self._dtype = dtype if dtype is not None else torch.float64

        n_v = input_size + hidden_size  # concat [x, h]

        # 4 gate VQCs
        self.vqc_i = make_photonic_vqc_2n_modes(
            n_v, hidden_size, shots, dtype=self._dtype
        )
        self.vqc_f = make_photonic_vqc_2n_modes(
            n_v, hidden_size, shots, dtype=self._dtype
        )
        self.vqc_g = make_photonic_vqc_2n_modes(
            n_v, hidden_size, shots, dtype=self._dtype
        )
        self.vqc_o = make_photonic_vqc_2n_modes(
            n_v, hidden_size, shots, dtype=self._dtype
        )

        # Output mapping: classical or photonic
        if use_photonic_head:
            self.vqc_head = make_photonic_vqc_2n_modes(
                hidden_size, output_size, shots, dtype=self._dtype
            )
        else:
            self.output_proj = nn.Linear(
                hidden_size,
                output_size,
                dtype=self._dtype,
            )

        # --- Logging: architecture summary ---
        logger = logging.getLogger(__name__)
        gate_modes = 2 * n_v
        gate_photons = n_v
        head_modes = 2 * hidden_size if use_photonic_head else None
        head_photons = hidden_size if use_photonic_head else None

        def _params(module: nn.Module) -> int:
            return sum(
                p.numel()
                for p in module.parameters()
                if getattr(p, "requires_grad", False)
            )

        try:
            total_params = _params(self)
        except Exception:
            total_params = -1

        measurement = getattr(self.vqc_i.quantum_layer, "measurement_strategy", None)
        gate_out_dim = getattr(self.vqc_i.quantum_layer, "output_size", "?")

        parts = [
            "PhotonicQLSTMCell initialized:",
            f"  - dtype={self._dtype}",
            f"  - shots={shots}",
            "  - gates: 4 photonic VQC on 2n modes (n=input+hidden)",
            f"    · gate input n = {n_v} -> modes={gate_modes}, photons={gate_photons}",
            f"    · measurement={measurement} -> gate output dim={gate_out_dim}",
        ]
        if use_photonic_head:
            parts.append(
                f"  - head: photonic VQC on 2*hidden modes -> modes={head_modes}, photons={head_photons}"
            )
        else:
            parts.append(f"  - head: classical Linear({hidden_size} -> {output_size})")
        parts.append(
            f"  - total trainable parameters ≈ {total_params if total_params >= 0 else 'N/A'}"
        )
        logger.info("\n".join(parts))

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        # Ensure internal dtype consistency for MerLin ops and Linear
        h_prev = h_prev.to(self._dtype)
        c_prev = c_prev.to(self._dtype)
        v_t = torch.cat([x_t, h_prev], dim=1).to(self._dtype)

        i = torch.sigmoid(self.vqc_i(v_t))
        f = torch.sigmoid(self.vqc_f(v_t))
        g = torch.tanh(self.vqc_g(v_t))
        o = torch.sigmoid(self.vqc_o(v_t))

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        if self.use_photonic_head:
            y_t = self.vqc_head(h_t)
        else:
            y_t = self.output_proj(h_t)

        return y_t, (h_t, c_t)


# Backward-compatibility alias (older code/tests may import this name)
PhotonicQuantumLSTMCell = PhotonicQLSTMCell
