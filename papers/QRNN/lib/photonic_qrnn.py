from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import merlin as ml
import torch
from merlin.core.partial_measurement import PartialMeasurement
from torch import nn

from .angle_encoding import apply_input_encoding
from .qrb_circuit import QRBCircuitSpec, build_qrb_circuit
from .qrb_readout import (
    ReadoutSpec,
    kron_states,
    partial_measurement_to_probabilities,
    project_fock_amplitudes_to_dual_rail,
)

logger = logging.getLogger(__name__)


@dataclass
class PhotonicQRNNConfig:
    kd: int  # logical "data" photons
    kh: int  # logical "hidden" photons
    depth: int = 1  # number of QRB layers per time step
    shots: int = 0  # 0 => full statevector
    dtype: torch.dtype | None = None
    input_encoding: str = "identity"  # identity|arccos
    # Post-selection / computational subspace for the measured D register.
    # - "dual_rail": outcomes correspond to bitstrings (size 2**kd)
    # - "unbunched": outcomes are any 0/1 occupation patterns with kd photons (size binom(2kd, kd))
    measurement_space: str = "dual_rail"


@dataclass(frozen=True)
class HiddenBranch:
    """A classical mixture component for the hidden register.

    prob:
        The classical probability weight of this branch.
    hidden_amp:
        Dual-rail amplitude vector for H (shape (2**kh,)).
    """

    prob: torch.Tensor
    hidden_amp: torch.Tensor


class PhotonicQRNNCell(nn.Module):
    """MerLin-based QRNN cell (prototype).

    Design assumptions (as agreed):
    - Dual-rail on 2m modes with m photons where m = kd + kh.
    - Partial measurement on the first 2*kd modes (D register).
    - dim(x_t) == 2*kd and x_t[i] maps directly to the i-th phase shifter in the encoding column.
    - Readout works directly in the 2**kd outcome space and uses a trainable linear layer:
        y_t = Linear(p_t) where p_t is in R^{2**kd}.

    Hidden state propagation:
    - Prototype implementation supports batch_size==1.
    - Partial-measurement branching is propagated exhaustively (exact classical mixture).
    """

    def __init__(self, input_size: int, config: PhotonicQRNNConfig) -> None:
        super().__init__()
        self.input_size = input_size
        self.config = config

        self._dtype = config.dtype if config.dtype is not None else torch.float64
        self._complex_dtype = (
            torch.complex64 if self._dtype == torch.float32 else torch.complex128
        )

        if config.shots not in (0, None):
            raise ValueError(
                "Partial measurement does not support sampling in Merlin; use shots=0."
            )
        if int(config.depth) <= 0:
            raise ValueError("depth must be positive")

        kd = int(config.kd)
        kh = int(config.kh)
        m = kd + kh
        self._required_input_size = 2 * kd
        if input_size > self._required_input_size:
            raise ValueError(
                f"Photonic QRNN expects input_size<=2*kd={self._required_input_size}, got {input_size}"
            )
        if input_size < self._required_input_size:
            logger.warning(
                "Photonic QRNN received input_size=%d but requires 2*kd=%d; will pad inputs with zeros.",
                input_size,
                self._required_input_size,
            )

        # Build QRB circuit and Merlin layer.
        circuit = build_qrb_circuit(QRBCircuitSpec(m=m, k=kd))
        measurement_space = ml.ComputationSpace.coerce(config.measurement_space)
        if measurement_space not in (
            ml.ComputationSpace.DUAL_RAIL,
            ml.ComputationSpace.UNBUNCHED,
        ):
            raise ValueError("measurement_space must be 'dual_rail' or 'unbunched'")

        strategy = ml.MeasurementStrategy.partial(
            modes=list(range(2 * kd)),
            computation_space=measurement_space,
        )

        # computation_space must be carried by MeasurementStrategy.* (Merlin v0.3+)
        self.qrb = ml.QuantumLayer(
            circuit=circuit,
            n_photons=m,
            trainable_parameters=["theta", "phi", "psi"],
            input_parameters=["in"],
            measurement_strategy=strategy,
            return_object=True,
            dtype=self._dtype,
        )

        # Readout dimension depends on the post-selected measurement space.
        if measurement_space is ml.ComputationSpace.DUAL_RAIL:
            readout_in_dim = 2**kd
        else:
            readout_in_dim = int(
                ml.Combinadics("unbunched", n=kd, m=2 * kd).compute_space_size()
            )
        # Match baseline API: QRNN is currently a scalar regressor.
        self.readout = nn.Linear(readout_in_dim, 1, dtype=self._dtype)

        # Fixed injected D state: |0...0> in dual-rail, i.e. [1,0]*kd.
        d_basis = ml.Combinadics("dual_rail", n=kd, m=2 * kd)
        d0_idx = int(d_basis.fock_to_index([1, 0] * kd))
        d0 = torch.zeros(2**kd, dtype=self._complex_dtype)
        d0[d0_idx] = 1.0 + 0.0j
        self.register_buffer("_d0", d0)

        # Initial hidden state (dual-rail) for H: |0...0> on kh qubits.
        h_basis = ml.Combinadics("dual_rail", n=kh, m=2 * kh) if kh > 0 else None
        h0 = torch.zeros(2**kh if kh > 0 else 1, dtype=self._complex_dtype)
        if kh > 0 and h_basis is not None:
            h0_idx = int(h_basis.fock_to_index([1, 0] * kh))
            h0[h0_idx] = 1.0 + 0.0j
        else:
            h0[0] = 1.0 + 0.0j
        self.register_buffer("_h0", h0)

        self.available = True
        logger.info(
            "PhotonicQRNNCell initialized (kd=%d, kh=%d, m=%d, depth=%d).",
            kd,
            kh,
            m,
            config.depth,
        )

    def _ensure_branch_tensor(self, prob: torch.Tensor) -> torch.Tensor:
        if prob.ndim == 0:
            return prob
        if prob.ndim == 1 and prob.numel() == 1:
            return prob.view(())
        raise ValueError(
            "This prototype expects scalar branch probabilities (batch_size==1)"
        )

    def _pad_features_to_required(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Pad feature dimension with zeros up to 2*kd.

        Merlin's QRB encoding expects exactly 2*kd input angles (one per D-mode phase
        shifter). When the dataset provides fewer features, we pad with zeros.
        """

        required = int(self._required_input_size)
        feat = int(x_batch.shape[-1])
        if feat == required:
            return x_batch
        if feat > required:
            raise ValueError(
                f"Photonic QRNN expects at most {required} features (2*kd), got {feat}"
            )

        pad = x_batch.new_zeros((x_batch.shape[0], required - feat))
        return torch.cat([x_batch, pad], dim=-1)

    def _expand_all_branches(
        self,
        pm: PartialMeasurement,
        *,
        branch_prob: torch.Tensor,
        kh: int,
    ) -> list[HiddenBranch]:
        """Expand a PartialMeasurement into HiddenBranch objects.

        Each measurement outcome contributes a new hidden branch with probability
        `branch_prob * p(outcome)` and with hidden amplitudes given by the conditional
        remaining state projected back into dual-rail.
        """

        out: list[HiddenBranch] = []
        for br in pm.branches:
            p = self._ensure_branch_tensor(br.probability)
            new_prob = branch_prob * p
            if kh == 0:
                new_hidden = self._h0
            else:
                new_hidden = project_fock_amplitudes_to_dual_rail(
                    br.amplitudes, n_qubits=kh
                )
            out.append(HiddenBranch(prob=new_prob, hidden_amp=new_hidden))
        return out

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | Iterable[HiddenBranch]
    ) -> tuple[torch.Tensor, list[HiddenBranch]]:
        """Apply `depth` QRB layers at a single time step.

        Parameters
        ----------
        x:
            Tensor of shape (batch, 2*kd) or (2*kd,). Angle encoding inputs.
        hidden:
            Either a single dual-rail amplitude vector for the hidden subsystem H
            (shape (2**kh,)) or an iterable of HiddenBranch objects.

        Returns
        -------
        y:
            Tensor of shape (batch, 1).
        next_hidden_branches:
            List of HiddenBranch objects representing a classical mixture over H.
        """

        if x.ndim == 1:
            x_batch = x.unsqueeze(0)
        elif x.ndim == 2:
            x_batch = x
        else:
            raise ValueError(f"Expected x to be 1D or 2D, got {tuple(x.shape)}")

        if x_batch.shape[0] != 1:
            raise ValueError(
                "Prototype photonic QRNN currently supports batch_size==1 only"
            )

        x_batch = self._pad_features_to_required(x_batch)
        x_batch = apply_input_encoding(
            x_batch.to(dtype=self._dtype), self.config.input_encoding
        )

        kh = int(self.config.kh)
        depth = int(self.config.depth)

        if isinstance(hidden, torch.Tensor):
            branches: list[HiddenBranch] = [
                HiddenBranch(
                    prob=torch.tensor(1.0, dtype=self._dtype), hidden_amp=hidden
                )
            ]
        else:
            branches = list(hidden)
            if not branches:
                raise ValueError("hidden branch list must be non-empty")

        # Run `depth` QRB layers. The number of branches grows multiplicatively with
        # the number of measurement outcomes.
        p_total: torch.Tensor | None = None
        for _layer_idx in range(depth):
            next_branches: list[HiddenBranch] = []
            p_accum = None

            for hb in branches:
                # Build full input superposition amplitude for 2*m modes in dual-rail basis.
                full_amp = kron_states(self._d0, hb.hidden_amp)
                self.qrb.set_input_state(full_amp)

                pm = self.qrb(x_batch)
                if not isinstance(pm, PartialMeasurement):
                    raise TypeError(f"Expected PartialMeasurement, got {type(pm)}")

                p_branch = partial_measurement_to_probabilities(
                    pm,
                    spec=ReadoutSpec(
                        k=int(self.config.kd),
                        computation_space=ml.ComputationSpace.coerce(
                            self.config.measurement_space
                        ),
                    ),
                )

                # Unconditioned distribution at this layer is a mixture over hidden branches.
                weight = self._ensure_branch_tensor(hb.prob)
                if p_accum is None:
                    p_accum = weight * p_branch
                else:
                    p_accum = p_accum + weight * p_branch

                next_branches.extend(
                    self._expand_all_branches(pm, branch_prob=weight, kh=kh)
                )

            assert p_accum is not None
            p_total = p_accum
            branches = next_branches

        assert p_total is not None
        y = self.readout(p_total)
        return y, branches


class PhotonicQRNNRegressor(nn.Module):
    """Unrolled photonic QRNN regressor (prototype).

    The forward pass returns a scalar prediction based on the last timestep.
    """

    def __init__(self, input_size: int, config: PhotonicQRNNConfig) -> None:
        super().__init__()
        self.cell = PhotonicQRNNCell(input_size, config)
        self.hidden_dim = config.kh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape (batch, seq_len, feat), got {tuple(x.shape)}"
            )
        batch, seq_len, feat = x.shape

        required = int(self.cell._required_input_size)
        if feat > required:
            raise ValueError(f"Expected feature dim <= {required} (2*kd), got {feat}")
        if feat < required:
            pad = x.new_zeros((batch, seq_len, required - feat))
            x = torch.cat([x, pad], dim=-1)

        # Note: we currently evaluate each sample sequentially (no vectorized batching)
        # because partial measurement returns structured branch objects.
        outputs: list[torch.Tensor] = []
        for b in range(batch):
            hidden: list[HiddenBranch] = [
                HiddenBranch(
                    prob=torch.tensor(1.0, dtype=self.cell._dtype),
                    hidden_amp=self.cell._h0,
                )
            ]
            y_b = None
            for t in range(seq_len):
                y_b, hidden = self.cell(x[b, t, :], hidden)
            assert y_b is not None
            outputs.append(y_b)

        y = torch.cat(outputs, dim=0)
        return y.squeeze(-1)


__all__ = ["PhotonicQRNNConfig", "PhotonicQRNNCell", "PhotonicQRNNRegressor"]
