"""Photonic *reservoir* models (MerLin) for the time-series benchmark.

EXTENSION, NOT REPRODUCTION
---------------------------
The paper (arXiv:2504.12416) benchmarks *variational* QML: every quantum model
has trainable circuit parameters optimised by gradient descent.  The models in
this file are **not variational** -- the photonic circuit is frozen at random
initialisation and only a classical linear readout is trained.  They therefore
sit outside the paper's stated scope and are reported separately from the
reproduction.  They are included because the paper's own outlook names quantum
reservoir computing as the more promising direction, and because the benchmark's
protocol (fixed splits, 27 tasks, published classical reference numbers) is a
calibrated yardstick to test that suggestion against.

Two models, forming a clean ablation of *where the memory lives*:

``PhotonicReservoir``
    QORC-style static optical reservoir.  The whole window is flattened, pushed
    through a fixed random projection into phase-shifter angles, scrambled by a
    frozen interferometric mesh, and read out in the Fock basis.  Memory of the
    sequence is supplied entirely by the input window -- the chip is memoryless.

``PhotonicMemristiveReservoir``
    Same frozen mesh, but the window is fed **one time step at a time** and the
    chip carries a memristive phase shifter (MerLin 0.4
    ``CircuitBuilder.add_memristive_ps``) whose phase is updated from the
    measured output after every step.  The optical state therefore has its own
    intrinsic temporal memory, in the spirit of the quantum-memristor reservoir
    of arXiv:2504.18694 (reproduced at ``papers/qrc_memristor``), but using
    MerLin's native memristive component instead of an external feedback layer.

Comparing the two isolates the contribution of the *optical* memory, with the
readout, mesh size, photon number and training budget held fixed.

Requires ``merlinquantum`` >= 0.4.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def parse_reservoir_ansatz(ansatz: str) -> dict:
    """Decode reservoir hyperparameters from the ansatz string.

    Follows the repo's existing convention of encoding architecture in the ansatz
    (cf. ``relu_16_16``, ``original_2``).  Grammar::

        reservoir[_scale<float>][_leak<float>][_mem<int>]

    e.g. ``reservoir_scale1.57_leak0.9_mem3``.  Unspecified keys keep their
    defaults, so a bare ``reservoir`` reproduces the original configuration.
    """
    out: dict = {}
    for tok in str(ansatz).split("_")[1:]:
        for key, cast in (("scale", float), ("leak", float), ("mem", int)):
            if tok.startswith(key):
                try:
                    out[key] = cast(tok[len(key) :])
                except ValueError as exc:
                    raise ValueError(f"bad reservoir ansatz token {tok!r}") from exc
                break
        else:
            raise ValueError(f"unrecognised reservoir ansatz token {tok!r}")
    return out


def _output_dim(data_label: str) -> int:
    if data_label.startswith("lorenz"):
        return 3
    if data_label.startswith("henon"):
        return 2
    return 1


def _default_input_state(n_modes: int, n_photons: int) -> list[int]:
    """Photons spread across modes so every mode is in the light-cone.

    Identical convention to ``lib/photonic.py`` so the reservoirs and the
    dressed photonic QNN share an input state.
    """
    state = [0] * n_modes
    step = max(1, n_modes // n_photons)
    placed = 0
    for m in range(0, n_modes, step):
        if placed >= n_photons:
            break
        state[m] = 1
        placed += 1
    for m in range(n_modes):  # top up if rounding left photons unplaced
        if placed >= n_photons:
            break
        if state[m] == 0:
            state[m] = 1
            placed += 1
    return state


def _freeze_circuit(qlayer: nn.Module) -> int:
    """Freeze every parameter of the quantum layer (reservoir condition).

    Returns the number of frozen scalar parameters, i.e. the size of the
    random optical feature map that is *not* trained.
    """
    frozen = 0
    for p in qlayer.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
        frozen += p.numel()
    return frozen


class PhotonicReservoir(nn.Module):
    """Static optical reservoir: frozen mesh + trainable linear readout.

    window (l*d) --fixed random proj--> n_modes angles --[frozen mesh]-->
        Fock probabilities --trainable Linear--> d prediction

    Parameters
    ----------
    data_label : str
        Dataset label (sets input/output dimension).
    seq_length : int
        Sliding-window length ``l``.
    n_modes : int
        Number of optical modes. Default value is 6.
    n_photons : int
        Number of photons injected. Default value is 3.
    random_id : int
        Seed for the fixed random projection and mesh. Default value is 42.
    """

    def __init__(
        self,
        data_label: str,
        seq_length: int,
        n_modes: int = 6,
        n_photons: int = 3,
        random_id: int = 42,
        scale: float = float(np.pi),
        **_,
    ) -> None:
        super().__init__()
        import merlin as ml

        self.data_label = data_label
        self.seq_length = seq_length
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.scale = float(scale)
        d = _output_dim(data_label)

        torch.manual_seed(random_id)
        modes = list(range(n_modes))
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=modes, scale=float(scale))
        builder.add_entangling_layer()

        self.input_state = _default_input_state(n_modes, n_photons)
        self.qlayer = ml.QuantumLayer(
            input_size=n_modes,
            builder=builder,
            input_state=self.input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            ),
        )
        self.n_frozen = _freeze_circuit(self.qlayer)

        # Fixed random input projection (reservoir input weights are untrained).
        self.proj = nn.Linear(d * seq_length, n_modes)
        for p in self.proj.parameters():
            p.requires_grad_(False)

        # The only trainable component.
        self.readout = nn.Linear(self.qlayer.output_size, d)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Reservoir feature vector (frozen part) for a batch of windows."""
        x = torch.reshape(x, (x.size(0), -1))
        angles = torch.sigmoid(self.proj(x))  # keep phases in [0, pi]
        return self.qlayer(angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x))


class PhotonicMemristiveReservoir(nn.Module):
    """Optical reservoir with a memristive phase shifter carrying the memory.

    The window is consumed step by step.  At step ``t`` the ``d``-dimensional
    observation is projected (fixed random weights) onto ``n_modes`` phases and
    run through the frozen mesh; the memristive phase shifter then updates its
    own phase from the measured distribution, so step ``t+1`` sees a chip whose
    transfer function depends on the whole history of the window.

    The per-step probability vectors are concatenated and mapped to the
    prediction by the single trainable linear readout.

    Parameters
    ----------
    data_label : str
        Dataset label (sets input/output dimension).
    seq_length : int
        Sliding-window length ``l``; also the number of chip evaluations.
    n_modes : int
        Number of optical modes. Default value is 6.
    n_photons : int
        Number of photons injected. Default value is 3.
    random_id : int
        Seed for the fixed random projection and mesh. Default value is 42.
    leak : float
        Memristive leak/retention in ``[0, 1]``: the fraction of the previous
        phase retained at each update. Default value is 0.5.
    n_memristors : int
        How many modes carry a memristive phase shifter. Default value is 2.
        Setting ``n_memristors=0`` builds the **capacity control**: the window is
        still consumed step by step and the per-step outputs still concatenated
        (so the trainable readout has exactly the same size), but the chip has no
        memory.  Any gap between the two is then attributable to the memristive
        dynamics rather than to readout capacity.
    """

    def __init__(
        self,
        data_label: str,
        seq_length: int,
        n_modes: int = 6,
        n_photons: int = 3,
        random_id: int = 42,
        scale: float = float(np.pi),
        leak: float = 0.5,
        n_memristors: int = 2,
        **_,
    ) -> None:
        super().__init__()
        import merlin as ml

        self.data_label = data_label
        self.seq_length = seq_length
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.scale = float(scale)
        self.leak = float(leak)
        d = _output_dim(data_label)
        self.d = d

        torch.manual_seed(random_id)
        modes = list(range(n_modes))
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=modes, scale=float(scale))

        # Memristive phase shifters: phase_{t+1} = leak*phase_t
        #                                       + (1-leak)*pi*<readout observable>
        # The observable is the total probability mass on the memristor's own
        # mode-marginal, so each memristor tracks a different projection of the
        # output distribution.
        self._mem_modes = list(range(min(max(n_memristors, 0), n_modes)))
        self.has_memory = len(self._mem_modes) > 0
        for k in self._mem_modes:
            builder.add_memristive_ps(
                mode=k,
                update_rule=self._make_update_rule(k),
                initial_state=float(np.pi) / 2.0,
                name=f"mem{k}",
            )
        builder.add_entangling_layer()

        self.input_state = _default_input_state(n_modes, n_photons)
        self.qlayer = ml.QuantumLayer(
            input_size=n_modes,
            builder=builder,
            input_state=self.input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            ),
        )
        self.n_frozen = _freeze_circuit(self.qlayer)

        self.proj = nn.Linear(d, n_modes)
        for p in self.proj.parameters():
            p.requires_grad_(False)

        self.readout = nn.Linear(self.qlayer.output_size * seq_length, d)

    def _make_update_rule(self, k: int):
        """Build the update rule for the memristor sitting on mode ``k``.

        ``update_rule(state, output) -> new_state`` with ``new_state`` of shape
        ``[batch_size]``; ``output`` is the measured probability distribution.
        """
        leak = None  # resolved lazily so `leak` follows the instance attribute

        def rule(state: torch.Tensor, output) -> torch.Tensor:
            nonlocal leak
            if leak is None:
                leak = self.leak
            probs = output if torch.is_tensor(output) else torch.as_tensor(output)
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            # A distinct, deterministic projection of the distribution per
            # memristor: probability mass on every stride-th outcome.  The stride
            # is floored at 2 because a stride of 1 would sum the whole
            # (normalised) distribution to exactly 1.0 for every input, making
            # the memristor a fixed clock that carries no data -- degenerate.
            stride = max(len(self._mem_modes), 2)
            drive = probs[:, k % stride :: stride].sum(dim=-1)
            drive = torch.clamp(drive, 0.0, 1.0)
            new = leak * state + (1.0 - leak) * float(np.pi) * drive
            return new.to(state.dtype)

        return rule

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenated per-step reservoir states (frozen part)."""
        batch = x.size(0)
        x = torch.reshape(x, (batch, self.seq_length, self.d))
        # New sequence => memristive phases back to their initial value, so
        # windows stay independent of one another and of batch ordering.
        if self.has_memory:
            self.qlayer.reset(batch)
        outs = []
        for t in range(self.seq_length):
            angles = torch.sigmoid(self.proj(x[:, t, :]))
            outs.append(self.qlayer(angles))
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x))
