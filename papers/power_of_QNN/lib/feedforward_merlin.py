"""Feed-forward photonic model (Phase 04, Amendment 3, T2c): mid-circuit detection and
outcome-dependent interferometers, built with MerLin's ``FeedForwardBlock``.

Chip: 6 modes, 3 photons, input ``[1, 0, 1, 0, 1, 0]``.

- Stage 1 (data): interleaved re-uploading ``W_1 E(x) W_2 [E(x)]`` with the hard (two
  repetitions) or easy (one repetition) data map of ``lib.photonic_merlin``; PS-then-BS
  MZIs.
- Mode 0 is measured with a photon-number-resolving detector.
- Stage 2 (feed-forward): an ``FFCircuitProvider`` applies one of three trainable meshes to
  modes 1-5, for 0, 1 or 2 photons detected on mode 0 (all three photons on mode 0 leave
  modes 1-5 empty; that outcome uses the default circuit).
- Readout: parity of threshold clicks on modes 1, 3, 5.

The trainable phases are ``FeedForwardBlock`` parameters shared by a batch, unlike the
per-sample ``input_parameters`` of the other photonic models. ``summed_fisher_shared``
therefore sets the parameters for one theta and takes per-sample gradients over the input
batch. ``lib.fisher.empirical_fisher`` dispatches to it.
"""

from __future__ import annotations

import math

import merlin as ML
import perceval as pcvl
import torch

N_MODES = 6
INPUT_STATE = [1, 0, 1, 0, 1, 0]
STAGE1_MZIS = (7, 7)
BRANCH_MZIS = 2
BRANCHES = ("u", "v", "y")  # 0 photons (default), 1 photon, 2 photons on mode 0


def _mesh(n_modes: int, prefix: str, n_mzis: int) -> pcvl.Circuit:
    circuit = pcvl.Circuit(n_modes)
    placed, column = 0, 0
    while placed < n_mzis:
        for upper in range(column % 2, n_modes - 1, 2):
            if placed == n_mzis:
                break
            circuit.add(upper, pcvl.PS(pcvl.P(f"{prefix}{2 * placed:03d}")))
            circuit.add(upper, pcvl.BS(pcvl.P(f"{prefix}{2 * placed + 1:03d}")))
            placed += 1
        column += 1
    return circuit


def _mixer(circuit: pcvl.Circuit) -> None:
    for start in (0, 1):
        for upper in range(start, N_MODES - 1, 2):
            circuit.add(upper, pcvl.BS())


class FeedForwardPhotonicModel:
    """Photonic classifier with one feed-forward step (d = 40, s_in = 4).

    Parameters
    ----------
    encoding : str
        ``"hard"`` or ``"easy"`` (the data maps of ``lib.photonic_merlin``).
    """

    output_size = 2
    input_size = 4
    shared_parameters = True

    def __init__(self, encoding: str):
        if encoding not in ("hard", "easy"):
            raise ValueError(f"Unsupported encoding {encoding!r}")
        self.encoding = encoding
        self.repetitions = 2 if encoding == "hard" else 1
        self.group_sizes = {
            "w": 2 * STAGE1_MZIS[0],
            "z": 2 * STAGE1_MZIS[1],
            **dict.fromkeys(BRANCHES, 2 * BRANCH_MZIS),
        }
        self.groups = ["w", "z", *BRANCHES]
        self.d = sum(self.group_sizes.values())
        self._blocks: dict[tuple, ML.FeedForwardBlock] = {}

    def _experiment(self) -> pcvl.Experiment:
        stage = pcvl.Circuit(N_MODES)
        stage.add(0, _mesh(N_MODES, "w", STAGE1_MZIS[0]), merge=True)
        for repetition in range(self.repetitions):
            _mixer(stage)
            for feature in range(self.input_size):
                stage.add(feature, pcvl.PS(pcvl.P(f"a{repetition}{feature:03d}")))
            if self.encoding == "hard":
                _mixer(stage)
                for feature in range(self.input_size - 1):
                    stage.add(feature, pcvl.PS(pcvl.P(f"b{repetition}{feature:03d}")))
            if repetition == 0:
                stage.add(0, _mesh(N_MODES, "z", STAGE1_MZIS[1]), merge=True)
        experiment = pcvl.Experiment(N_MODES)
        experiment.add(0, stage)
        experiment.add(0, pcvl.Detector.pnr())
        provider = pcvl.FFCircuitProvider(1, 0, _mesh(N_MODES - 1, "u", BRANCH_MZIS))
        provider.add_configuration([1], _mesh(N_MODES - 1, "v", BRANCH_MZIS))
        # Two photons on mode 0 leave one photon in modes 1-5; with all three on mode 0 nothing
        # remains, so that outcome can use the default circuit (Perceval forbids reusing the
        # "y" names in a second configuration).
        provider.add_configuration([2], _mesh(N_MODES - 1, "y", BRANCH_MZIS))
        experiment.add(0, provider)
        return experiment

    def block(self, device: torch.device, dtype: torch.dtype) -> ML.FeedForwardBlock:
        """Return (and cache) the MerLin feed-forward block."""
        key = (str(device), dtype)
        if key not in self._blocks:
            self._blocks[key] = ML.FeedForwardBlock(
                self._experiment(),
                input_state=INPUT_STATE,
                input_parameters=["a", "b"] if self.encoding == "hard" else ["a"],
                trainable_parameters=self.groups,
                device=device,
                dtype=dtype,
            )
        return self._blocks[key]

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        columns = [(2.0 * x).repeat(1, self.repetitions)]
        if self.encoding == "hard":
            columns.append(
                (2.0 * (math.pi - x[:, :-1]) * (math.pi - x[:, 1:])).repeat(
                    1, self.repetitions
                )
            )
        return torch.cat(columns, dim=1)

    def _set_theta(
        self, block: ML.FeedForwardBlock, theta: torch.Tensor
    ) -> list[torch.Tensor]:
        parameters = dict(block.named_parameters())
        offset, ordered = 0, []
        with torch.no_grad():
            for group in self.groups:
                size = self.group_sizes[group]
                parameters[group].copy_(theta[offset : offset + size])
                ordered.append(parameters[group])
                offset += size
        return ordered

    def _class_probabilities(
        self, block: ML.FeedForwardBlock, x: torch.Tensor
    ) -> torch.Tensor:
        distribution = block(self._features(x))
        odd = torch.tensor(
            [
                sum(1 for mode in (1, 3, 5) if state[mode] > 0) % 2 == 1
                for state in block.output_keys
            ],
            device=x.device,
        )
        return torch.stack(
            (distribution[:, ~odd].sum(dim=1), distribution[:, odd].sum(dim=1)), dim=1
        )

    def probabilities_shared(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Class probabilities ``(B, 2)`` for one parameter vector ``theta`` ``(d,)``."""
        block = self.block(x.device, x.dtype)
        self._set_theta(block, theta)
        return self._class_probabilities(block, x)

    def summed_fisher_shared(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """``sum_b sum_y (d p_y)(d p_y)^T / p_y`` over the rows of ``x`` at one ``theta``."""
        block = self.block(x.device, x.dtype)
        parameters = self._set_theta(block, theta)
        probabilities = self._class_probabilities(block, x)
        total = torch.zeros(self.d, self.d, dtype=x.dtype, device=x.device)
        for output in range(probabilities.shape[1]):
            rows = []
            for row in range(x.shape[0]):
                gradients = torch.autograd.grad(
                    probabilities[row, output], parameters, retain_graph=True
                )
                rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
            jacobian = torch.stack(rows)
            weight = (
                probabilities[:, output].detach().clamp_min(torch.finfo(x.dtype).tiny)
            )
            scaled = jacobian / weight.sqrt().unsqueeze(1)
            total = total + scaled.T @ scaled
        return total.detach()

    def describe(self) -> dict:
        """Hardware-aware description (Phase 4 reporting contract)."""
        return {
            "computation_space": "FOCK",
            "detector_model": "mid-circuit PNR on mode 0; final ideal threshold clicks on modes 1-5",
            "photons": 3,
            "modes": N_MODES,
            "input_state": INPUT_STATE,
            "encoding": f"{self.encoding}: interleaved W1 E W2{' E' if self.encoding == 'hard' else ''} (data map of lib.photonic_merlin)",
            "trainable": f"stage 1: {STAGE1_MZIS} MZIs; feed-forward: 3 outcome-dependent meshes of {BRANCH_MZIS} MZIs on modes 1-5; d = {self.d}",
            "measurement_strategy": "FeedForwardBlock probs -> parity of clicks on modes 1, 3, 5",
            "postselection": "none (all detection outcomes kept, with feed-forward)",
            "simulator": "MerLin FeedForwardBlock analytic simulation (shots = 0)",
        }
