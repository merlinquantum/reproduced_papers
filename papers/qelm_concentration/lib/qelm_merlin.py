"""MerLin photonic adaptation of the QELM reservoir (Phase 4; an adaptation, not the paper).

The kicked-Ising brickwall of paper Fig. 1 is mapped to a fixed linear-optical brickwall
on ``n_modes`` modes holding ``n_photons`` indistinguishable photons:

- a block on modes ``(a, a + 1)`` is ``PS(s h)`` on mode ``a + 1``, ``BS(theta)``, then
  ``PS(s h)`` on mode ``a + 1``. For one photon in the pair, ``BS(theta)`` acts like
  ``RX(theta)`` on a dual-rail qubit and the phase shifter like ``RZ``, so the Table I
  values carry over as ``theta = s b`` (dynamics layers) and ``theta = s a_in u``
  (encoding layers, every ``encoding_period``-th layer), with ``s = angle_scale``;
- the ``RZZ(J)`` interaction has no linear-optical counterpart and is dropped: the only
  multi-photon effect is bosonic interference;
- layers alternate between pairs ``(0, 1), (2, 3), ...`` and ``(1, 2), (3, 4), ...`` on an
  open chain (a planar chip has no ring closure);
- every mode carries an ideal threshold detector, simulated in the FOCK computation
  space, so outcomes are click patterns and nothing is postselected. MerLin's
  ``UNBUNCHED`` space renormalises on collision-free outcomes, which is postselection
  (LOG.md, Phase 4 plan).

Readouts mirror the qubit ones on disjoint mode pairs ``(0, 1), (2, 3), ...``: each pair has
4 click outcomes indexed ``2 * click_a + click_b``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import merlin as ml
import numpy as np
import perceval as pcvl
import torch
from scipy.stats import unitary_group

from .eigentasks import EigentaskBasis, signal_scale, subset_eigentasks, unit_scale
from .landsat_task import eigentask_readout

logger = logging.getLogger(__name__)

RESERVOIR_KINDS = ("brickwall", "haar")
CLICK_READOUT_KINDS = ("click", "et", "etcut")
READOUT_SCALINGS = ("unit", "scaled")
REQUIRED_ARCHITECTURE_KEYS = (
    "n_modes",
    "n_photons",
    "n_layers",
    "encoding_period",
    "input_shift",
    "angle_scale",
)
REQUIRED_BRICKWALL_KEYS = (
    "input_strength",
    "b_mean",
    "b_std",
    "h_mean",
    "h_std",
    "h_min",
)


@dataclass(frozen=True)
class PhotonicArchitecture:
    """Size and layer layout of the photonic reservoir.

    Attributes
    ----------
    n_modes : int
        Number of optical modes (even).
    n_photons : int
        Number of photons; one photon every ``n_modes // n_photons`` modes.
    n_layers : int
        Brickwall layers; layer ``l`` encodes data when ``l % encoding_period == 0``.
    encoding_period : int
        Encoding period, as in the qubit reservoir (4 = 1 encoding : 3 dynamics layers).
    input_shift : int
        Block ``j`` of encoding layer ``k`` receives ``u[(k * input_shift + j) % dim]``.
    angle_scale : float
        Angle multiplier ``s`` (2 matches the qubit ``figure1`` convention).
    """

    n_modes: int
    n_photons: int
    n_layers: int
    encoding_period: int
    input_shift: int
    angle_scale: float

    def __post_init__(self) -> None:
        if self.n_modes % 2 or self.n_modes < 4:
            raise ValueError(f"n_modes must be even and >= 4, got {self.n_modes}")
        if self.n_photons < 1 or self.n_modes % self.n_photons:
            raise ValueError(
                f"n_photons ({self.n_photons}) must divide n_modes ({self.n_modes})"
            )
        if self.encoding_period < 1 or self.n_layers < 1:
            raise ValueError("n_layers and encoding_period must be >= 1")

    @classmethod
    def from_config(cls, section: dict, **overrides) -> PhotonicArchitecture:
        """Build the architecture from ``cfg["merlin_reservoir"]``, requiring every key."""
        values = {**section, **overrides}
        missing = [key for key in REQUIRED_ARCHITECTURE_KEYS if key not in values]
        if missing:
            raise KeyError(f"Missing merlin_reservoir keys: {missing}")
        return cls(
            n_modes=int(values["n_modes"]),
            n_photons=int(values["n_photons"]),
            n_layers=int(values["n_layers"]),
            encoding_period=int(values["encoding_period"]),
            input_shift=int(values["input_shift"]),
            angle_scale=float(values["angle_scale"]),
        )

    def layer_pairs(self, layer: int) -> list[tuple[int, int]]:
        """Mode pairs of one brickwall layer (offset by one mode on odd layers)."""
        return [(a, a + 1) for a in range(layer % 2, self.n_modes - 1, 2)]

    def is_encoding_layer(self, layer: int) -> bool:
        return layer % self.encoding_period == 0

    @property
    def window(self) -> int:
        """Inputs per encoding layer when it falls on an even layer (``n_modes / 2``)."""
        return self.n_modes // 2

    @property
    def input_state(self) -> list[int]:
        """One photon every ``n_modes // n_photons`` modes, starting at mode 0."""
        spacing = self.n_modes // self.n_photons
        return [1 if mode % spacing == 0 else 0 for mode in range(self.n_modes)]

    def disjoint_pairs(self) -> list[tuple[int, int]]:
        return [(a, a + 1) for a in range(0, self.n_modes, 2)]

    def neighbour_pairs(self) -> list[tuple[int, int]]:
        return [(a, a + 1) for a in range(self.n_modes - 1)]


def _sample_brickwall_angle(rng, mean: float, std: float) -> float:
    return float(rng.normal(mean, std))


def build_photonic_circuit(
    arch: PhotonicArchitecture, setting: dict, seed: int, input_dim: int
) -> tuple[pcvl.Circuit, np.ndarray]:
    """Fixed reservoir circuit and the input index fed to each encoding parameter.

    Parameters
    ----------
    arch : PhotonicArchitecture
        Reservoir layout.
    setting : dict
        ``kind`` (``"brickwall"`` or ``"haar"``) plus, for both kinds, ``input_strength``
        and the brickwall hyperparameters ``b_mean, b_std, h_mean, h_std, h_min``
        (phases of the encoding blocks also use ``h`` for ``"haar"``).
    seed : int
        Reservoir realisation seed.
    input_dim : int
        Length of the input vectors.

    Returns
    -------
    tuple
        ``(circuit, feed)``: the Perceval circuit whose input parameters are named
        ``x0000, x0001, ...`` in creation order, and ``feed[k]``, the input index of
        parameter ``k``.

    Raises
    ------
    KeyError
        If a required setting key is missing.
    ValueError
        If ``setting["kind"]`` is unknown.
    """
    kind = setting["kind"]
    if kind not in RESERVOIR_KINDS:
        raise ValueError(f"Unknown reservoir kind {kind!r}; expected {RESERVOIR_KINDS}")
    missing = [key for key in REQUIRED_BRICKWALL_KEYS if key not in setting]
    if missing:
        raise KeyError(f"Missing reservoir setting keys: {missing}")
    rng = np.random.default_rng([int(seed), 17])
    scale = arch.angle_scale
    circuit = pcvl.Circuit(arch.n_modes)
    feed: list[int] = []
    encoding_layer = 0

    def phase(mode: int) -> None:
        h = _sample_brickwall_angle(rng, setting["h_mean"], setting["h_std"])
        while h <= setting["h_min"]:
            h = _sample_brickwall_angle(rng, setting["h_mean"], setting["h_std"])
        circuit.add(mode, pcvl.PS(scale * h))

    for layer in range(arch.n_layers):
        encoding = arch.is_encoding_layer(layer)
        if kind == "haar" and not encoding:
            unitary = unitary_group.rvs(arch.n_modes, random_state=rng)
            circuit.add(0, pcvl.Unitary(pcvl.Matrix(unitary)))
            continue
        for block, (mode_a, mode_b) in enumerate(arch.layer_pairs(layer)):
            phase(mode_b)
            if encoding:
                name = f"x{len(feed):04d}"
                circuit.add(mode_a, pcvl.BS(theta=pcvl.P(name)))
                feed.append(
                    (encoding_layer * arch.input_shift + block) % int(input_dim)
                )
            else:
                b = _sample_brickwall_angle(rng, setting["b_mean"], setting["b_std"])
                circuit.add(mode_a, pcvl.BS(theta=scale * b))
            phase(mode_b)
        if encoding:
            encoding_layer += 1
    return circuit, np.asarray(feed, dtype=int)


class PhotonicReservoir:
    """Fixed photonic reservoir: MerLin ``QuantumLayer`` with threshold detectors.

    Parameters
    ----------
    arch : PhotonicArchitecture
        Reservoir layout.
    setting : dict
        Reservoir kind and hyperparameters (see `build_photonic_circuit`).
    seed : int
        Reservoir realisation seed.
    input_dim : int
        Length of the input vectors.
    dtype : torch.dtype
        Real dtype of the simulation.
    """

    def __init__(
        self,
        arch: PhotonicArchitecture,
        setting: dict,
        seed: int,
        input_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.arch = arch
        self.input_strength = float(setting["input_strength"])
        circuit, self.feed = build_photonic_circuit(arch, setting, seed, input_dim)
        experiment = pcvl.Experiment(circuit)
        for mode in range(arch.n_modes):
            experiment.add(mode, pcvl.Detector.threshold())
        self.layer = ml.QuantumLayer(
            experiment=experiment,
            input_parameters=["x"],
            input_state=arch.input_state,
            n_photons=arch.n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
            dtype=dtype,
        )
        self.layer.eval()
        self.dtype = dtype
        clicks = np.array([list(key) for key in self.layer.output_keys], dtype=float)
        if clicks.max() > 1:
            raise RuntimeError("Threshold detectors returned photon counts above 1")
        # (n_outcomes, n_modes) click indicator of every detected pattern.
        self.clicks = torch.as_tensor(clicks, dtype=dtype)

    def encoding_angles(self, inputs: np.ndarray) -> torch.Tensor:
        """``theta = s a_in u`` for every encoding parameter, shape ``(batch, n_params)``."""
        angles = self.arch.angle_scale * self.input_strength * inputs[:, self.feed]
        return torch.as_tensor(angles, dtype=self.dtype)

    def pair_click_probabilities(
        self, inputs: np.ndarray, pairs: list[tuple[int, int]], chunk_size: int
    ) -> np.ndarray:
        """Exact 4-outcome click marginals of mode pairs.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, len(pairs), 4)``, outcome index ``2 * click_a + click_b``.
        """
        first = torch.as_tensor([a for a, _ in pairs])
        second = torch.as_tensor([b for _, b in pairs])
        both = self.clicks[:, first] * self.clicks[:, second]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(inputs), chunk_size):
                probs = self.layer(
                    self.encoding_angles(inputs[start : start + chunk_size])
                )
                single = probs @ self.clicks
                p11 = probs @ both
                p10 = single[:, first] - p11
                p01 = single[:, second] - p11
                p00 = 1.0 - single[:, first] - single[:, second] + p11
                chunks.append(torch.stack([p00, p01, p10, p11], dim=-1))
        marginals = torch.cat(chunks).double().numpy()
        # Float32 round-off can leave values of order -1e-7; probabilities are clipped and
        # renormalised so multinomial sampling is well defined.
        marginals = np.clip(marginals, 0.0, None)
        return marginals / marginals.sum(axis=-1, keepdims=True)


def reservoir_from_config(
    cfg: dict, seed: int, input_dim: int, setting: dict | None = None, **overrides
) -> PhotonicReservoir:
    """Photonic reservoir of ``cfg["merlin_reservoir"]`` with Table I (mapped) by default.

    ``setting`` defaults to ``{"kind": "brickwall", **cfg["hyperparameters"]}``; ``J`` keys
    are ignored because the interaction has no photonic counterpart.
    """
    from .qelm import real_dtype_from_config

    arch = PhotonicArchitecture.from_config(cfg["merlin_reservoir"], **overrides)
    if setting is None:
        setting = {"kind": "brickwall", **cfg["hyperparameters"]}
    return PhotonicReservoir(
        arch, setting, seed, input_dim, real_dtype_from_config(cfg)
    )


def single_mode_frequencies(pair_frequencies: np.ndarray) -> np.ndarray:
    """Marginalise pair click frequencies to ``(n_samples, 2 * n_pairs, 2)`` per mode."""
    grid = pair_frequencies.reshape(*pair_frequencies.shape[:2], 2, 2)
    first = grid.sum(axis=3)
    second = grid.sum(axis=2)
    return np.stack([first, second], axis=2).reshape(pair_frequencies.shape[0], -1, 2)


def click_statistics(pair_frequencies: np.ndarray, weight: int) -> np.ndarray:
    """Click readout: per-mode clicks (weight 1) or ``(n_a, n_b, n_a n_b)`` per pair (weight 2)."""
    if weight == 1:
        return single_mode_frequencies(pair_frequencies)[:, :, 1]
    clicks_a = pair_frequencies[:, :, 2] + pair_frequencies[:, :, 3]
    clicks_b = pair_frequencies[:, :, 1] + pair_frequencies[:, :, 3]
    both = pair_frequencies[:, :, 3]
    return np.stack([clicks_a, clicks_b, both], axis=2).reshape(
        len(pair_frequencies), -1
    )


def fit_click_eigentasks(train_sets: np.ndarray, shots: int) -> EigentaskBasis:
    """Eigentasks of every subset; a subset with ``k`` click outcomes has ``k - 1`` tasks."""
    coefficients, nsr_values, subset_ids = [], [], []
    n_effects = train_sets.shape[2]
    for subset in range(train_sets.shape[1]):
        coeff, nsr = subset_eigentasks(train_sets[:, subset], shots, n_effects)
        coefficients.append(coeff)
        nsr_values.append(nsr)
        subset_ids.append(np.full(nsr.shape, subset))
    return EigentaskBasis(
        coefficients=coefficients,
        nsr=np.concatenate(nsr_values),
        subset_index=np.concatenate(subset_ids),
    )


def parse_click_readout(name: str) -> tuple[int, str, str]:
    """Split ``w2_etcut_scaled`` into ``(2, "etcut", "scaled")`` and validate it."""
    parts = name.split("_")
    if (
        len(parts) != 3
        or parts[0] not in ("w1", "w2")
        or parts[1] not in CLICK_READOUT_KINDS
        or parts[2] not in READOUT_SCALINGS
    ):
        raise ValueError(f"Invalid click readout name {name!r}")
    return int(parts[0][1]), parts[1], parts[2]


def click_readout_features(
    name: str,
    train_pair_freq: np.ndarray,
    test_pair_freq: np.ndarray,
    shots: int,
    nsr_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Photonic counterpart of `lib.landsat_task.readout_features`.

    Readout names are ``w{1,2}_{click,et,etcut}_{unit,scaled}``: ``click`` = click
    probabilities (signal-aware scaling for ``scaled``); ``et``/``etcut`` = local
    eigentasks on single modes (w1, 1 per mode) or disjoint pairs (w2, 3 per pair), with
    the NSR cutoff and NSR-aware scaling of the qubit readouts.
    """
    weight, kind, scaling = parse_click_readout(name)
    if kind == "click":
        train = click_statistics(train_pair_freq, weight)
        test = click_statistics(test_pair_freq, weight)
        if scaling == "unit":
            train, test, _ = unit_scale(train, test)
        else:
            train, test = signal_scale(train, test)
        info: dict = {}
    else:
        if weight == 1:
            train_sets = single_mode_frequencies(train_pair_freq)
            test_sets = single_mode_frequencies(test_pair_freq)
        else:
            train_sets, test_sets = train_pair_freq, test_pair_freq
        basis = fit_click_eigentasks(train_sets, shots)
        train, test, info = eigentask_readout(
            basis, train_sets, test_sets, shots, kind, scaling, nsr_cutoff
        )
    info["n_features"] = int(train.shape[1])
    return train, test, info


def neighbour_click_features(pair_probabilities: np.ndarray) -> np.ndarray:
    """Exact per-mode clicks (``n_modes``) and neighbour coincidences (``n_modes - 1``).

    ``pair_probabilities`` are the click marginals of the open-chain neighbour pairs
    ``(0, 1), ..., (m - 2, m - 1)``.
    """
    singles = np.concatenate(
        [
            pair_probabilities[:, :, 2] + pair_probabilities[:, :, 3],
            (pair_probabilities[:, -1, 1] + pair_probabilities[:, -1, 3])[:, None],
        ],
        axis=1,
    )
    return np.concatenate([singles, pair_probabilities[:, :, 3]], axis=1)


def sampled_neighbour_click_features(
    pair_probabilities: np.ndarray, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Finite-shot estimate of `neighbour_click_features` with ``shots`` per sample.

    Each feature is an independent binomial frequency with its exact mean and shot
    variance; correlations between features of the same shots are not modelled, as in the
    qubit ``shadow`` noise condition.
    """
    exact = neighbour_click_features(pair_probabilities)
    return rng.binomial(int(shots), np.clip(exact, 0.0, 1.0)) / int(shots)
