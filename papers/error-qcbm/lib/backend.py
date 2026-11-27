"""MerLin probability and finite-shot backends for the photonic QCBM.

Perceval supplies circuit, detector and noise descriptions. MerLin performs
the optical evolution, source-noise mixture, loss transform and sampling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from time import perf_counter

import merlin as ml
import numpy as np
import perceval as pcvl
import torch

DEMO_INPUT_STATE = (0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)


def build_clements_circuit(modes: int = 12) -> pcvl.Circuit:
    """Build the author's single variational Clements block.

    Parameters
    ----------
    modes : int
        Number of optical modes. Default value is 12.

    Returns
    -------
    perceval.Circuit
        Rectangular interferometer with two free phases per MZI, in author
        traversal order: ``phi_a0_1``, ``phi_b0_1``, and so on.
    """
    if isinstance(modes, bool) or not isinstance(modes, Integral) or modes < 2:
        raise ValueError("modes must be an integer of at least two")
    # build_circuit replaces the old catalog.generate API used by the authors.
    block = pcvl.GenericInterferometer(
        int(modes), lambda index: pcvl.catalog["mzi phase last"].build_circuit(i=index)
    )
    for parameter in block.get_parameters():
        parameter.name += "_1"
    circuit = pcvl.Circuit(int(modes), name="photonic_qcbm")
    circuit.add(0, block)
    expected_parameters = modes * (modes - 1)
    if len(circuit.get_parameters()) != expected_parameters:
        raise RuntimeError("Clements circuit has an unexpected parameter count")
    return circuit


def initialize_parameters(
    parameter_count: int, seeds: Sequence[int] = tuple(range(20))
) -> np.ndarray:
    """Draw paired initial phases using the author's normal distribution.

    Parameters
    ----------
    parameter_count : int
        Number of variational phases.
    seeds : sequence of int
        Independent NumPy legacy RandomState seeds. Default value is 0 to 19.

    Returns
    -------
    numpy.ndarray
        One row per seed, drawn from ``Normal(0, 2*pi)``. RandomState preserves
        the author's ``np.random.normal`` sequence for a given explicit seed.
    """
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, Integral)
        or parameter_count <= 0
    ):
        raise ValueError("parameter_count must be a positive integer")
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one initialization seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise ValueError("Initialization seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Initialization seeds must be distinct")
    return np.stack(
        [
            np.random.RandomState(int(seed)).normal(0, 2 * np.pi, parameter_count)
            for seed in seeds
        ]
    )


class PhotonicBackend:
    """Evaluate one circuit with an explicit MerLin noise profile.

    Parameters
    ----------
    input_state : sequence of int
        Occupations of the injected Fock state.
    profile : dict
        Required keys are ``name``, ``transmittance`` and
        ``indistinguishability``. Optional ``g2`` defaults to 0, and
        ``detector`` defaults to ``"threshold"`` (``"pnr"`` is also supported).
        The two-photon source extension restricts g2 to [0, 0.5].
        Brightness is fixed to one: transmittance already represents total
        system efficiency, and must not be applied twice. A nonzero g2 uses
        Perceval's default distinguishable extra-photon model.
    dtype : torch.dtype or str
        Probability precision, float64 or float32. Default value is float64.
    device : str or torch.device
        PyTorch compute device. Default value is ``"cpu"``.
    circuit : perceval.Circuit or None
        Custom circuit for analytic controls. Default value is None, which
        builds the author's one-block Clements circuit for the input mode count.
    native_chunk_shots : int or None
        Maximum native shots per call, aggregating counts before estimation.
        Default value is None, which uses one native call per evaluation.
    native_count_accumulation : str
        python for stock MerLin or vectorized for the project-local exact count
        adapter. Default value is python. Vectorized validation covers CPU float64.

    Notes
    -----
    All outputs include vacuum and all lower photon/click sectors. Sampling
    uses an initial attempt budget, with no accepted-event post-selection.
    ``layer`` exposes the native differentiable MerLin module; ``exact`` and
    ``sample`` are inference methods and run without autograd.
    """

    def __init__(
        self,
        input_state: Sequence[int],
        profile: dict,
        dtype: torch.dtype | str = torch.float64,
        device: str | torch.device = "cpu",
        circuit: pcvl.Circuit | None = None,
        native_chunk_shots: int | None = None,
        native_count_accumulation: str = "python",
    ):
        if native_count_accumulation not in ("python", "vectorized"):
            raise ValueError("native_count_accumulation must be python or vectorized")
        self.native_count_accumulation = native_count_accumulation
        if native_chunk_shots is not None and (
            isinstance(native_chunk_shots, bool)
            or not isinstance(native_chunk_shots, Integral)
            or native_chunk_shots <= 0
        ):
            raise ValueError("native_chunk_shots must be a positive integer or None")
        self.native_chunk_shots = native_chunk_shots
        occupations = tuple(input_state)
        if (
            len(occupations) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in occupations
            )
            or sum(occupations) < 1
        ):
            raise ValueError("input_state must contain nonnegative integer occupations")
        self.input_state = tuple(int(value) for value in occupations)
        if dtype in ("float64", torch.float64):
            self.dtype = torch.float64
        elif dtype in ("float32", torch.float32):
            self.dtype = torch.float32
        else:
            raise ValueError("Only float64 and float32 are supported")
        self.device = torch.device(device)
        self.profile = dict(profile)
        if not isinstance(self.profile["name"], str) or not self.profile["name"]:
            raise ValueError("profile.name must be a nonempty string")
        for name in ("transmittance", "indistinguishability"):
            value = self.profile[name]
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"profile.{name} must be a probability in [0, 1]")
        g2 = self.profile.get("g2", 0.0)
        if isinstance(g2, bool) or not math.isfinite(g2) or not 0 <= g2 <= 0.5:
            raise ValueError(
                "profile.g2 must be in [0, 0.5] for MerLin's two-photon source model"
            )
        detector = self.profile.get("detector", "threshold")
        if detector not in ("threshold", "pnr"):
            raise ValueError("profile.detector must be threshold or pnr")
        self.profile.update(g2=float(g2), detector=detector, brightness=1.0)
        self.circuit = (
            build_clements_circuit(len(self.input_state))
            if circuit is None
            else circuit.copy()
        )
        if self.circuit.m != len(self.input_state):
            raise ValueError("Circuit and input_state mode counts must match")
        self.parameter_names = [
            parameter.name for parameter in self.circuit.get_parameters()
        ]
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("Circuit parameter names must be unique")
        noise = pcvl.NoiseModel(
            brightness=1.0,
            transmittance=float(self.profile["transmittance"]),
            indistinguishability=float(self.profile["indistinguishability"]),
            g2=float(g2),
        )
        experiment = pcvl.Experiment(self.circuit)
        experiment.with_input(pcvl.BasicState(self.input_state))
        detector_factory = (
            pcvl.Detector.threshold if detector == "threshold" else pcvl.Detector.pnr
        )
        for mode in range(self.circuit.m):
            experiment.add(mode, detector_factory())
        self.layer = ml.QuantumLayer(
            input_size=0,
            experiment=experiment,
            noise=noise,
            trainable_parameters=self.parameter_names,
            measurement_strategy=ml.MeasurementStrategy.probs(ml.ComputationSpace.FOCK),
            dtype=self.dtype,
            device=self.device,
        )
        if self.layer.theta_names != self.parameter_names:
            raise RuntimeError(
                "MerLin's phase order differs from circuit traversal order"
            )
        if sum(parameter.numel() for parameter in self.layer.parameters()) != len(
            self.parameter_names
        ):
            raise RuntimeError(
                "MerLin's trainable phase count differs from the circuit"
            )
        self.timings: dict[str, float | str] = {}
        # Native initialization is random; initialize both the MerLin tensors and
        # exposed circuit consistently until the caller selects a paired seed.
        self.set_parameters(np.zeros(len(self.parameter_names)))

    def set_parameters(
        self, values: Sequence[float] | np.ndarray | torch.Tensor
    ) -> None:
        """Bind phases in the documented circuit order.

        Parameters
        ----------
        values : sequence of float or numpy.ndarray or torch.Tensor
            Finite one-dimensional vector with exactly one entry per phase.
        """
        vector = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if vector.ndim != 1 or vector.numel() != len(self.parameter_names):
            raise ValueError(f"Expected {len(self.parameter_names)} phase values")
        if not torch.isfinite(vector).all():
            raise ValueError("Phase values must be finite")
        parameter_map = dict(self.layer.named_parameters())
        with torch.no_grad():
            for index, name in enumerate(self.parameter_names):
                parameter_map[name].copy_(vector[index].reshape_as(parameter_map[name]))
        for parameter, value in zip(
            self.circuit.get_parameters(), vector.detach().cpu().tolist(), strict=True
        ):
            parameter.set_value(value)

    def _probabilities(
        self, shots: int = 0
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        from .sampling import count_accumulation

        if (
            shots
            and self.native_count_accumulation == "vectorized"
            and (self.device.type != "cpu" or self.dtype != torch.float64)
        ):
            raise ValueError("Vector counts are validated only for CPU float64")
        if (
            shots
            and self.native_count_accumulation == "vectorized"
            and self.profile["g2"] != 0
        ):
            raise ValueError(
                "Vector counts have not been validated for g2 source noise"
            )
        with (
            torch.no_grad(),
            count_accumulation(self.native_count_accumulation if shots else "python"),
        ):
            values = self.layer(shots=shots).reshape(-1).detach().cpu()
        keys = [tuple(int(value) for value in key) for key in self.layer.output_keys]
        if len(keys) != len(values) or len(set(keys)) != len(keys):
            raise RuntimeError("MerLin output key order or uniqueness is invalid")
        tolerance = 1e-10 if self.dtype == torch.float64 else 2e-5
        if not torch.isfinite(values).all() or (values < 0).any():
            raise RuntimeError("MerLin produced invalid probabilities")
        if abs(values.sum().item() - 1.0) > tolerance:
            raise RuntimeError("MerLin probabilities do not sum to one")
        return keys, values

    def exact(self) -> dict[tuple[int, ...], float]:
        """Return the complete exact detector distribution from MerLin.

        Returns
        -------
        dict[tuple[int, ...], float]
            Output Fock occupations or threshold clicks, including vacuum.
        """
        started = perf_counter()
        keys, values = self._probabilities()
        self.timings["exact_seconds"] = perf_counter() - started
        return dict(zip(keys, values.tolist(), strict=True))

    def sample(self, initial_shots: int, seed: int) -> dict[tuple[int, ...], int]:
        """Draw MerLin shots with stock or explicitly selected count accumulation.

        Parameters
        ----------
        initial_shots : int
            Initial circuit attempts, including vacuum outcomes.
        seed : int
            Torch sampling seed; the caller's global RNG state is restored.

        Returns
        -------
        dict[tuple[int, ...], int]
            Native MerLin counts with total exactly ``initial_shots``.
        """
        counts = self.sample_native(initial_shots, seed)
        self.timings["sample_seconds"] = self.timings["native_sample_seconds"]
        self.timings["sample_engine"] = (
            "merlin_native_multinomial_vectorized_counts"
            if self.native_count_accumulation == "vectorized"
            else "merlin_native_QuantumLayer_shots"
        )
        return counts

    def sample_native(
        self, initial_shots: int, seed: int
    ) -> dict[tuple[int, ...], int]:
        """Draw native MerLin shots and recover integer detector counts.

        Parameters
        ----------
        initial_shots : int
            Number of initial circuit attempts, including vacuum outcomes.
        seed : int
            Torch sampling seed. The caller's global random state is restored.

        Returns
        -------
        dict[tuple[int, ...], int]
            Nonzero observed counts with total exactly ``initial_shots``.
        """
        if (
            isinstance(initial_shots, bool)
            or not isinstance(initial_shots, Integral)
            or initial_shots <= 0
        ):
            raise ValueError("initial_shots must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        # MerLin returns frequencies. At float32, sufficiently large integer
        # counts cannot be recovered exactly, so require float64 for large runs.
        if self.dtype == torch.float32 and initial_shots > 2**24:
            raise ValueError("Use float64 for more than 2**24 initial shots")
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index or torch.cuda.current_device()]
        started = perf_counter()
        aggregate = {}
        remaining = int(initial_shots)
        chunks = 0
        from .sampling import count_accumulation

        # Serialize the whole fork/seed/draw/restore sequence, including chunks.
        # Forward-only locking would let another project thread reseed this RNG.
        with count_accumulation("python"), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            while remaining:
                budget = min(remaining, self.native_chunk_shots or int(initial_shots))
                keys, frequencies = self._probabilities(shots=budget)
                recovered = frequencies.to(torch.float64) * budget
                counts = recovered.round().to(torch.int64)
                quantization_tolerance = 1e-6 if self.dtype == torch.float64 else 1.0
                if (recovered - counts).abs().max().item() > quantization_tolerance:
                    raise RuntimeError(
                        "MerLin sampling output is not quantized by initial_shots"
                    )
                if counts.sum().item() != budget:
                    raise RuntimeError(
                        "MerLin sampling did not preserve the initial shot budget"
                    )
                for key, count in zip(keys, counts.tolist(), strict=True):
                    if count:
                        aggregate[key] = aggregate.get(key, 0) + int(count)
                remaining -= budget
                chunks += 1
        if sum(aggregate.values()) != initial_shots:
            raise RuntimeError(
                "Native chunk aggregation did not preserve the initial shot budget"
            )
        self.timings["native_sample_seconds"] = perf_counter() - started
        self.timings["native_chunks"] = chunks
        return aggregate
