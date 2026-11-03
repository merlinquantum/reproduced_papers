"""Official Perceval sampling of fully distinguishable tagged photons."""

from __future__ import annotations

import time

import numpy as np
import perceval as pc


def fully_distinguishable_state(occupations: list[int]):
    """Assign a distinct Perceval noise tag to every input photon.

    Parameters
    ----------
    occupations : list[int]
        Nonnegative integer occupations in spatial-mode order.

    Returns
    -------
    NoisyFockState
        Photon tags 0 through n-1; equal output modes remain allowed.
    """
    values = np.asarray(occupations)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError(
            "Input occupations must be nonnegative integers in at least two modes"
        )
    photons = int(values.sum())
    if not 1 <= photons <= 256:
        raise ValueError(
            "Distinct integer noise tags require between 1 and 256 photons"
        )
    return pc.BasicState(values.tolist(), list(range(photons)))


def sample_distinguishable(
    circuit,
    occupations: list[int],
    samples: int,
    seed: int,
    implementation: str = "native",
) -> tuple[np.ndarray, dict]:
    """Sample an explicit fully distinguishable state through public Perceval APIs.

    Parameters
    ----------
    circuit : perceval.Circuit
        Lossless spatial circuit shared with the Indist control.
    occupations : list[int]
        Input photon counts, with the same mode count as the circuit.
    samples : int
        Positive event budget. All output collisions are retained.
    seed : int
        Perceval event seed; thread settings are configured by the caller.
    implementation : str
        ``native`` uses SimulatedComputer, which selects the exqalibur sampling
        wrapper in Perceval 1.3.1. ``python`` uses its documented Python simulator
        as an independent API-path check. Default is ``native``.

    Returns
    -------
    tuple[np.ndarray, dict]
        Integer occupation events and backend/timing metadata. No latent
        normalization, training-pool replacement, or file writes are performed.
    """
    state = fully_distinguishable_state(occupations)
    if state.m != circuit.m:
        raise ValueError("Input occupations and circuit mode counts differ")
    if type(samples) is not int or samples < 1:
        raise ValueError("The sample budget must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("The event seed must be a uint32 integer")
    if implementation not in {"native", "python"}:
        raise ValueError("Implementation must be native or python")
    mixture = pc.SVDistribution({state: 1})
    started = time.perf_counter()
    pc.random_seed(seed)
    if implementation == "native":
        experiment = pc.Experiment(circuit)
        experiment.with_input(mixture)
        experiment.min_detected_photons_filter(state.n)
        result = pc.SimulatedComputer("CliffordClifford2017").samples(
            experiment, max_samples=samples, max_shots=samples
        )
        api = "SimulatedComputer.samples -> ExqaliburNoisySamplingSimulator -> exqalibur.SamplingSimulator"
    else:
        from perceval.simulators import NoisySamplingSimulator

        simulator = NoisySamplingSimulator(pc.Clifford2017Backend())
        simulator.set_circuit(circuit)
        simulator.set_selection(min_detected_photons_filter=state.n)
        simulator.sleep_between_batches = 0
        result = simulator.samples(mixture, max_samples=samples, max_shots=samples)
        api = "NoisySamplingSimulator.samples -> tag groups -> Clifford2017Backend.samples -> merge"
    sampled = time.perf_counter()
    counts = np.asarray([list(event) for event in result["results"]], dtype=np.uint16)
    if counts.shape != (samples, state.m) or not np.all(counts.sum(axis=1) == state.n):
        raise ValueError(
            "Perceval returned an incomplete or photon-nonconserving event array"
        )
    if result["global_perf"] != 1:
        raise ValueError("Lossless tagged sampling unexpectedly rejected events")
    metadata = {
        "perceval_api": api,
        "input_state": str(state),
        "noise_tags": list(range(state.n)),
        "global_perf": float(result["global_perf"]),
        "events": samples,
        "event_seed": seed,
        "sampling_and_setup_seconds": sampled - started,
        "conversion_seconds": time.perf_counter() - sampled,
        "elapsed_seconds": time.perf_counter() - started,
        "collisions_retained": True,
        "detectors": "Default ideal photon-number-resolving",
        "extra_noise_or_postselection": False,
    }
    return counts, metadata
