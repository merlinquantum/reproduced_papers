"""MerLin photonic adaptation of the paper's quantum reservoir.

Photonic status: ``PARTIAL_MERLIN_TRANSLATION`` (see LOG.md and README).

What is translated
------------------
The scientific role of the paper's reservoir is a *fixed, untrained* quantum map
that turns a 3-step feature window into a small readout vector on which only a
linear ridge layer is trained. The photonic counterpart implemented here keeps
all of that:

=========================  ==========================================  ==========================================
Role                       Qubit reservoir (paper)                     Photonic reservoir (this module)
=========================  ==========================================  ==========================================
Fixed scrambling dynamics  ``exp(-i tau H)``, ``H`` a fully connected   frozen Haar-random MZI mesh
                           transverse-field Ising Hamiltonian           (``add_entangling_layer``, never optimised)
Feature encoding           ``RY(pi * x_j)`` on ``n1 = 7`` input qubits  phase shifters ``pi * x_j`` on 7 input modes
Temporal structure         encode, evolve, encode, evolve, encode,      encode / mesh blocks applied in lag order
                           evolve, measure                             on a shared 10-mode register
Memory register            ``n2 = 3`` hidden qubits never re-encoded    3 hidden modes never re-encoded
Nonlinearity               multi-qubit entangling dynamics             multi-photon interference (>= 2 photons, so
                                                                       the Fock-space map is not a trivial
                                                                       single-photon linear map)
Readout                    ``<Z_j>`` on all 10 qubits                  ``<n_j>`` on all 10 modes
                           (QR2: two evolution times -> 20 features)   (PQR2: two final meshes -> 20 features)
Training                   rolling ridge, ``delta = 1e-8``, no bias    identical
=========================  ==========================================  ==========================================

What is not translated
----------------------
The qubit reservoir *discards* its input qubits between steps (``rho_h =
Tr_I[...]``), which makes the step map non-unitary and bounds the memory. A
photonic equivalent would have to detect the input-mode photons, herald on the
photon number left in the hidden modes, and re-inject fresh photons into a
*mixed* hidden state. ``merlin.QuantumLayer`` accepts only a pure Fock
``input_state``, so a mixed hidden state cannot be fed back through the public
API, and ``MeasurementStrategy.partial`` returns the branch decomposition but
provides no way to re-enter it as an input. Implementing that would require a
custom Fock-space density-matrix simulator built on low-level Perceval, i.e. a
second simulator rather than a MerLin adaptation. The photonic register here
therefore retains information the qubit reservoir throws away, which makes this
translation an upper bound on the qubit architecture's memory rather than a
faithful copy of it. This is recorded as the single material deviation.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


def default_input_state(n_modes: int, n_photons: int) -> list[int]:
    """Spread photons evenly over the mode register.

    Photon placement is a modelling choice: the photons are laid down at equal
    spacing starting from mode 0, so every photon sits on an *encoded* mode when
    ``n_photons <= n_input``, putting the encoding phases inside the photon light
    cone while leaving the hidden modes to be populated by the mesh.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    n_photons : int
        Number of photons.

    Returns
    -------
    list of int
        Occupation numbers summing to ``n_photons``.

    Raises
    ------
    ValueError
        If ``n_photons`` is below 2 (a single-photon circuit is a trivial linear
        map and must not be presented as a photonic implementation) or exceeds
        ``n_modes``.
    """
    if n_photons < 2:
        raise ValueError(
            f"n_photons={n_photons} < 2: a single-photon linear-optical circuit is "
            "a trivial linear map, not a photonic reservoir"
        )
    if n_photons > n_modes:
        raise ValueError(f"n_photons={n_photons} exceeds n_modes={n_modes}")
    state = [0] * n_modes
    spacing = n_modes // n_photons
    for photon in range(n_photons):
        state[photon * spacing] = 1
    return state


class PhotonicReservoir:
    """Frozen MerLin photonic reservoir over a lagged feature window.

    Parameters
    ----------
    n_input : int
        Number of feature-carrying modes ``n1``.
    n_lags : int
        Memory depth ``k``; one encoding block per lag.
    n_modes : int
        Total mode count. Default value is 10.
    n_photons : int
        Photon number; must be at least 2. Default value is 3.
    seed : int
        Seed for the frozen mesh phases. Different seeds are different reservoir
        instances, the photonic analogue of a different coupling draw. Default
        value is 0.
    readout : {"mode_expectations", "probabilities"}
        ``mode_expectations`` gives one feature per mode, the direct analogue of
        the paper's ``<Z_j>`` readout. ``probabilities`` gives the full
        unbunched outcome distribution. Default value is "mode_expectations".
    ensemble : bool
        When True, build a second reservoir that shares every mesh except the
        last and concatenate the two readouts, mirroring QR2's ``{tau, tau/2}``
        ensemble. Default value is False.
    encoding_scale : float
        Phase-shifter scale applied to each feature. Default value is ``pi``,
        matching ``RY(pi * x)`` in the reference code.
    input_state : list of int or None
        Explicit Fock input state. Derived from ``n_modes`` and ``n_photons``
        when omitted. Default value is None.

    Attributes
    ----------
    n_readout : int
        Width of the readout vector.
    metadata : dict
        Hardware-aware description of the circuit for result tables.
    """

    def __init__(
        self,
        n_input: int,
        n_lags: int,
        *,
        n_modes: int = 10,
        n_photons: int = 3,
        seed: int = 0,
        readout: str = "mode_expectations",
        ensemble: bool = False,
        encoding_scale: float = math.pi,
        input_state: list[int] | None = None,
    ) -> None:
        import merlin as ml

        if n_input > n_modes:
            raise ValueError(f"n_input={n_input} exceeds n_modes={n_modes}")
        self.n_input = n_input
        self.n_lags = n_lags
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.readout = readout
        self.ensemble = ensemble
        self.input_state = list(input_state) if input_state else default_input_state(
            n_modes, n_photons
        )
        if sum(self.input_state) != n_photons:
            raise ValueError(
                f"input_state {self.input_state} carries {sum(self.input_state)} "
                f"photons but n_photons={n_photons}"
            )

        builder = ml.CircuitBuilder(n_modes=n_modes)
        encoded_modes = list(range(n_input))
        # Name every mesh so the ensemble variant can share meshes by name rather
        # than by guessing the block order inside one flat parameter tensor.
        # MerLin 0.4 groups parameters whose names differ only by a trailing
        # integer into ONE tensor, so `mesh_0`/`mesh_1` would silently merge;
        # letter suffixes keep the meshes addressable separately.
        self._mesh_names = [
            f"mesh{chr(ord('A') + index)}" for index in range(n_lags + 1)
        ]
        for index in range(n_lags):
            builder.add_entangling_layer(name=self._mesh_names[index])
            builder.add_angle_encoding(modes=encoded_modes, scale=float(encoding_scale))
        builder.add_entangling_layer(name=self._mesh_names[-1])
        self._final_mesh_name = self._mesh_names[-1]

        if readout == "mode_expectations":
            strategy = ml.MeasurementStrategy.mode_expectations(
                computation_space=ml.ComputationSpace.UNBUNCHED
            )
        elif readout == "probabilities":
            strategy = ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            )
        else:
            raise ValueError(f"unknown readout {readout!r}")

        def make_layer(layer_seed: int):
            torch.manual_seed(layer_seed)
            layer = ml.QuantumLayer(
                input_size=n_lags * n_input,
                builder=builder,
                input_state=self.input_state,
                n_photons=n_photons,
                measurement_strategy=strategy,
            )
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            return layer

        self._layers = [make_layer(seed)]
        if ensemble:
            # QR2 reads the same reservoir out after two different final
            # evolutions, so the photonic ensemble shares every mesh except the
            # last one and redraws only that. Named meshes make the sharing exact.
            second = make_layer(seed + 10_000)
            shared_phases = dict(self._layers[0].named_parameters())
            with torch.no_grad():
                for name, parameter in second.named_parameters():
                    if name != self._final_mesh_name and name in shared_phases:
                        parameter.copy_(shared_phases[name])
            self._layers.append(second)

        self.n_readout = sum(layer.output_size for layer in self._layers)
        self.metadata = {
            "framework": f"MerLin {getattr(ml, '__version__', 'unknown')}",
            "computation_space": "UNBUNCHED",
            "detector_model": "threshold (unbunched subspace)",
            "n_photons": n_photons,
            "n_modes": n_modes,
            "input_state": self.input_state,
            "encoding": (
                f"angle encoding on modes 0..{n_input - 1}, "
                f"scale={encoding_scale:.6f} rad (pi/{math.pi / encoding_scale:.3g}), "
                f"{n_lags} sequential lag blocks"
            ),
            "encoding_scale_radians": float(encoding_scale),
            "measurement_strategy": (
                "MeasurementStrategy.mode_expectations"
                if readout == "mode_expectations"
                else "MeasurementStrategy.probs"
            ),
            "postselection": "none",
            "simulator": "MerLin CPU simulator (analytic statevector, shots=0)",
            "shots": None,
            "n_frozen_circuit_parameters": int(
                sum(p.numel() for layer in self._layers for p in layer.parameters())
            ),
            "n_trainable_parameters": self.n_readout,
            "readout_width": self.n_readout,
            "mesh_seed": seed,
            "ensemble": ensemble,
        }
        logger.info(
            "PHOTONIC_RESERVOIR_BUILT | modes=%d | photons=%d | input_modes=%d | "
            "readout=%s | width=%d | frozen_params=%d | input_state=%s",
            n_modes, n_photons, n_input, readout, self.n_readout,
            self.metadata["n_frozen_circuit_parameters"], self.input_state,
        )

    def evaluate(self, windows: np.ndarray, batch: int = 512) -> np.ndarray:
        """Readout vector for a batch of lagged feature windows.

        Parameters
        ----------
        windows : numpy.ndarray, shape (batch, n_lags, n_input)
            ``windows[b, 0]`` is the oldest lag.
        batch : int
            Chunk size for the photonic forward pass. Default value is 512.

        Returns
        -------
        numpy.ndarray, shape (batch, n_readout)
        """
        flat = torch.tensor(
            windows.reshape(len(windows), -1), dtype=torch.float32
        )
        pieces = []
        with torch.no_grad():
            for start in range(0, len(flat), batch):
                chunk = flat[start:start + batch]
                pieces.append(
                    torch.cat([layer(chunk) for layer in self._layers], dim=-1)
                )
        return torch.cat(pieces, dim=0).double().numpy()


def photonic_readout_matrix(
    windows: np.ndarray, n_lags: int, **kwargs
) -> tuple[np.ndarray, dict]:
    """Readout matrix for a whole time series, zero-padding the first ``n_lags`` rows.

    Returns
    -------
    tuple
        ``(readout, metadata)`` where ``readout`` has shape ``(T, n_readout)``.
    """
    reservoir = PhotonicReservoir(windows.shape[2], n_lags, **kwargs)
    matrix = np.zeros((len(windows), reservoir.n_readout))
    started = time.perf_counter()
    matrix[n_lags:] = reservoir.evaluate(windows[n_lags:])
    metadata = dict(reservoir.metadata)
    metadata["wall_clock_seconds"] = round(time.perf_counter() - started, 3)
    return matrix, metadata


def photonic_closed_loop_forecast(
    windows: np.ndarray,
    target: np.ndarray,
    readout: np.ndarray,
    train_slices,
    origin_index,
    *,
    rv_column: int,
    horizon: int,
    delta: float,
    reservoir: PhotonicReservoir,
) -> np.ndarray:
    """Closed-loop multi-step forecasts for a photonic reservoir.

    Mirrors :func:`lib.qrc.closed_loop_forecast`: the readout is trained on
    ground-truth open-loop features, then rolled forward ``horizon`` times with
    the model's own realized-volatility prediction replacing the newest ``RV``
    input while exogenous features stay at their ground-truth values.
    """
    from .qrc import ridge_weight_paths

    weights = ridge_weight_paths(readout, target, train_slices, delta)
    origins = np.asarray(origin_index)
    current = windows[origins].copy()
    predictions = np.zeros(len(origins))
    for step in range(horizon):
        if step > 0:
            rows = np.clip(origins + step, 0, len(windows) - 1)
            current = np.roll(current, -1, axis=1)
            current[:, -1, :] = windows[rows][:, -1, :]
            current[:, -1, rv_column] = predictions
        predictions = np.einsum("pf,pf->p", weights, reservoir.evaluate(current))
    return predictions


def run_photonic(cfg, run_dir):
    """Photonic experiment: sweep reservoir instances for each photonic variant.

    Writes one candidate run directory per (variant, instance) plus the
    coordinator's ``sweep.log``/``sweep_status.json``/``sweep_summary.json``.
    """
    import json
    import uuid
    from pathlib import Path

    from . import metrics as metric_lib
    from .data import build_lagged_inputs, denormalise_log_rv
    from .experiment_logging import configure_logging, git_state, utc_now, write_json
    from .qrc import rolling_ridge_forecast
    from .runner import (
        SCHEMA_VERSION,
        REPO_ROOT,
        Sample,
        _forecast_metrics,
        _write_candidate_end,
        _write_candidate_start,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "sweep.log")
    sweep_id = f"photonic-{uuid.uuid4().hex[:8]}"
    settings = cfg["photonic"]
    variants = settings["variants"]
    n_instances = settings["n_instances"]
    expected = len(variants) * n_instances * len(settings["encoding_scale_divisors"])

    sweep_status = {
        "schema_version": SCHEMA_VERSION, "sweep_id": sweep_id, "status": "RUNNING",
        "started_at": utc_now(), "completed_at": None, "code": git_state(REPO_ROOT),
        "expected_runs": expected, "runs": [],
        "summary_path": str((run_dir / "sweep_summary.json").resolve()),
        "selected_candidates": None, "error": None,
    }
    write_json(run_dir / "sweep_status.json", sweep_status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | selection_metric=mse | "
        "selection_split=%s | direction=minimize",
        sweep_id, expected, cfg["evaluation"]["selection_split"],
    )

    sample = Sample(cfg)
    features = list(cfg["feature_selection"]["paper_optimal_qr1"])
    windows = build_lagged_inputs(sample.normalised, features, sample.n_lags)
    rv_column = features.index("RV")
    delta = cfg["quantum_reservoir"]["ridge_delta"]
    divisors = list(settings["encoding_scale_divisors"])

    records, hardware_by_candidate = [], {}
    for variant_name, variant_cfg in variants.items():
        for divisor in divisors:
            for instance in range(n_instances):
                candidate = {"variant": variant_name, "instance": instance,
                             "encoding_scale_divisor": divisor}
                name = f"{variant_name}-s{divisor:02d}-i{instance:03d}"
                candidate_dir = run_dir / "candidates" / name
                candidate_status = _write_candidate_start(
                    candidate_dir, cfg, name, sweep_id, candidate, seed=instance,
                    dataset_note=f" | encoding_scale=pi/{divisor}")
                logger.info("CANDIDATE_STARTED | run_id=%s | candidate=%s",
                            name, json.dumps(candidate))
                reservoir = PhotonicReservoir(
                    len(features), sample.n_lags, seed=instance,
                    n_modes=variant_cfg["n_modes"], n_photons=variant_cfg["n_photons"],
                    readout=variant_cfg["readout"], ensemble=variant_cfg["ensemble"],
                    encoding_scale=math.pi / divisor,
                )
                matrix = np.zeros((sample.n_total, reservoir.n_readout))
                started = time.perf_counter()
                matrix[sample.n_lags:] = reservoir.evaluate(windows[sample.n_lags:])
                elapsed = time.perf_counter() - started

                forecasts = {
                    1: rolling_ridge_forecast(
                        matrix, sample.target, sample.train_slices,
                        sample.origin_index, delta=delta),
                }
                for horizon in sample.horizons:
                    if horizon == 1:
                        continue
                    forecasts[horizon] = photonic_closed_loop_forecast(
                        windows, sample.target, matrix, sample.train_slices,
                        sample.origin_index, rv_column=rv_column, horizon=horizon,
                        delta=delta, reservoir=reservoir)
                validation = rolling_ridge_forecast(
                    matrix, sample.target, sample.validation_slices,
                    sample.validation_index, delta=delta)

                per_horizon = {}
                for horizon in sample.horizons:
                    _, scores = _forecast_metrics(sample, forecasts[horizon], horizon)
                    per_horizon[str(horizon)] = scores
                validation_scores = metric_lib.summarise(
                    denormalise_log_rv(validation), sample.validation_actual)
                hardware = dict(reservoir.metadata)
                hardware["wall_clock_seconds"] = round(elapsed, 3)
                hardware_by_candidate[(variant_name, divisor, instance)] = hardware

                _write_candidate_end(
                    candidate_dir, candidate_status,
                    {"candidate": candidate, "horizons": per_horizon,
                     "validation": validation_scores, "hardware": hardware},
                    f"EVALUATION_COMPLETED | test_mse={per_horizon['1']['mse']:.6f} | "
                    f"validation_mse={validation_scores['mse']:.6f}",
                )
                np.save(candidate_dir / "forecast_S1.npy", forecasts[1])
                row = {"variant": variant_name, "instance": instance,
                       "encoding_scale_divisor": divisor, "status": "DONE",
                       "run_id": name,
                       "metrics_path": str((candidate_dir / "metrics.json").resolve()),
                       "test_mse": per_horizon["1"]["mse"],
                       "test_qlike": per_horizon["1"]["qlike"],
                       "validation_mse": validation_scores["mse"],
                       "wall_clock_seconds": round(elapsed, 3)}
                for horizon in sample.horizons:
                    row[f"test_mse_S{horizon}"] = per_horizon[str(horizon)]["mse"]
                    row[f"test_qlike_S{horizon}"] = per_horizon[str(horizon)]["qlike"]
                records.append(row)
                sweep_status["runs"].append({
                    "run_id": name, "candidate": candidate, "seed": instance,
                    "repetition": 0, "status": "COMPLETED",
                    "run_dir": str(candidate_dir.resolve()),
                    "run_log": str((candidate_dir / "run.log").resolve()),
                    "run_status": str((candidate_dir / "run_status.json").resolve()),
                    "config_path": str((candidate_dir / "config_snapshot.json").resolve()),
                    "metrics_path": str((candidate_dir / "metrics.json").resolve())})
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s | test_mse=%.6f | "
                    "validation_mse=%.6f | wall_clock_s=%.2f",
                    name, candidate_dir / "metrics.json", row["test_mse"],
                    row["validation_mse"], elapsed,
                )
                write_json(run_dir / "sweep_status.json", sweep_status)

    import pandas as pd

    frame = pd.DataFrame(records)
    frame.to_csv(run_dir / "photonic_sweep.csv", index=False)
    summary = {"experiment": "photonic",
               "photonic_status": "PARTIAL_MERLIN_TRANSLATION",
               "selection_rule": {
                   "metric": "mse", "direction": "minimize",
                   "split": cfg["evaluation"]["selection_split"],
                   "candidates": "(encoding_scale_divisor, reservoir instance) jointly",
               },
               "hardware": {}, "per_variant": {},
               "per_variant_scale": {}, "selected": {}}
    for variant_name, group in frame.groupby("variant"):
        best_test = group.loc[group["test_mse"].idxmin()]
        best_validation = group.loc[group["validation_mse"].idxmin()]
        entry = {
            "n_completed": int(len(group)),
            "n_expected": int(n_instances) * len(divisors),
            "test_mse": {
                "mean": float(group["test_mse"].mean()),
                "std": float(group["test_mse"].std(ddof=1)),
                "median": float(group["test_mse"].median()),
                "min": float(group["test_mse"].min()),
                "max": float(group["test_mse"].max()),
            },
            "best_on_test": {
                "instance": int(best_test["instance"]),
                "encoding_scale_divisor": int(best_test["encoding_scale_divisor"]),
                "test_mse": float(best_test["test_mse"]),
                "test_qlike": float(best_test["test_qlike"]),
            },
            "selected_on_validation": {
                "instance": int(best_validation["instance"]),
                "encoding_scale_divisor": int(best_validation["encoding_scale_divisor"]),
                "validation_mse": float(best_validation["validation_mse"]),
                "test_mse": float(best_validation["test_mse"]),
                "test_qlike": float(best_validation["test_qlike"]),
            },
            "mean_wall_clock_seconds": float(group["wall_clock_seconds"].mean()),
        }
        for horizon in sample.horizons:
            column = f"test_mse_S{horizon}"
            entry[f"test_mse_S{horizon}"] = {
                "mean": float(group[column].mean()),
                "min": float(group[column].min()),
                "at_validation_selected": float(best_validation[column]),
            }
        summary["per_variant"][variant_name] = entry
        summary["per_variant_scale"][variant_name] = {
            f"pi/{int(divisor)}": {
                "n": int(len(block)),
                "test_mse_mean": float(block["test_mse"].mean()),
                "test_mse_std": float(block["test_mse"].std(ddof=1)),
                "test_mse_min": float(block["test_mse"].min()),
                "validation_mse_min": float(block["validation_mse"].min()),
            }
            for divisor, block in group.groupby("encoding_scale_divisor")
        }
        summary["selected"][variant_name] = {
            "instance": int(best_validation["instance"]),
            "encoding_scale_divisor": int(best_validation["encoding_scale_divisor"]),
        }
        # Hardware-aware report describes the validation-selected candidate.
        summary["hardware"][variant_name] = hardware_by_candidate[(
            variant_name,
            int(best_validation["encoding_scale_divisor"]),
            int(best_validation["instance"]),
        )]
        summary["hardware"][variant_name]["sweep_n_instances"] = int(n_instances)
        summary["hardware"][variant_name]["sweep_encoding_scale_divisors"] = divisors

    write_json(run_dir / "sweep_summary.json", summary)
    completed = len(records)
    sweep_status["selected_candidates"] = summary["selected"]
    sweep_status["status"] = "COMPLETED" if completed == expected else "PARTIAL"
    sweep_status["completed_at"] = utc_now()
    write_json(run_dir / "sweep_status.json", sweep_status)
    logger.info("%s | summary=%s | selected=%s | completed=%d/%d",
                "SWEEP_COMPLETED" if completed == expected else "SWEEP_PARTIAL",
                run_dir / "sweep_summary.json", json.dumps(summary["selected"]),
                completed, expected)
    return summary
