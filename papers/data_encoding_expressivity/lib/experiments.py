"""The four experiment kinds behind ``cfg["experiment"]``.

- ``fit``: Figs. 3-4 (E1, E2). Train quantum models or baseline B2 on target
  Fourier series; compare every run with the B1 least-squares oracle.
- ``coefficients``: Fig. 5 (E3). Fourier coefficients of random parallel
  models for several ansaetze.
- ``spectrum``: Sections II-III and App. A-B (E4). Exact checks that the FFT
  support of random models equals the predicted spectrum.
- ``universality``: Section IV and App. C (E5). The proof's construction,
  checked for exact reconstruction and for convergence on a square wave.

Each function logs ``DATASET_READY`` once and at least one
``EVALUATION_COMPLETED``, writes its artifacts into ``run_dir`` and returns a
JSON-serialisable metrics dict whose artifact paths are relative to ``run_dir``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import least_squares_oracle
from .circuits import ParallelPauliModel, SerialPauliModel
from .data import build_dataset
from .fourier import (
    distinct_sum_size_bound,
    fourier_coefficients,
    largest_coefficient_outside,
    paper_size_bound,
    periodic_grid,
    predicted_spectrum,
    reference_coefficients,
    spectrum_size,
    spectrum_support,
)
from .plotting import plot_coefficient_clouds, plot_fit_cases, plot_universality
from .spectrum import (
    generator_with_spectrum,
    haar_unitary,
    layered_model_values,
    multivariate_model_values,
    pauli_x_sum_generator,
    random_hermitian,
)
from .training import (
    MODEL_KINDS,
    build_fit_model,
    count_parameters,
    model_frequencies,
    train_fit_model,
)
from .universality import (
    square_wave_coefficients,
    square_wave_tail_rms,
    truncated_series_values,
    universal_model_values,
    universal_observable,
)

LOGGER = logging.getLogger(__name__)

TWO_PI = 2 * np.pi


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    """Return the mean and the sample standard deviation (0 for one value)."""
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std


def _target_statistics(targets: np.ndarray) -> dict:
    return {
        "min": float(targets.min()),
        "max": float(targets.max()),
        "mean": float(targets.mean()),
        "mean_square": float(np.mean(targets**2)),
    }


def run_fit_experiment(
    fit_cfg: Mapping[str, Any], run_dir: Path, device: torch.device, dtype: torch.dtype
) -> dict:
    """Run every case x model kind x seed of a fitting experiment (E1, E2).

    Parameters
    ----------
    fit_cfg : Mapping
        Keys ``grid``, ``training``, ``seeds``, ``models``,
        ``success_tolerance`` and ``cases`` (each with ``name``, ``target`` and
        ``model``).
    run_dir : Path
        Run directory for artifacts.
    device : torch.device
        Device for training.
    dtype : torch.dtype
        Real dtype of models and data.

    Returns
    -------
    dict
        Per-case B1 references, per-run results, per-case summaries and
        artifact paths.

    Raises
    ------
    ValueError
        If a model kind is unknown or the case names are not unique.
    """
    model_kinds = list(fit_cfg["models"])
    unknown_kinds = set(model_kinds) - set(MODEL_KINDS)
    if unknown_kinds:
        raise ValueError(
            f"Unknown model kinds {sorted(unknown_kinds)}; expected {MODEL_KINDS}"
        )
    seeds = [int(seed) for seed in fit_cfg["seeds"]]
    tolerance = float(fit_cfg["success_tolerance"])
    training_cfg = fit_cfg["training"]
    case_names = [case["name"] for case in fit_cfg["cases"]]
    if len(set(case_names)) != len(case_names):
        raise ValueError(f"Case names must be unique: {case_names}")

    prepared = []
    for case in fit_cfg["cases"]:
        case_cfg = {
            "grid": fit_cfg["grid"],
            "target": case["target"],
            "training": training_cfg,
            "model": case["model"],
        }
        inputs, targets = build_dataset(case_cfg, dtype)
        frequencies, scaling = model_frequencies(case["model"])
        oracle = least_squares_oracle(
            inputs.numpy(), targets.numpy(), frequencies, scaling
        )
        prepared.append(
            {
                "name": case["name"],
                "cfg": case_cfg,
                "inputs": inputs.numpy(),
                "targets": targets.numpy(),
                "frequencies": frequencies,
                "scaling": scaling,
                "b1_mse": oracle["mse"],
                "b1_predictions": oracle["predictions"],
            }
        )
        LOGGER.info(
            "TARGET_BUILT | case=%s | points=%d | b1_frequencies=%s | b1_scaling=%g | b1_mse=%.6e",
            case["name"],
            len(inputs),
            frequencies,
            scaling,
            oracle["mse"],
        )
    LOGGER.info(
        "DATASET_READY | name=synthetic_fourier_targets | cases=%d | grid=linspace(%s, %s, %s)",
        len(prepared),
        fit_cfg["grid"]["start"],
        fit_cfg["grid"]["stop"],
        fit_cfg["grid"]["num"],
    )

    LOGGER.info(
        "TRAINING_STARTED | cases=%d | models=%s | seeds=%s | optimizer=adam | lr=%s | "
        "batch_size=%s | steps=%s",
        len(prepared),
        model_kinds,
        seeds,
        training_cfg["lr"],
        training_cfg["batch_size"],
        training_cfg["steps"],
    )
    runs = []
    predictions: dict[str, np.ndarray] = {}
    for case in prepared:
        for kind in model_kinds:
            for seed in seeds:
                model = build_fit_model(kind, case["cfg"]["model"], dtype)
                label = f"case={case['name']} | model={kind} | seed={seed}"
                n_parameters = count_parameters(model)
                LOGGER.info("FIT_STARTED | %s | parameters=%d", label, n_parameters)
                outcome = train_fit_model(model, case["cfg"], seed, device, label)
                LOGGER.info(
                    "FIT_COMPLETED | %s | final_mse=%.6e", label, outcome["final_mse"]
                )
                LOGGER.info("EVALUATION_STARTED | %s | split=full_grid", label)
                gap = outcome["final_mse"] - case["b1_mse"]
                success = gap <= tolerance
                LOGGER.info(
                    "EVALUATION_COMPLETED | %s | split=full_grid | mse=%.6e | b1_mse=%.6e | "
                    "gap_to_b1=%.3e | success=%s",
                    label,
                    outcome["final_mse"],
                    case["b1_mse"],
                    gap,
                    success,
                )
                predictions[f"{case['name']}/{kind}/{seed}"] = outcome[
                    "predictions"
                ].numpy()
                runs.append(
                    {
                        "case": case["name"],
                        "model_kind": kind,
                        "seed": seed,
                        "n_parameters": n_parameters,
                        "final_mse": outcome["final_mse"],
                        "final_half_mse": outcome["final_half_mse"],
                        "b1_mse": case["b1_mse"],
                        "gap_to_b1": gap,
                        "success": success,
                        "mse_history": outcome["full_grid_mse"],
                    }
                )

    summary = []
    for case in prepared:
        for kind in model_kinds:
            case_runs = [
                r for r in runs if r["case"] == case["name"] and r["model_kind"] == kind
            ]
            mean_mse, std_mse = _mean_and_std([r["final_mse"] for r in case_runs])
            summary.append(
                {
                    "case": case["name"],
                    "model_kind": kind,
                    "n_seeds": len(case_runs),
                    "n_parameters": case_runs[0]["n_parameters"],
                    "final_mse_mean": mean_mse,
                    "final_mse_std": std_mse,
                    "final_mse_min": float(min(r["final_mse"] for r in case_runs)),
                    "final_mse_max": float(max(r["final_mse"] for r in case_runs)),
                    "b1_mse": case["b1_mse"],
                    "success_fraction": sum(r["success"] for r in case_runs)
                    / len(case_runs),
                }
            )

    predictions_path = run_dir / "fit_predictions.npz"
    np.savez(
        predictions_path,
        inputs=prepared[0]["inputs"],
        **{f"target/{case['name']}": case["targets"] for case in prepared},
        **{f"b1/{case['name']}": case["b1_predictions"] for case in prepared},
        **{f"model/{key}": value for key, value in predictions.items()},
    )
    LOGGER.info("ARTIFACT_WRITTEN | path=%s", predictions_path.name)
    figure_path = run_dir / "fit_cases.png"
    plot_fit_cases(prepared[0]["inputs"], prepared, runs, predictions, figure_path)
    LOGGER.info("ARTIFACT_WRITTEN | path=%s", figure_path.name)

    return {
        "experiment": "fit",
        "success_tolerance": tolerance,
        "success_rule": "final_mse - b1_mse <= success_tolerance (plain MSE on the full grid)",
        "seeds": seeds,
        "model_kinds": model_kinds,
        "training": dict(training_cfg),
        "cases": [
            {
                "name": case["name"],
                "target": case["cfg"]["target"],
                "model": case["cfg"]["model"],
                "b1_frequencies": case["frequencies"],
                "b1_scaling": case["scaling"],
                "b1_mse": case["b1_mse"],
                "target_statistics": _target_statistics(case["targets"]),
            }
            for case in prepared
        ],
        "runs": runs,
        "summary": summary,
        "artifacts": {"predictions": predictions_path.name, "figure": figure_path.name},
    }


def _coefficient_model(
    setting: Mapping[str, Any], coefficient_cfg: Mapping[str, Any], dtype: torch.dtype
) -> tuple[str, torch.nn.Module, int]:
    """Build one E3 model: the paper's parallel-RX model, or a photonic one (Phase 4).

    A setting with a ``photonic`` section builds ``lib.photonic_merlin``'s
    model and needs a ``label``. Otherwise the setting gives ``ansatz`` and
    ``ansatz_layers`` for the parallel ``RX`` model on ``n_qubits`` wires.

    Returns
    -------
    tuple
        Label, model and predicted degree (largest frequency of the spectrum).
    """
    if "photonic" in setting:
        from .photonic_merlin import build_photonic_model, photonic_positive_frequencies

        photonic_cfg = setting["photonic"]
        return (
            str(setting["label"]),
            build_photonic_model(photonic_cfg, dtype),
            max(photonic_positive_frequencies(photonic_cfg)),
        )
    n_qubits = int(coefficient_cfg["n_qubits"])
    ansatz = setting["ansatz"]
    layers = int(setting["ansatz_layers"])
    # Fig. 5 samples the paper's fixed <Z_0> observable.
    model = ParallelPauliModel(
        n_qubits,
        float(coefficient_cfg["encoding_scaling"]),
        ansatz,
        layers,
        "z0",
        dtype,
    )
    return f"{ansatz}_l{layers}", model, n_qubits


def run_coefficient_experiment(
    coefficient_cfg: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    """Sample Fourier coefficients of random parallel models (E3, Fig. 5).

    Coefficients are extracted as in the reference notebook: ``rfft`` of
    ``2K + 1`` samples on ``[0, 2 pi)`` divided by ``2K + 1``.

    Parameters
    ----------
    coefficient_cfg : Mapping
        Keys ``n_samples``, ``max_frequency`` (K), ``init_high``,
        ``zero_tolerance`` and ``settings``. Qubit settings give ``ansatz`` and
        ``ansatz_layers`` and need top-level ``n_qubits`` and
        ``encoding_scaling``; photonic settings give ``label`` and a
        ``photonic`` model section (see ``_coefficient_model``).
    run_dir : Path
        Run directory for artifacts.
    device : torch.device
        Device for evaluation.
    dtype : torch.dtype
        Real dtype.
    seed : int
        Seed of the parameter draws; each setting reuses it.

    Returns
    -------
    dict
        Per-setting, per-frequency statistics and artifact paths.
    """
    n_samples = int(coefficient_cfg["n_samples"])
    max_frequency = int(coefficient_cfg["max_frequency"])
    init_high = float(coefficient_cfg["init_high"])
    zero_tolerance = float(coefficient_cfg["zero_tolerance"])
    grid = torch.tensor(
        periodic_grid(2 * max_frequency + 1), dtype=dtype, device=device
    )
    LOGGER.info(
        "DATASET_READY | name=none (random circuits) | settings=%d | samples_per_setting=%d | "
        "grid_points=%d",
        len(coefficient_cfg["settings"]),
        n_samples,
        len(grid),
    )

    samples: dict[str, np.ndarray] = {}
    settings_metrics = []
    for setting in coefficient_cfg["settings"]:
        label, model, predicted_degree = _coefficient_model(
            setting, coefficient_cfg, dtype
        )
        model = model.to(device)
        LOGGER.info(
            "EVALUATION_STARTED | setting=%s | predicted_degree=%d",
            label,
            predicted_degree,
        )
        generator = torch.Generator().manual_seed(seed)
        draws = []
        for _ in range(n_samples):
            model.reset_parameters(generator, init_high)
            model.to(device)
            with torch.no_grad():
                values = model(grid).cpu().numpy()
            draws.append(reference_coefficients(values))
        coefficients = np.stack(draws)
        samples[label] = coefficients
        per_frequency = []
        for frequency in range(max_frequency + 1):
            column = coefficients[:, frequency]
            per_frequency.append(
                {
                    "frequency": frequency,
                    "mean_abs": float(np.abs(column).mean()),
                    "max_abs": float(np.abs(column).max()),
                    "var_real": float(column.real.var()),
                    "var_imag": float(column.imag.var()),
                    "var_total": float(column.real.var() + column.imag.var()),
                    "fraction_zero": float(np.mean(np.abs(column) <= zero_tolerance)),
                }
            )
        settings_metrics.append(
            {
                "label": label,
                "setting": dict(setting),
                "predicted_degree": predicted_degree,
                "n_parameters": count_parameters(model),
                "per_frequency": per_frequency,
            }
        )
        LOGGER.info(
            "EVALUATION_COMPLETED | setting=%s | var_total=%s | fraction_zero=%s",
            label,
            [round(entry["var_total"], 6) for entry in per_frequency],
            [entry["fraction_zero"] for entry in per_frequency],
        )

    samples_path = run_dir / "coefficients.npz"
    np.savez(samples_path, **samples)
    LOGGER.info("ARTIFACT_WRITTEN | path=%s", samples_path.name)
    figure_path = run_dir / "coefficient_clouds.png"
    plot_coefficient_clouds(samples, figure_path)
    LOGGER.info("ARTIFACT_WRITTEN | path=%s", figure_path.name)
    return {
        "experiment": "coefficients",
        "n_samples": n_samples,
        "max_frequency": max_frequency,
        "zero_tolerance": zero_tolerance,
        "seed": seed,
        "settings": settings_metrics,
        "artifacts": {"coefficients": samples_path.name, "figure": figure_path.name},
    }


def _support_check(
    name: str,
    predicted: np.ndarray,
    value_draws: list[np.ndarray],
    tolerance: float,
    extra: Mapping[str, Any],
) -> dict:
    """Compare the union of FFT supports over random draws with a predicted spectrum.

    ``subset`` checks that no draw has a coefficient outside the prediction
    (the theorem). ``equal`` checks that every predicted frequency is reached
    by some draw (the prediction is tight for generic ``W``).
    """
    predicted_set = sorted(int(round(v)) for v in predicted)
    observed: set[int] = set()
    largest_outside = 0.0
    for values in value_draws:
        frequencies, coefficients = fourier_coefficients(values)
        observed.update(spectrum_support(frequencies, coefficients, tolerance))
        largest_outside = max(
            largest_outside,
            largest_coefficient_outside(frequencies, coefficients, predicted_set),
        )
    subset = observed.issubset(predicted_set)
    equal = observed == set(predicted_set)
    record = {
        "name": name,
        "predicted": predicted_set,
        "observed_union": sorted(observed),
        "largest_coefficient_outside": largest_outside,
        "subset": subset,
        "equal": equal,
        "passed": subset and equal,
        **extra,
    }
    LOGGER.info(
        "EVALUATION_COMPLETED | check=%s | predicted=%s | observed=%s | max_outside=%.2e | passed=%s",
        name,
        predicted_set,
        sorted(observed),
        largest_outside,
        record["passed"],
    )
    return record


def _grid_points_for(max_frequency: float) -> int:
    """Return enough periodic samples to resolve frequencies up to ``max_frequency`` and beyond."""
    return int(4 * np.ceil(max_frequency) + 8)


def run_spectrum_experiment(
    spectrum_cfg: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    """Run the exact spectrum checks (E4).

    Checks, all on periodic grids with enough points to see frequencies above
    the prediction:

    1. Pauli ``RX`` repeated ``r`` times in series and in parallel (the torch
       models of Figs. 3-5): support is ``{-r..r}`` and both layouts agree
       (C1, C4a, C6 with ``K = L``).
    2. Dense models with arbitrary generator spectra, Haar ``W`` and random
       ``M``, with diagonal and rotated generators (C1, C6).
    3. Two-feature model of App. A: 2D support is the Cartesian product (C10).
    4. Commensurable non-integer eigenvalues of App. B: integer support in
       units of the base frequency (C11).

    Parameters
    ----------
    spectrum_cfg : Mapping
        Keys ``fft_tolerance``, ``random_draws``, ``init_high``,
        ``pauli_repetitions``, ``parallel_ansatz``, ``parallel_ansatz_layers``,
        ``dense_cases``, ``multivariate`` and ``non_integer``.
    run_dir : Path
        Run directory (no artifacts beyond metrics).
    device : torch.device
        Device for the torch models.
    dtype : torch.dtype
        Real dtype of the torch models.
    seed : int
        Seed of all random draws.

    Returns
    -------
    dict
        Every check record, a bound table and ``all_passed``.
    """
    tolerance = float(spectrum_cfg["fft_tolerance"])
    draws = int(spectrum_cfg["random_draws"])
    init_high = float(spectrum_cfg["init_high"])
    rng = np.random.default_rng(seed)
    torch_generator = torch.Generator().manual_seed(seed)
    LOGGER.info(
        "DATASET_READY | name=none (random circuits and analytic spectra) | draws_per_check=%d | "
        "fft_tolerance=%.1e",
        draws,
        tolerance,
    )

    checks = []
    for repetitions in spectrum_cfg["pauli_repetitions"]:
        repetitions = int(repetitions)
        serial_spectrum = predicted_spectrum([-0.5, 0.5], repetitions)
        parallel_eigenvalues = np.linalg.eigvalsh(pauli_x_sum_generator(repetitions))
        parallel_spectrum = predicted_spectrum(parallel_eigenvalues, 1)
        grid = torch.tensor(
            periodic_grid(_grid_points_for(repetitions)), dtype=dtype, device=device
        )
        models = {
            # The paper's fixed <Z_0>; the spectrum does not depend on the observable.
            "serial": SerialPauliModel(repetitions, 1.0, "z0", dtype),
            "parallel": ParallelPauliModel(
                repetitions,
                1.0,
                spectrum_cfg["parallel_ansatz"],
                int(spectrum_cfg["parallel_ansatz_layers"]),
                "z0",
                dtype,
            ),
        }
        for layout, model in models.items():
            value_draws = []
            for _ in range(draws):
                model.reset_parameters(torch_generator, init_high)
                model.to(device)
                with torch.no_grad():
                    value_draws.append(model(grid).cpu().numpy())
            predicted = serial_spectrum if layout == "serial" else parallel_spectrum
            checks.append(
                _support_check(
                    f"pauli_{layout}_r{repetitions}",
                    predicted,
                    value_draws,
                    tolerance,
                    {
                        "kind": "pauli",
                        "layout": layout,
                        "repetitions": repetitions,
                        "spectrum_size": spectrum_size(predicted),
                        "size_equals_repetitions": spectrum_size(predicted)
                        == repetitions,
                        "serial_equals_parallel_prediction": bool(
                            np.array_equal(serial_spectrum, parallel_spectrum)
                        ),
                    },
                )
            )

    bound_table = []
    for case in spectrum_cfg["dense_cases"]:
        eigenvalues = [float(v) for v in case["eigenvalues"]]
        layers = int(case["layers"])
        rotate = bool(case["rotate_generator"])
        generator = generator_with_spectrum(eigenvalues, rotate, rng)
        predicted = predicted_spectrum(eigenvalues, layers)
        if not np.allclose(predicted, np.round(predicted)):
            raise ValueError(
                f"dense case {case['name']} needs integer frequencies, got {predicted}"
            )
        dimension = len(eigenvalues)
        grid = periodic_grid(_grid_points_for(np.abs(predicted).max()))
        value_draws = []
        coefficient_draws = []
        for _ in range(draws):
            unitaries = [haar_unitary(dimension, rng) for _ in range(layers + 1)]
            observable = random_hermitian(dimension, rng)
            values = layered_model_values(grid, generator, unitaries, observable)
            value_draws.append(values)
            coefficient_draws.append(fourier_coefficients(values)[1])
        size = spectrum_size(predicted)
        paper_bound = paper_size_bound(dimension, layers)
        tighter_bound = distinct_sum_size_bound(dimension, layers)
        checks.append(
            _support_check(
                f"dense_{case['name']}",
                predicted,
                value_draws,
                tolerance,
                {
                    "kind": "dense",
                    "eigenvalues": eigenvalues,
                    "layers": layers,
                    "rotated_generator": rotate,
                    # Different W and M change the coefficients but not the support.
                    "coefficient_spread_across_draws": float(
                        np.abs(np.stack(coefficient_draws)).std(axis=0).max()
                    ),
                },
            )
        )
        bound_table.append(
            {
                "case": case["name"],
                "dimension": dimension,
                "layers": layers,
                "spectrum_size": size,
                "paper_bound": paper_bound,
                "distinct_sum_bound": tighter_bound,
                "within_paper_bound": size <= paper_bound,
                "within_distinct_sum_bound": size <= tighter_bound,
            }
        )

    multivariate_cfg = spectrum_cfg["multivariate"]
    first_reps, second_reps = (
        int(v) for v in multivariate_cfg["repetitions_per_feature"]
    )
    first_generator = pauli_x_sum_generator(first_reps)
    second_generator = pauli_x_sum_generator(second_reps)
    joint_dimension = 2 ** (first_reps + second_reps)
    first_grid = periodic_grid(_grid_points_for(first_reps))
    second_grid = periodic_grid(_grid_points_for(second_reps))
    predicted_pairs = {
        (a, b)
        for a in range(-first_reps, first_reps + 1)
        for b in range(-second_reps, second_reps + 1)
    }
    observed_pairs: set[tuple[int, int]] = set()
    largest_outside = 0.0
    for _ in range(draws):
        unitaries = [haar_unitary(joint_dimension, rng) for _ in range(2)]
        observable = random_hermitian(joint_dimension, rng)
        values = multivariate_model_values(
            first_grid,
            second_grid,
            first_generator,
            second_generator,
            unitaries,
            observable,
        )
        coefficients = np.fft.fft2(values) / values.size
        first_freqs = np.rint(
            np.fft.fftfreq(len(first_grid), d=1.0 / len(first_grid))
        ).astype(int)
        second_freqs = np.rint(
            np.fft.fftfreq(len(second_grid), d=1.0 / len(second_grid))
        ).astype(int)
        for i, a in enumerate(first_freqs):
            for j, b in enumerate(second_freqs):
                magnitude = abs(coefficients[i, j])
                if magnitude > tolerance:
                    observed_pairs.add((int(a), int(b)))
                if (int(a), int(b)) not in predicted_pairs:
                    largest_outside = max(largest_outside, magnitude)
    multivariate_record = {
        "name": f"multivariate_r{first_reps}x{second_reps}",
        "kind": "multivariate",
        "repetitions_per_feature": [first_reps, second_reps],
        "predicted_size": len(predicted_pairs),
        "observed_size": len(observed_pairs),
        "largest_coefficient_outside": largest_outside,
        "subset": observed_pairs.issubset(predicted_pairs),
        "equal": observed_pairs == predicted_pairs,
    }
    multivariate_record["passed"] = (
        multivariate_record["subset"] and multivariate_record["equal"]
    )
    LOGGER.info(
        "EVALUATION_COMPLETED | check=%s | predicted_size=%d | observed_size=%d | max_outside=%.2e | passed=%s",
        multivariate_record["name"],
        len(predicted_pairs),
        len(observed_pairs),
        largest_outside,
        multivariate_record["passed"],
    )
    checks.append(multivariate_record)

    non_integer_cfg = spectrum_cfg["non_integer"]
    base_frequency = float(non_integer_cfg["base_frequency"])
    multiples = [float(v) for v in non_integer_cfg["eigenvalue_multiples"]]
    layers = int(non_integer_cfg["layers"])
    eigenvalues = [base_frequency * multiple for multiple in multiples]
    predicted_in_base_units = predicted_spectrum(multiples, layers)
    generator = generator_with_spectrum(eigenvalues, True, rng)
    # Sampling one period 2 pi / w0 maps frequency n w0 to FFT index n (App. B).
    grid = periodic_grid(
        _grid_points_for(np.abs(predicted_in_base_units).max()),
        period=TWO_PI / base_frequency,
    )
    value_draws = []
    for _ in range(draws):
        unitaries = [haar_unitary(len(eigenvalues), rng) for _ in range(layers + 1)]
        observable = random_hermitian(len(eigenvalues), rng)
        value_draws.append(layered_model_values(grid, generator, unitaries, observable))
    checks.append(
        _support_check(
            f"non_integer_w0_{base_frequency:g}",
            predicted_in_base_units,
            value_draws,
            tolerance,
            {
                "kind": "non_integer",
                "base_frequency": base_frequency,
                "eigenvalues": eigenvalues,
                "layers": layers,
            },
        )
    )

    all_passed = all(check["passed"] for check in checks) and all(
        row["within_paper_bound"] and row["within_distinct_sum_bound"]
        for row in bound_table
    )
    return {
        "experiment": "spectrum",
        "fft_tolerance": tolerance,
        "random_draws": draws,
        "seed": seed,
        "checks": checks,
        "bound_table": bound_table,
        "all_passed": all_passed,
    }


def run_universality_experiment(
    universality_cfg: Mapping[str, Any], run_dir: Path, seed: int
) -> dict:
    """Check the App. C construction (E5).

    Part 1: random truncated series of degree ``K`` are reproduced exactly by
    ``m = K + extra`` subsystems. Part 2: for the square wave ``sign(sin x)``
    the constructed ``f_m`` (``K = m``) converges in RMS at exactly the
    Parseval truncation rate.

    Parameters
    ----------
    universality_cfg : Mapping
        Keys ``exact_degrees``, ``extra_qubits``, ``coefficient_bound``,
        ``evaluation_points`` and ``square_wave_qubits``.
    run_dir : Path
        Run directory for the figure.
    seed : int
        Seed of the random target coefficients.

    Returns
    -------
    dict
        Exact-reconstruction records, square-wave records and artifact paths.
    """
    rng = np.random.default_rng(seed)
    bound = float(universality_cfg["coefficient_bound"])
    n_points = int(universality_cfg["evaluation_points"])
    # A half-step offset keeps grid points off the square wave's jumps at 0 and pi.
    grid = periodic_grid(n_points) + np.pi / n_points
    LOGGER.info(
        "DATASET_READY | name=synthetic truncated series and sign(sin x) | evaluation_points=%d",
        n_points,
    )

    exact_records = []
    for degree in universality_cfg["exact_degrees"]:
        degree = int(degree)
        constant = float(rng.uniform(-bound, bound))
        coefficients = list(
            rng.uniform(-bound, bound, degree) + 1j * rng.uniform(-bound, bound, degree)
        )
        target = truncated_series_values(grid, constant, coefficients)
        for extra in universality_cfg["extra_qubits"]:
            n_qubits = degree + int(extra)
            observable = universal_observable(constant, coefficients, n_qubits)
            values = universal_model_values(grid, observable, n_qubits)
            record = {
                "degree": degree,
                "n_qubits": n_qubits,
                "max_abs_error": float(np.max(np.abs(values - target))),
                "observable_is_hermitian": bool(
                    np.allclose(observable, observable.conj().T)
                ),
            }
            exact_records.append(record)
            LOGGER.info(
                "EVALUATION_COMPLETED | part=exact | degree=%d | n_qubits=%d | max_abs_error=%.3e",
                degree,
                n_qubits,
                record["max_abs_error"],
            )

    square_target = np.sign(np.sin(grid))
    square_records = []
    for n_qubits in universality_cfg["square_wave_qubits"]:
        n_qubits = int(n_qubits)
        constant, coefficients = square_wave_coefficients(n_qubits)
        observable = universal_observable(constant, coefficients, n_qubits)
        values = universal_model_values(grid, observable, n_qubits)
        truncation = truncated_series_values(grid, constant, coefficients)
        record = {
            "n_qubits": n_qubits,
            "degree": n_qubits,
            "rms_to_target": float(np.sqrt(np.mean((values - square_target) ** 2))),
            "parseval_tail_rms": square_wave_tail_rms(n_qubits),
            "max_abs_error_to_truncation": float(np.max(np.abs(values - truncation))),
        }
        square_records.append(record)
        LOGGER.info(
            "EVALUATION_COMPLETED | part=square_wave | n_qubits=%d | rms_to_target=%.4e | "
            "parseval_tail_rms=%.4e | max_abs_error_to_truncation=%.3e",
            n_qubits,
            record["rms_to_target"],
            record["parseval_tail_rms"],
            record["max_abs_error_to_truncation"],
        )

    figure_path = run_dir / "universality_square_wave.png"
    plot_universality(square_records, figure_path)
    LOGGER.info("ARTIFACT_WRITTEN | path=%s", figure_path.name)
    return {
        "experiment": "universality",
        "generator_family": "H_m = (1/2) sum_i Z_i (Pauli-rotation family, Section III.B)",
        "seed": seed,
        "evaluation_points": n_points,
        "exact": exact_records,
        "square_wave": square_records,
        "artifacts": {"figure": figure_path.name},
    }


def run_photonic_spectrum_experiment(
    spectrum_cfg: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    """Check the photonic (MerLin) spectrum against the photon-number prediction (Phase 4, E6).

    For every case, random mesh phases are drawn and the FFT support of every
    outcome probability and of the threshold readout is collected. The
    prediction is ``{Lambda_k - Lambda_j}`` built from the eigenvalues of
    ``s sum_{k in E} n_k`` on the ``n``-photon Fock space (paper Eq. 9).

    FOCK cases are expected to be band-limited (support equals the
    prediction). UNBUNCHED cases postselect and renormalise, so they are
    expected *not* to be band-limited. ``passed`` records whether the observed
    behaviour matches the case's ``expect_band_limited`` flag.

    Parameters
    ----------
    spectrum_cfg : Mapping
        Keys ``fft_tolerance``, ``random_draws``, ``init_high``,
        ``grid_points`` and ``cases`` (each with ``name``,
        ``expect_band_limited`` and the photonic model keys).
    run_dir : Path
        Run directory (no artifacts beyond metrics).
    device : torch.device
        Device for the MerLin models.
    dtype : torch.dtype
        Real dtype of the simulation.
    seed : int
        Seed of the random mesh phases.

    Returns
    -------
    dict
        Every check record and ``all_passed``.
    """
    from .photonic_merlin import build_photonic_model, photonic_predicted_spectrum

    tolerance = float(spectrum_cfg["fft_tolerance"])
    draws = int(spectrum_cfg["random_draws"])
    init_high = float(spectrum_cfg["init_high"])
    n_points = int(spectrum_cfg["grid_points"])
    generator = torch.Generator().manual_seed(seed)
    LOGGER.info(
        "DATASET_READY | name=none (random photonic circuits) | cases=%d | draws_per_case=%d | "
        "grid_points=%d | fft_tolerance=%.1e",
        len(spectrum_cfg["cases"]),
        draws,
        n_points,
        tolerance,
    )

    checks = []
    for case in spectrum_cfg["cases"]:
        scaling = float(case["encoding_scaling"])
        predicted = [value / scaling for value in photonic_predicted_spectrum(case)]
        predicted_degree = max(abs(value) for value in predicted)
        if n_points <= 2 * predicted_degree + 1:
            raise ValueError(
                f"grid_points={n_points} cannot resolve degree {predicted_degree}"
            )
        model = build_photonic_model(case, dtype).to(device)
        # One period of the encoding: frequency w * s maps to FFT index w.
        grid = torch.tensor(
            periodic_grid(n_points, period=TWO_PI / scaling), dtype=dtype, device=device
        )
        value_draws = []
        for _ in range(draws):
            model.reset_parameters(generator, init_high)
            model.to(device)
            with torch.no_grad():
                probabilities = model.probabilities(grid).cpu().numpy()
                readout = model(grid).cpu().numpy()
            value_draws.extend(probabilities.T)
            value_draws.append(readout)
        record = _support_check(
            f"photonic_{case['name']}",
            np.array(predicted),
            value_draws,
            tolerance,
            {
                "kind": "photonic",
                "computation_space": case["computation_space"],
                "n_photons": int(case["n_photons"]),
                "n_modes": int(case["n_modes"]),
                "encoded_modes": list(case["encoded_modes"]),
                "encoding_layers": int(case["encoding_layers"]),
                "input_state": model.input_state,
                "n_outcomes": len(model.output_keys),
                "expect_band_limited": bool(case["expect_band_limited"]),
            },
        )
        band_limited = record["subset"]
        record["band_limited"] = band_limited
        record["passed"] = (
            record["equal"] if record["expect_band_limited"] else not band_limited
        )
        LOGGER.info(
            "CHECK_VERDICT | check=%s | space=%s | band_limited=%s | expected_band_limited=%s | passed=%s",
            record["name"],
            case["computation_space"],
            band_limited,
            record["expect_band_limited"],
            record["passed"],
        )
        checks.append(record)

    return {
        "experiment": "photonic_spectrum",
        "adaptation": "MerLin photonic adaptation (Phase 4), not the paper's model",
        "fft_tolerance": tolerance,
        "random_draws": draws,
        "grid_points": n_points,
        "seed": seed,
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
    }


def run_coefficient_reach_experiment(
    reach_cfg: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Maximise one Fourier coefficient of a model's output (Phase 4 diagnostic A).

    For each case, ``|c_k|`` of ``f(x) = sum_w c_w exp(i w s x)`` is
    maximised over all trainable parameters from several random
    initialisations. Here ``k`` is the target's top frequency. The
    coefficient is an exact, differentiable FFT over one period ``2 pi / s``.
    The best value found is a lower bound on the model's largest reachable
    ``|c_k|``. If it stays below the target's ``|c_k|``, the target is out of
    reach whatever the training budget.

    Parameters
    ----------
    reach_cfg : Mapping
        Keys ``lr``, ``steps``, ``init_seeds``, ``init_high`` and ``cases``.
        Each case has ``name``, ``kind`` (a fit model kind), ``model``,
        ``frequency`` (k) and ``target_abs_coefficient``.
    run_dir : Path
        Run directory (no artifacts beyond metrics).
    device : torch.device
        Device for the models.
    dtype : torch.dtype
        Real dtype.

    Returns
    -------
    dict
        Per case: best and median ``|c_k|`` over initialisations, the target
        value and whether it was reached.
    """
    learning_rate = float(reach_cfg["lr"])
    steps = int(reach_cfg["steps"])
    init_seeds = [int(seed) for seed in reach_cfg["init_seeds"]]
    init_high = float(reach_cfg["init_high"])
    LOGGER.info(
        "DATASET_READY | name=none (coefficient maximisation, no data) | cases=%d | inits=%d",
        len(reach_cfg["cases"]),
        len(init_seeds),
    )
    LOGGER.info(
        "TRAINING_STARTED | objective=maximise |c_k| | optimizer=adam | lr=%g | steps=%d | inits=%s",
        learning_rate,
        steps,
        init_seeds,
    )

    records = []
    for case in reach_cfg["cases"]:
        frequency = int(case["frequency"])
        model_cfg = case["model"]
        model_frequencies_, scaling = model_frequencies(model_cfg)
        if frequency > max(model_frequencies_):
            raise ValueError(
                f"case {case['name']}: frequency {frequency} exceeds the model's degree {max(model_frequencies_)}"
            )
        n_points = 4 * max(model_frequencies_) + 8
        grid = torch.tensor(
            periodic_grid(n_points, period=TWO_PI / scaling), dtype=dtype, device=device
        )
        phase = torch.exp(
            -1j
            * frequency
            * torch.arange(n_points, dtype=dtype, device=device)
            * TWO_PI
            / n_points
        )
        best_per_init = []
        for init_seed in init_seeds:
            model = build_fit_model(case["kind"], model_cfg, dtype)
            model.reset_parameters(torch.Generator().manual_seed(init_seed), init_high)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            best = 0.0
            label = f"case={case['name']} | init_seed={init_seed}"
            for step in range(1, steps + 1):
                coefficient = (model(grid).to(phase.dtype) * phase).mean()
                magnitude = coefficient.abs()
                optimizer.zero_grad()
                (-magnitude).backward()
                optimizer.step()
                best = max(best, float(magnitude))
                LOGGER.info(
                    "TRAIN_STEP_COMPLETED | %s | step=%d/%d | abs_c=%.6e | best=%.6e",
                    label,
                    step,
                    steps,
                    float(magnitude),
                    best,
                )
            with torch.no_grad():
                final = float((model(grid).to(phase.dtype) * phase).mean().abs())
            best_per_init.append(max(best, final))
        target = float(case["target_abs_coefficient"])
        record = {
            "name": case["name"],
            "kind": case["kind"],
            "model": dict(model_cfg),
            "frequency": frequency,
            "model_degree": max(model_frequencies_),
            "target_abs_coefficient": target,
            "best_abs_coefficient": max(best_per_init),
            "median_abs_coefficient": float(np.median(best_per_init)),
            "per_init_best": best_per_init,
            "target_reached": max(best_per_init) >= target,
        }
        records.append(record)
        LOGGER.info(
            "EVALUATION_COMPLETED | case=%s | k=%d | best_abs_c=%.4e | median_abs_c=%.4e | target_abs_c=%.4e | reached=%s",
            case["name"],
            frequency,
            record["best_abs_coefficient"],
            record["median_abs_coefficient"],
            target,
            record["target_reached"],
        )

    return {
        "experiment": "coefficient_reach",
        "adaptation": "Phase 4 diagnostic: lower bound on the reachable |c_k| per model",
        "lr": learning_rate,
        "steps": steps,
        "init_seeds": init_seeds,
        "cases": records,
    }
