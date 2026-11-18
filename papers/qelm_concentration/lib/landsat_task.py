"""Landsat classification with the QELM (E5, E3, E4, E7, E8) and classical baselines (E10).

QELM pipeline per (split seed, reservoir seed):

1. simulate the exact output states and keep the RDMs of the disjoint pairs;
2. compute the 36 randomized-measurement probabilities per pair (exact);
3. per shot seed, sample finite-shot frequencies, build each readout, fit ridge.

Readout names are ``w{1,2}_{x,pauli,et,etcut}_{unit,scaled}`` (paper Fig. 4):
``x`` = Pauli strings from {I, X}; ``pauli`` = all non-identity strings on the subset;
``et`` = local eigentasks; ``etcut`` = eigentasks with ``beta^2 / S <= lambda``;
``unit`` = unit-variance; ``scaled`` = signal-aware (Pauli/X) or NSR-aware (eigentasks).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .baselines import (
    polynomial_features,
    random_fourier_features,
    random_tanh_features,
    run_paper_baselines,
    summarise_runs,
    tune_and_evaluate,
)
from .data import LANDSAT_N_FEATURES, make_landsat_split
from .eigentasks import (
    EigentaskBasis,
    fit_local_eigentasks,
    nsr_cutoff_mask,
    nsr_scale,
    resolvable_expressive_capacity,
    signal_scale,
    unit_scale,
)
from .observables import (
    TWO_QUBIT_PAULIS,
    TWO_QUBIT_X_PAULIS,
    pair_povm_probabilities,
    sample_pair_frequencies,
    shadow_pair_paulis,
    shadow_single_paulis,
    single_qubit_frequencies,
)
from .qelm import architecture_from_config, build_reservoir, real_dtype_from_config
from .readout import alpha_grid, classification_metrics, fit_ridge_classifier
from .reservoir import ReservoirHyperparameters, simulate_pair_rdms

logger = logging.getLogger(__name__)

PAPER_DIR = Path(__file__).resolve().parents[1]
# Config sections that determine the exact probabilities of a realisation. A saved
# probability file is reused only when all of them agree with the current run.
PROBABILITY_DEFINING_KEYS = ("hyperparameters", "dtype")
PROBABILITY_DEFINING_RESERVOIR_KEYS = (
    "n_qubits",
    "n_layers",
    "encoding_period",
    "input_shift",
    "angle_scale",
    "brickwall_offset",
    "block_order",
    "bell_pairs",
)


def _dtype_label(value) -> str:
    return value.label if hasattr(value, "label") else str(value)


def _probability_fingerprint(cfg: dict) -> dict:
    """Config values that fix the exact pair-POVM probabilities of a realisation."""
    return {
        "hyperparameters": {
            key: value
            for key, value in cfg["hyperparameters"].items()
            if not key.startswith("_")
        },
        "dtype": _dtype_label(cfg["dtype"]),
        "reservoir": {
            key: cfg["reservoir"][key] for key in PROBABILITY_DEFINING_RESERVOIR_KEYS
        },
        # Snapshots written before `input_scaling` existed always used min-max scaling.
        "landsat": {"input_scaling": "minmax"}
        | {
            key: value
            for key, value in cfg["dataset"]["landsat"].items()
            if not key.startswith("_")
        },
    }


def realisation_pairs(exp: dict) -> list[tuple[int, int]]:
    """(split seed, reservoir seed) pairs: explicit ``realisations`` or the cross product."""
    if exp["realisations"] is not None:
        return [
            (int(split), int(reservoir)) for split, reservoir in exp["realisations"]
        ]
    return [
        (int(split), int(reservoir))
        for split in exp["split_seeds"]
        for reservoir in exp["reservoir_seeds"]
    ]


def load_saved_probabilities(
    cfg: dict, data, split_seed: int, reservoir_seed: int, sources: list[str]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load exact probabilities saved by an earlier run of the same configuration.

    Raises
    ------
    FileNotFoundError
        If no source run holds the file for this realisation.
    ValueError
        If the source run's configuration or the stored split differs from this run.
    """
    name = f"probabilities_split{split_seed}_res{reservoir_seed}.npz"
    for source in sources:
        source_dir = PAPER_DIR / source
        path = source_dir / name
        if not path.exists():
            continue
        source_cfg = json.loads((source_dir / "config_snapshot.json").read_text())
        if _probability_fingerprint(source_cfg) != _probability_fingerprint(cfg):
            raise ValueError(
                f"{source_dir} was produced with a different configuration"
            )
        saved = np.load(path)
        for key, current in (
            ("subset_indices", data.subset_indices),
            ("train_y", data.train_y),
            ("test_y", data.test_y),
        ):
            if not np.array_equal(saved[key], current):
                raise ValueError(f"{path} does not match the current split ({key})")
        return saved["train"], saved["test"], f"{source}/{name}"
    raise FileNotFoundError(f"No probability source holds {name}: {sources}")


READOUT_WEIGHTS = ("w1", "w2")
READOUT_KINDS = ("x", "pauli", "et", "etcut")
READOUT_SCALINGS = ("unit", "scaled")


def parse_readout(name: str) -> tuple[int, str, str]:
    """Split ``w2_etcut_scaled`` into ``(2, "etcut", "scaled")`` and validate it."""
    parts = name.split("_")
    if (
        len(parts) != 3
        or parts[0] not in READOUT_WEIGHTS
        or parts[1] not in READOUT_KINDS
        or parts[2] not in READOUT_SCALINGS
    ):
        raise ValueError(f"Invalid readout name {name!r}")
    return int(parts[0][1]), parts[1], parts[2]


def readout_features(
    name: str,
    train_pair_freq: np.ndarray,
    test_pair_freq: np.ndarray,
    shots: int,
    nsr_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build train/test features of one readout from pair frequencies.

    Parameters
    ----------
    name : str
        Readout name (see module docstring).
    train_pair_freq, test_pair_freq : numpy.ndarray
        Shape ``(n_samples, N/2, 36)``.
    shots : int
        Shots per sample.
    nsr_cutoff : float
        ``lambda`` of the ``etcut`` readouts.

    Returns
    -------
    tuple
        ``(train_features, test_features, info)``; ``info`` holds the feature count and,
        for eigentasks, the retained count and REC.
    """
    weight, kind, scaling = parse_readout(name)
    if weight == 1:
        train_sets = single_qubit_frequencies(train_pair_freq)
        test_sets = single_qubit_frequencies(test_pair_freq)
    else:
        train_sets, test_sets = train_pair_freq, test_pair_freq
    n_train, n_test = len(train_pair_freq), len(test_pair_freq)
    info: dict = {}

    if kind in ("x", "pauli"):
        if weight == 1:
            bases = (1,) if kind == "x" else (1, 2, 3)
            train = shadow_single_paulis(train_sets, bases).reshape(n_train, -1)
            test = shadow_single_paulis(test_sets, bases).reshape(n_test, -1)
        else:
            strings = TWO_QUBIT_X_PAULIS if kind == "x" else TWO_QUBIT_PAULIS
            train = shadow_pair_paulis(train_sets, strings).reshape(n_train, -1)
            test = shadow_pair_paulis(test_sets, strings).reshape(n_test, -1)
        if scaling == "unit":
            train, test, _ = unit_scale(train, test)
        else:
            train, test = signal_scale(train, test)
    else:
        basis = fit_local_eigentasks(train_sets, shots, weight)
        train, test, info = eigentask_readout(
            basis, train_sets, test_sets, shots, kind, scaling, nsr_cutoff
        )
    info["n_features"] = int(train.shape[1])
    return train, test, info


def eigentask_readout(
    basis: EigentaskBasis,
    train_sets: np.ndarray,
    test_sets: np.ndarray,
    shots: int,
    kind: str,
    scaling: str,
    nsr_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Eigentask (``et``) or NSR-cut (``etcut``) features with unit or NSR-aware scaling.

    Shared by the qubit readouts and the MerLin click readouts; ``basis`` already holds
    the eigentasks fitted on the training frequencies ``train_sets``.
    """
    train, test = basis.transform(train_sets), basis.transform(test_sets)
    nsr = basis.nsr
    info: dict = {
        "n_eigentasks_total": int(nsr.size),
        "rec_total": resolvable_expressive_capacity(nsr, shots),
    }
    if kind == "etcut":
        keep = nsr_cutoff_mask(nsr, shots, nsr_cutoff)
        train, test, nsr = train[:, keep], test[:, keep], nsr[keep]
        info["nsr_cutoff"] = float(nsr_cutoff)
        info["rec_cut"] = resolvable_expressive_capacity(basis.nsr, shots, nsr_cutoff)
    info["n_eigentasks_retained"] = int(nsr.size)
    if nsr.size == 0:
        # A cutoff below every beta^2 / S leaves no feature (paper Fig. A3, small lambda).
        pass
    elif scaling == "unit":
        train, test, _ = unit_scale(train, test)
    else:
        train, test = nsr_scale(train, test, nsr)
    return train, test, info


def _simulate_probabilities(
    cfg: dict, data, reservoir_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pair POVM probabilities of the train and test inputs for one realisation."""
    arch = architecture_from_config(cfg["reservoir"])
    hyper = ReservoirHyperparameters.from_config(cfg["hyperparameters"])
    reservoir = build_reservoir(
        arch, hyper, reservoir_seed, cfg["device"], real_dtype_from_config(cfg)
    )
    chunk = int(cfg["reservoir"]["chunk_size"])
    pairs = arch.disjoint_pairs()
    train_probs = pair_povm_probabilities(
        simulate_pair_rdms(reservoir, data.train_x, pairs, chunk)
    )
    test_probs = pair_povm_probabilities(
        simulate_pair_rdms(reservoir, data.test_x, pairs, chunk)
    )
    return train_probs, test_probs


def _evaluate_readout(
    name,
    train_freq,
    test_freq,
    train_y,
    test_y,
    shots,
    cutoff,
    alphas,
    features=readout_features,
) -> dict:
    train, test, info = features(name, train_freq, test_freq, shots, cutoff)
    if train.shape[1] == 0:
        # With no features the ridge readout reduces to its intercept: every test sample
        # gets the training-majority class.
        values, counts = np.unique(train_y, return_counts=True)
        prediction = np.full(len(test_y), values[np.argmax(counts)])
        train_prediction = np.full(len(train_y), values[np.argmax(counts)])
        train_metrics = classification_metrics(train_y, train_prediction)
        metrics = {
            **classification_metrics(test_y, prediction),
            "train_f1_macro": train_metrics["f1_macro"],
            "train_accuracy": train_metrics["accuracy"],
            "alpha": None,
        }
        return {**metrics, **info, "empty_readout": True}
    metrics = fit_ridge_classifier(train, train_y, test, test_y, alphas)
    return {**metrics, **info, "empty_readout": False}


def run_landsat_readouts(
    cfg: dict,
    run_dir: Path,
    simulate=_simulate_probabilities,
    features=readout_features,
) -> dict:
    """QELM readout study on Landsat (E5 headline, E7/E8 cutoff sweep, E3 shots, E4 sizes).

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    run_dir : pathlib.Path
        Run directory for optional probability files.
    simulate : callable
        ``simulate(cfg, data, reservoir_seed) -> (train_probs, test_probs)`` with
        per-subset outcome probabilities of shape ``(n_samples, n_subsets, n_effects)``.
        Default value is the qubit reservoir (`_simulate_probabilities`); the MerLin
        adaptation passes its photonic reservoir.
    features : callable
        ``features(name, train_freq, test_freq, shots, cutoff)`` as `readout_features`.
        Default value is `readout_features`.

    Returns
    -------
    dict
        ``records``: one entry per evaluation (analysis, readout, seeds, metrics);
        ``summary``: mean, std and percentiles per analysis group over seeds/resamples.
    """
    exp = cfg["landsat_readouts"]
    alphas = alpha_grid(cfg["readout"])
    headline_shots = int(exp["headline_shots"])
    cutoff = float(exp["nsr_cutoff"])
    records: list[dict] = []
    pairs = realisation_pairs(exp)
    n_realisations = len(pairs)
    completed_realisations = 0
    logger.info(
        "TRAINING_STARTED | readout=closed_form_ridge | realisations=%d | shot_seeds=%d | readouts=%d",
        n_realisations,
        len(exp["shot_seeds"]),
        len(exp["readouts"]),
    )

    logger.info(
        "DATASET_READY | dataset=landsat | split_seeds=%s | n_subsample=%s | test_fraction=%s",
        list(exp["split_seeds"]),
        cfg["dataset"]["landsat"]["n_subsample"],
        cfg["dataset"]["landsat"]["test_fraction"],
    )
    for split_seed in dict.fromkeys(split for split, _ in pairs):
        data = make_landsat_split(cfg, int(split_seed))
        logger.info(
            "DATASET_SPLIT_READY | dataset=landsat | split_seed=%s | train=%d | test=%d | input_dim=%d",
            split_seed,
            len(data.train_y),
            len(data.test_y),
            data.train_x.shape[1],
        )
        for reservoir_seed in [res for split, res in pairs if split == split_seed]:
            if exp["probability_sources"] is None:
                logger.info(
                    "SIMULATION_STARTED | split_seed=%s | reservoir_seed=%s",
                    split_seed,
                    reservoir_seed,
                )
                train_probs, test_probs = simulate(cfg, data, int(reservoir_seed))
            else:
                train_probs, test_probs, source = load_saved_probabilities(
                    cfg,
                    data,
                    int(split_seed),
                    int(reservoir_seed),
                    exp["probability_sources"],
                )
                logger.info(
                    "ARTIFACT_LOADED | split_seed=%s | reservoir_seed=%s | path=%s",
                    split_seed,
                    reservoir_seed,
                    source,
                )
            if exp["save_probabilities"]:
                path = (
                    run_dir / f"probabilities_split{split_seed}_res{reservoir_seed}.npz"
                )
                np.savez_compressed(
                    path,
                    train=train_probs,
                    test=test_probs,
                    train_y=data.train_y,
                    test_y=data.test_y,
                    subset_indices=data.subset_indices,
                )
                logger.info("ARTIFACT_WRITTEN | path=%s", path.name)
            context = {
                "split_seed": int(split_seed),
                "reservoir_seed": int(reservoir_seed),
            }

            for shot_seed in exp["shot_seeds"]:
                rng = np.random.default_rng(
                    [int(split_seed), int(reservoir_seed), int(shot_seed)]
                )
                train_freq = sample_pair_frequencies(train_probs, headline_shots, rng)
                test_freq = sample_pair_frequencies(test_probs, headline_shots, rng)
                base = {**context, "shot_seed": int(shot_seed), "shots": headline_shots}

                for name in exp["readouts"]:
                    result = _evaluate_readout(
                        name,
                        train_freq,
                        test_freq,
                        data.train_y,
                        data.test_y,
                        headline_shots,
                        cutoff,
                        alphas,
                        features=features,
                    )
                    records.append(
                        {"analysis": "readouts", "readout": name, **base, **result}
                    )
                    logger.info(
                        "EVALUATION_COMPLETED | analysis=readouts | readout=%s | %s | f1_macro=%.4f",
                        name,
                        _context_text(base),
                        result["f1_macro"],
                    )

                for sweep_cutoff in exp["cutoff_sweep"]["cutoffs"]:
                    for name in exp["cutoff_sweep"]["readouts"]:
                        result = _evaluate_readout(
                            name,
                            train_freq,
                            test_freq,
                            data.train_y,
                            data.test_y,
                            headline_shots,
                            float(sweep_cutoff),
                            alphas,
                            features=features,
                        )
                        records.append(
                            {
                                "analysis": "cutoff_sweep",
                                "readout": name,
                                **base,
                                **result,
                            }
                        )
                        logger.info(
                            "EVALUATION_COMPLETED | analysis=cutoff_sweep | readout=%s | lambda=%g | retained=%d | f1_macro=%.4f",
                            name,
                            sweep_cutoff,
                            result["n_eigentasks_retained"],
                            result["f1_macro"],
                        )

                sizes_cfg = exp["train_size_sweep"]
                size_rng = np.random.default_rng([int(split_seed), int(shot_seed), 7])
                for size in sizes_cfg["sizes"]:
                    for resample in range(int(sizes_cfg["resamples"])):
                        chosen = size_rng.choice(
                            len(data.train_y), size=int(size), replace=True
                        )
                        result = _evaluate_readout(
                            sizes_cfg["readout"],
                            train_freq[chosen],
                            test_freq,
                            data.train_y[chosen],
                            data.test_y,
                            headline_shots,
                            cutoff,
                            alphas,
                            features=features,
                        )
                        records.append(
                            {
                                "analysis": "train_size_sweep",
                                "readout": sizes_cfg["readout"],
                                "train_size": int(size),
                                "resample": resample,
                                **base,
                                **result,
                            }
                        )
                    logger.info(
                        "EVALUATION_COMPLETED | analysis=train_size_sweep | size=%s | resamples=%s",
                        size,
                        sizes_cfg["resamples"],
                    )

            shots_cfg = exp["shots_sweep"]
            for shots in shots_cfg["shots"]:
                for shot_seed in exp["shot_seeds"]:
                    rng = np.random.default_rng(
                        [
                            int(split_seed),
                            int(reservoir_seed),
                            int(shot_seed),
                            int(shots),
                        ]
                    )
                    train_freq = sample_pair_frequencies(train_probs, int(shots), rng)
                    test_freq = sample_pair_frequencies(test_probs, int(shots), rng)
                    result = _evaluate_readout(
                        shots_cfg["readout"],
                        train_freq,
                        test_freq,
                        data.train_y,
                        data.test_y,
                        int(shots),
                        cutoff,
                        alphas,
                        features=features,
                    )
                    records.append(
                        {
                            "analysis": "shots_sweep",
                            "readout": shots_cfg["readout"],
                            **context,
                            "shot_seed": int(shot_seed),
                            "shots": int(shots),
                            **result,
                        }
                    )
                logger.info(
                    "EVALUATION_COMPLETED | analysis=shots_sweep | shots=%s | %s",
                    shots,
                    _context_text(context),
                )

            completed_realisations += 1
            logger.info(
                "TRAIN_REALISATION_COMPLETED | realisation=%d/%d | split_seed=%s | reservoir_seed=%s",
                completed_realisations,
                n_realisations,
                split_seed,
                reservoir_seed,
            )

    return {"records": records, "summary": summarise_records(records)}


def _context_text(context: dict) -> str:
    return " | ".join(f"{key}={value}" for key, value in context.items())


SUMMARY_KEYS = {
    "readouts": ("readout",),
    "cutoff_sweep": ("readout", "nsr_cutoff"),
    "train_size_sweep": ("readout", "train_size"),
    "shots_sweep": ("readout", "shots"),
}
SUMMARY_METRICS = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "n_features",
    "n_eigentasks_retained",
    "rec_total",
    "rec_cut",
)


def summarise_records(records: list[dict]) -> dict:
    """Mean, std and 2.5/97.5 percentiles per analysis group over all seeds/resamples."""
    summary: dict = {}
    for analysis, keys in SUMMARY_KEYS.items():
        groups: dict[str, list[dict]] = {}
        for record in records:
            if record["analysis"] != analysis:
                continue
            label = "|".join(f"{key}={record.get(key)}" for key in keys)
            groups.setdefault(label, []).append(record)
        summary[analysis] = {}
        for label, items in groups.items():
            entry: dict = {"n": len(items)}
            for metric in SUMMARY_METRICS:
                values = np.array(
                    [item[metric] for item in items if metric in item], dtype=float
                )
                if values.size:
                    entry[f"{metric}_mean"] = float(values.mean())
                    entry[f"{metric}_std"] = float(values.std())
                    entry[f"{metric}_p2.5"] = float(np.percentile(values, 2.5))
                    entry[f"{metric}_p97.5"] = float(np.percentile(values, 97.5))
            summary[analysis][label] = entry
    return summary


def run_landsat_classical(cfg: dict, run_dir: Path) -> dict:
    """Classical baselines on the same Landsat splits (E10 and the Fair Baseline Plan).

    Includes the paper's nine untuned models, ridge on degree-2 features, matched-dimension
    random-feature ELMs (optionally with Gaussian feature noise), and tuned RF/kNN/SVM.
    """
    exp = cfg["landsat_classical"]
    alphas = alpha_grid(cfg["readout"])
    results: dict = {"splits": {}}
    logger.info(
        "TRAINING_STARTED | stage=classical_baselines | splits=%d | paper_baselines=%s",
        len(exp["split_seeds"]),
        list(exp["paper_baselines"]),
    )
    logger.info(
        "DATASET_READY | dataset=landsat | split_seeds=%s | n_subsample=%s | test_fraction=%s",
        list(exp["split_seeds"]),
        cfg["dataset"]["landsat"]["n_subsample"],
        cfg["dataset"]["landsat"]["test_fraction"],
    )
    for split_index, split_seed in enumerate(exp["split_seeds"], start=1):
        data = make_landsat_split(cfg, int(split_seed))
        logger.info(
            "DATASET_SPLIT_READY | dataset=landsat | split_seed=%s | train=%d | test=%d",
            split_seed,
            len(data.train_y),
            len(data.test_y),
        )
        split_results: dict = {}
        split_results["paper_baselines"] = run_paper_baselines(
            data.train_raw,
            data.train_y,
            data.test_raw,
            data.test_y,
            list(exp["paper_baselines"]),
            int(exp["n_inits"]),
            int(split_seed),
        )

        poly_train, poly_test = polynomial_features(
            *unit_scale(data.train_raw, data.test_raw)[:2]
        )
        poly_train, poly_test, _ = unit_scale(poly_train, poly_test)
        split_results["ridge_degree2"] = fit_ridge_classifier(
            poly_train, data.train_y, poly_test, data.test_y, alphas
        )
        split_results["ridge_degree2"]["n_features"] = int(poly_train.shape[1])
        logger.info(
            "EVALUATION_COMPLETED | baseline=ridge_degree2 | split_seed=%s | f1_macro=%.4f",
            split_seed,
            split_results["ridge_degree2"]["f1_macro"],
        )

        scaled_train = data.train_x[:, :LANDSAT_N_FEATURES]
        scaled_test = data.test_x[:, :LANDSAT_N_FEATURES]
        elm_cfg = exp["matched_elm"]
        split_results["matched_elm"] = {}
        for n_features in elm_cfg["n_features"]:
            for family in elm_cfg["families"]:
                for noise_std in elm_cfg["feature_noise_std"]:
                    runs = []
                    for init in range(int(elm_cfg["n_inits"])):
                        seed = int(split_seed) * 1000 + init
                        if family == "tanh":
                            train, test = random_tanh_features(
                                scaled_train,
                                scaled_test,
                                int(n_features),
                                float(elm_cfg["tanh_input_scale"]),
                                seed,
                            )
                        elif family == "fourier":
                            train, test = random_fourier_features(
                                scaled_train,
                                scaled_test,
                                int(n_features),
                                float(elm_cfg["fourier_bandwidth"]),
                                seed,
                            )
                        else:
                            raise ValueError(f"Unknown ELM family {family!r}")
                        if noise_std:
                            noise_rng = np.random.default_rng(seed + 1)
                            train = train + noise_rng.normal(0, noise_std, train.shape)
                            test = test + noise_rng.normal(0, noise_std, test.shape)
                        train, test, _ = unit_scale(train, test)
                        runs.append(
                            fit_ridge_classifier(
                                train, data.train_y, test, data.test_y, alphas
                            )
                        )
                    label = f"{family}|n_features={n_features}|noise_std={noise_std}"
                    split_results["matched_elm"][label] = summarise_runs(runs)
                    logger.info(
                        "EVALUATION_COMPLETED | baseline=matched_elm | %s | split_seed=%s | f1_macro=%.4f",
                        label,
                        split_seed,
                        split_results["matched_elm"][label]["f1_macro_mean"],
                    )

        split_results["tuned"] = {}
        for name, grid in exp["tuned_baselines"]["grids"].items():
            split_results["tuned"][name] = tune_and_evaluate(
                name,
                grid,
                data.train_raw,
                data.train_y,
                data.test_raw,
                data.test_y,
                float(exp["tuned_baselines"]["validation_fraction"]),
                int(split_seed),
            )
            logger.info(
                "EVALUATION_COMPLETED | baseline=tuned_%s | split_seed=%s | params=%s | f1_macro=%.4f",
                name,
                split_seed,
                split_results["tuned"][name]["params"],
                split_results["tuned"][name]["f1_macro"],
            )
        results["splits"][str(split_seed)] = split_results
        logger.info(
            "TRAIN_SPLIT_COMPLETED | split=%d/%d | split_seed=%s",
            split_index,
            len(exp["split_seeds"]),
            split_seed,
        )
    return results


def run_landsat_gap_diagnostics(cfg: dict, run_dir: Path) -> dict:
    """Gap diagnostics D1/D2 on saved exact probabilities (no simulation).

    For each listed realisation and readout:

    - ``exact``: features from the exact pair-POVM probabilities (infinite shots; X and
      Pauli readouts only, since eigentask NSRs need a finite shot count);
    - ``finite``: features from ``shots`` sampled shots (shot seed 0), with train and test
      F1, to separate underfitting from overfitting.
    """
    exp = cfg["landsat_gap_diagnostics"]
    alphas = alpha_grid(cfg["readout"])
    shots = int(exp["shots"])
    pairs = realisation_pairs(exp)
    records: list[dict] = []
    logger.info(
        "TRAINING_STARTED | readout=closed_form_ridge | realisations=%d | readouts=%d",
        len(pairs),
        len(exp["readouts"]),
    )
    logger.info(
        "DATASET_READY | dataset=landsat | realisations=%s | n_subsample=%s",
        pairs,
        cfg["dataset"]["landsat"]["n_subsample"],
    )
    for index, (split_seed, reservoir_seed) in enumerate(pairs, start=1):
        data = make_landsat_split(cfg, split_seed)
        train_probs, test_probs, source = load_saved_probabilities(
            cfg, data, split_seed, reservoir_seed, exp["probability_sources"]
        )
        logger.info("ARTIFACT_LOADED | path=%s", source)
        rng = np.random.default_rng([split_seed, reservoir_seed, 0])
        train_freq = sample_pair_frequencies(train_probs, shots, rng)
        test_freq = sample_pair_frequencies(test_probs, shots, rng)
        context = {"split_seed": split_seed, "reservoir_seed": reservoir_seed}
        for name in exp["readouts"]:
            _, kind, _ = parse_readout(name)
            if kind in ("x", "pauli"):
                exact = _evaluate_readout(
                    name,
                    train_probs,
                    test_probs,
                    data.train_y,
                    data.test_y,
                    shots,
                    1.0,
                    alphas,
                )
                records.append(
                    {"condition": "exact", "readout": name, **context, **exact}
                )
            finite = _evaluate_readout(
                name,
                train_freq,
                test_freq,
                data.train_y,
                data.test_y,
                shots,
                1.0,
                alphas,
            )
            records.append(
                {"condition": f"shots_{shots}", "readout": name, **context, **finite}
            )
            logger.info(
                "EVALUATION_COMPLETED | readout=%s | split_seed=%s | reservoir_seed=%s | test_f1=%.4f | train_f1=%.4f",
                name,
                split_seed,
                reservoir_seed,
                finite["f1_macro"],
                finite["train_f1_macro"],
            )
        logger.info(
            "TRAIN_REALISATION_COMPLETED | realisation=%d/%d", index, len(pairs)
        )
    summary: dict = {}
    for record in records:
        key = f"{record['condition']}|{record['readout']}"
        summary.setdefault(key, []).append(record)
    return {
        "records": records,
        "summary": {
            key: {
                metric: {
                    "mean": float(np.mean([r[metric] for r in items])),
                    "std": float(np.std([r[metric] for r in items])),
                }
                for metric in ("f1_macro", "train_f1_macro", "accuracy")
            }
            | {"n": len(items)}
            for key, items in summary.items()
        },
    }
