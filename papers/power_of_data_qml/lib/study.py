"""Experiment orchestration for the paper's numerical studies.

One *unit* is a (seed, system size n) pair. Within a unit the classical kernels
are built once, and each embedding's quantum bank (states simulated once) feeds
every requested component:

====================  =====================================  ==================
Component             Paper evidence                          Records
====================  =====================================  ==================
dimension_geometry    arXiv Fig. 1a (journal Fig. 3a)         ``d`` and ``g``
quantum_dataset       arXiv Fig. 1b, Datasets (Q, E*)         MAE per family
classical_dataset     arXiv Fig. 1b, Dataset (C)              error per family
bound                 arXiv Fig. S3 (journal Supp. Fig. 4)    Eq. K.1 bound
engineered_pqk        arXiv Fig. 2 (journal Fig. 4)           accuracy per N
engineered_qk         arXiv Fig. S1 (journal Supp. Fig. 2)    accuracy per N
====================  =====================================  ==================

The training-size study (arXiv Fig. S2) has its own entry point,
:func:`run_training_size_study`.

Random streams are derived from ``(seed, stream, embedding, n)`` so that each
quantity is reproducible on its own. Per seed the subsample indices are shared
across system sizes (only the number of PCA components changes).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import embeddings, geometry, kernels, models, photonic_merlin
from .data import PCAFeatures, subsample_indices

LOGGER = logging.getLogger(__name__)

STREAM_SUBSAMPLE_ALL = 1
STREAM_SUBSAMPLE_CLASSICAL = 2
STREAM_HAAR = 3
STREAM_QNN = 4
STREAM_ENGINEERED_PQK = 5
STREAM_ENGINEERED_QK = 6
STREAM_MODELS = 7
STREAM_PHOTONIC_UNITARIES = 8
EMBEDDING_INDEX = {"E1": 1, "E2": 2, "E3": 3, "PA": 4, "PB": 5}

COMPONENTS = (
    "dimension_geometry",
    "quantum_dataset",
    "classical_dataset",
    "bound",
    "engineered_pqk",
    "engineered_qk",
)


def stream_rng(
    seed: int, stream: int, embedding: str | None = None, n: int = 0
) -> np.random.Generator:
    """Independent generator for one (seed, stream, embedding, n) combination."""
    return np.random.default_rng([seed, stream, EMBEDDING_INDEX.get(embedding, 0), n])


@dataclass
class StudySettings:
    """Validated settings of the ``study`` experiment.

    Every field maps to a required config key (see configs/defaults.json);
    missing keys raise ``KeyError`` instead of falling back to a default.
    """

    embeddings: tuple[str, ...]
    system_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    components: tuple[str, ...]
    subsample_size: int
    quantum_dataset_train_size: int
    classical_dataset_train_size: int
    classical_dataset_classes: tuple[int, ...]
    engineered_train_sizes: tuple[int, ...]
    bound_max_system_size: int
    quantum_kernel_max_qubits: int
    cv_system_sizes: tuple[int, ...]
    protocols: tuple[str, ...]
    grid: str
    include_polynomial_extension: bool
    pqk_gamma_scale_geometry: float
    geometry_gamma_scales: tuple[float, ...]
    lambdas: tuple[float, ...]
    g_tra_threshold: float
    random_label_probability: float
    engineered_label_threshold: str
    n_jobs: int
    cv_folds: int
    mlp_max_iter: int
    svr_epsilon: float
    batch_memory_mb: float
    gram_memory_mb: float
    e1_analytic: bool
    device: str
    photonic_max_photons: int | None

    @classmethod
    def from_config(cls, cfg: dict) -> StudySettings:
        study = cfg["study"]
        kernel_cfg = cfg["kernels"]
        model_cfg = cfg["models"]
        engineered_cfg = cfg["engineered"]
        simulation_cfg = cfg["simulation"]
        components = tuple(study["components"])
        unknown = set(components) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown study components: {sorted(unknown)}")
        unknown_embeddings = set(study["embeddings"]) - set(
            embeddings.EMBEDDINGS + photonic_merlin.PHOTONIC_EMBEDDINGS
        )
        if unknown_embeddings:
            raise ValueError(f"Unknown embeddings: {sorted(unknown_embeddings)}")
        return cls(
            embeddings=tuple(study["embeddings"]),
            system_sizes=tuple(int(n) for n in study["system_sizes"]),
            seeds=tuple(int(s) for s in study["seeds"]),
            components=components,
            subsample_size=int(study["subsample_size"]),
            quantum_dataset_train_size=int(study["quantum_dataset_train_size"]),
            classical_dataset_train_size=int(study["classical_dataset_train_size"]),
            classical_dataset_classes=tuple(
                int(c) for c in study["classical_dataset_classes"]
            ),
            engineered_train_sizes=tuple(
                int(n) for n in study["engineered_train_sizes"]
            ),
            bound_max_system_size=int(study["bound_max_system_size"]),
            quantum_kernel_max_qubits=int(study["quantum_kernel_max_qubits"]),
            cv_system_sizes=tuple(int(n) for n in study["cv_system_sizes"]),
            protocols=tuple(model_cfg["protocols"]),
            grid=str(model_cfg["grid"]),
            include_polynomial_extension=bool(
                model_cfg["include_polynomial_extension"]
            ),
            pqk_gamma_scale_geometry=float(kernel_cfg["pqk_gamma_scale_geometry"]),
            geometry_gamma_scales=tuple(
                float(s) for s in kernel_cfg["geometry_gamma_scales"]
            ),
            lambdas=tuple(float(v) for v in kernel_cfg["lambdas"]),
            g_tra_threshold=float(kernel_cfg["g_tra_threshold"]),
            random_label_probability=float(engineered_cfg["random_label_probability"]),
            engineered_label_threshold=str(engineered_cfg["label_threshold"]),
            n_jobs=int(model_cfg["n_jobs"]),
            cv_folds=int(model_cfg["cv_folds"]),
            mlp_max_iter=int(model_cfg["mlp_max_iter"]),
            svr_epsilon=float(model_cfg["svr_epsilon"]),
            batch_memory_mb=float(simulation_cfg["batch_memory_mb"]),
            gram_memory_mb=float(simulation_cfg["gram_memory_mb"]),
            e1_analytic=bool(simulation_cfg["e1_analytic"]),
            device=str(cfg["device"]),
            photonic_max_photons=_optional_photon_cap(cfg["photonic"]["max_photons"]),
        )

    def photonic_layout(self, embedding: str, n: int) -> dict:
        """Modes, photons and Fock dimension of a photonic unit (logged per record)."""
        n_modes = photonic_merlin.mode_count(embedding, n)
        n_photons = photonic_merlin.photon_count(n_modes, self.photonic_max_photons)
        return {
            "n_photons": n_photons,
            "fock_dimension": photonic_merlin.fock_dimension(n_modes, n_photons),
        }

    def protocols_for(self, n: int) -> tuple[str, ...]:
        """Protocols run at system size ``n`` (CV only at ``cv_system_sizes``)."""
        return tuple(
            p for p in self.protocols if p != "cv" or n in self.cv_system_sizes
        )


# --------------------------------------------------------------------------- inputs


@dataclass
class ClassicalInputs:
    """Classical kernels and features of one subsample at one system size."""

    features: np.ndarray
    geometry_kernels: dict[str, np.ndarray]
    model_inputs: dict[str, np.ndarray]
    model_kernel_keys: dict[str, list[tuple[str, dict]]]
    paper_spectra: list[geometry.ClassicalSpectrum]
    extension_spectra: list[geometry.ClassicalSpectrum]


def build_classical_inputs(
    features: np.ndarray, n: int, settings: StudySettings
) -> ClassicalInputs:
    """Classical kernels for the ``g`` minimization and the model grids."""
    degrees = (
        models.PAPER_GRIDS["polynomial_degrees"]
        if settings.grid == "paper"
        else models.SMOKE_GRIDS["polynomial_degrees"]
    )
    degrees = tuple(degrees) if settings.include_polynomial_extension else ()
    geometry_kernels = kernels.classical_geometry_kernels(
        features,
        n,
        gamma_scales=settings.geometry_gamma_scales,
        polynomial_degrees=degrees,
    )
    paper_names = ["linear"] + [f"rbf_{s:g}" for s in settings.geometry_gamma_scales]
    paper_spectra = [
        geometry.ClassicalSpectrum.from_kernel(k, geometry_kernels[k])
        for k in paper_names
    ]
    extension_spectra = paper_spectra + [
        geometry.ClassicalSpectrum.from_kernel(k, geometry_kernels[k])
        for k in geometry_kernels
        if k.startswith("poly_")
    ]

    grids = models.PAPER_GRIDS if settings.grid == "paper" else models.SMOKE_GRIDS
    normalizer = kernels.bandwidth_normalizer(features, n)
    distances = kernels.squared_distances(features)
    model_inputs = {"features": features, "linear": kernels.linear_kernel(features)}
    model_kernel_keys: dict[str, list[tuple[str, dict]]] = {
        "linear_kernel": [("linear", {})]
    }
    model_kernel_keys["gaussian_kernel"] = []
    for scale in grids["svm_gamma_scales"]:
        key = f"rbf_{scale:g}"
        model_inputs[key] = kernels.rbf_kernel_from_distances(
            distances, scale * normalizer
        )
        model_kernel_keys["gaussian_kernel"].append((key, {"gamma_scale": scale}))
    if settings.include_polynomial_extension:
        model_kernel_keys["polynomial_kernel"] = []
        for degree in degrees:
            key = f"poly_{degree}"
            model_inputs[key] = kernels.polynomial_kernel(features, degree, normalizer)
            model_kernel_keys["polynomial_kernel"].append((key, {"degree": degree}))
    return ClassicalInputs(
        features,
        geometry_kernels,
        model_inputs,
        model_kernel_keys,
        paper_spectra,
        extension_spectra,
    )


def pqk_model_kernels(
    pauli_features: np.ndarray, n: int, settings: StudySettings, prefix: str
) -> tuple[dict[str, np.ndarray], list[tuple[str, dict]]]:
    """PQK matrices over the model gamma grid (App. J.6)."""
    flat = kernels.pqk_flat_features(pauli_features)
    normalizer = kernels.bandwidth_normalizer(flat, n)
    distances = kernels.squared_distances(flat)
    grids = models.PAPER_GRIDS if settings.grid == "paper" else models.SMOKE_GRIDS
    matrices, keys = {}, []
    for scale in grids["svm_gamma_scales"]:
        key = f"{prefix}_{scale:g}"
        matrices[key] = kernels.projected_quantum_kernel(distances, scale, normalizer)
        keys.append((key, {"gamma_scale": scale}))
    return matrices, keys


def pqk_geometry_kernel(
    pauli_features: np.ndarray, n: int, gamma_scale: float
) -> np.ndarray:
    """PQK at the geometry bandwidth (AMB-13: ``gamma = scale / (n Var)``)."""
    flat = kernels.pqk_flat_features(pauli_features)
    return kernels.projected_quantum_kernel(
        kernels.squared_distances(flat),
        gamma_scale,
        kernels.bandwidth_normalizer(flat, n),
    )


# --------------------------------------------------------------------------- tasks


def classical_candidates(
    kind: str, classical: ClassicalInputs, settings: StudySettings, random_state: int
):
    """All classical-family candidates (paper suite plus optional extension)."""
    candidates = []
    for family, keys in classical.model_kernel_keys.items():
        family_candidates = models.kernel_family_candidates(
            family,
            kind,
            keys,
            grid_name=settings.grid,
            svr_epsilon=settings.svr_epsilon,
        )
        if family in models.EXTENSION_FAMILIES and kind == "regression":
            # libsvm SVR converges very slowly on the unbounded polynomial kernel at
            # large C (51 s for one fit at degree 3, C = 1024); KernelRidge spans the
            # same function class in closed form. Extension baseline only, not a
            # paper family.
            family_candidates = [
                c for c in family_candidates if c.params["estimator"] == "kernel_ridge"
            ]
        candidates += family_candidates
    candidates += models.feature_family_candidates(
        kind,
        grid_name=settings.grid,
        random_state=random_state,
        mlp_max_iter=settings.mlp_max_iter,
    )
    return candidates


def evaluate_task(
    task: models.Task,
    candidates: list[models.Candidate],
    settings: StudySettings,
    n: int,
    random_state: int,
    label: str,
    quantum_families: tuple[str, ...],
) -> dict:
    """Run every protocol on one task and attach the best-classical summaries."""
    LOGGER.info(
        "EVALUATION_STARTED | task=%s | n=%d | train=%d | test=%d | candidates=%d",
        label,
        n,
        task.train_indices.size,
        task.test_indices.size,
        len(candidates),
    )
    started = time.perf_counter()
    results = {}
    for protocol in settings.protocols_for(n):
        families = models.run_protocol(
            task,
            candidates,
            protocol=protocol,
            n_jobs=settings.n_jobs,
            cv_folds=settings.cv_folds,
            random_state=random_state,
        )
        summary = {
            "families": families,
            "best_classical": models.best_of(
                task.kind, families, models.CLASSICAL_FAMILIES
            ),
        }
        if settings.include_polynomial_extension:
            summary["best_classical_with_extension"] = models.best_of(
                task.kind,
                families,
                models.CLASSICAL_FAMILIES + models.EXTENSION_FAMILIES,
            )
        results[protocol] = summary
        quantum_scores = {
            f: families[f]["test_score"] for f in quantum_families if f in families
        }
        LOGGER.info(
            "EVALUATION_COMPLETED | task=%s | n=%d | protocol=%s | best_classical=%s:%.6f | quantum=%s",
            label,
            n,
            protocol,
            summary["best_classical"]["family"],
            summary["best_classical"]["test_score"],
            json.dumps({k: round(v, 6) for k, v in quantum_scores.items()}),
        )
    return {
        "train_size": int(task.train_indices.size),
        "test_size": int(task.test_indices.size),
        "metric": "accuracy" if task.kind == "classification" else "mae",
        "protocols": results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _geometry_record(result: geometry.GeometricDifferenceResult) -> dict:
    return {
        "g": result.g,
        "kernel": result.kernel,
        "lambda": result.lam,
        "g_tra": result.g_tra,
        "threshold_met": result.threshold_met,
        "per_kernel": result.per_kernel,
    }


def reset_peak_device_memory(device: str) -> None:
    """Reset CUDA peak-memory statistics, creating the CUDA context first.

    ``torch.cuda.reset_peak_memory_stats`` raises "Invalid device argument"
    before the first allocation on that device.
    """
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def _optional_photon_cap(value) -> int | None:
    if value is None:
        return None
    if int(value) != value or int(value) < 2:
        raise ValueError(
            f"photonic.max_photons must be null or an integer >= 2, got {value}"
        )
    return int(value)


# --------------------------------------------------------------------------- units


def _quantum_kernel_feasible(n_qubits: int, settings: StudySettings) -> bool:
    return n_qubits <= settings.quantum_kernel_max_qubits


def register_size(embedding: str, n: int, conventions) -> int:
    """Qubits (gate embeddings) or optical modes (photonic embeddings)."""
    if embedding in photonic_merlin.PHOTONIC_EMBEDDINGS:
        return photonic_merlin.mode_count(embedding, n)
    return embeddings.register_size(embedding, n, conventions)


def build_quantum_bank(
    embedding: str,
    features: np.ndarray,
    n: int,
    seed: int,
    settings: StudySettings,
    conventions: embeddings.EmbeddingConventions,
    complex_dtype,
    *,
    want_kernel: bool,
    want_labels: bool,
) -> embeddings.QuantumBank:
    """Quantum bank for a gate (E1-E3) or photonic MerLin (PA, PB) embedding.

    The fidelity kernel is built only when requested and the register (qubits
    or modes) is within ``quantum_kernel_max_qubits``. Random streams per
    (seed, n, embedding): E3 Haar state, gate QNN couplings, photonic fixed
    interferometers, photonic labelling interferometer.
    """
    n_register = register_size(embedding, n, conventions)
    need_kernel = want_kernel and _quantum_kernel_feasible(n_register, settings)
    if embedding in photonic_merlin.PHOTONIC_EMBEDDINGS:
        return photonic_merlin.compute_photonic_bank(
            embedding,
            features,
            input_scale=conventions.input_scale,
            unitary_rng=stream_rng(seed, STREAM_PHOTONIC_UNITARIES, embedding, n),
            label_rng=stream_rng(seed, STREAM_QNN, embedding, n)
            if want_labels
            else None,
            need_quantum_kernel=need_kernel,
            max_photons=settings.photonic_max_photons,
            device=settings.device,
            complex_dtype=complex_dtype,
            batch_memory_mb=settings.batch_memory_mb,
            gram_memory_mb=settings.gram_memory_mb,
        )
    haar_state = (
        embeddings.sample_haar_product_state(
            n_register, stream_rng(seed, STREAM_HAAR, embedding, n)
        )
        if embedding == "E3"
        else None
    )
    couplings = (
        stream_rng(seed, STREAM_QNN, embedding, n).standard_normal(
            embeddings.qnn_coupling_count(n_register, conventions)
        )
        if want_labels
        else None
    )
    return embeddings.compute_quantum_bank(
        embedding,
        features,
        conventions,
        haar_product_state=haar_state,
        qnn_couplings=couplings,
        need_quantum_kernel=need_kernel,
        device=settings.device,
        complex_dtype=complex_dtype,
        batch_memory_mb=settings.batch_memory_mb,
        e1_analytic=settings.e1_analytic,
    )


def run_embedding_unit(
    embedding: str,
    features: np.ndarray,
    classical: ClassicalInputs,
    n: int,
    seed: int,
    settings: StudySettings,
    conventions: embeddings.EmbeddingConventions,
    complex_dtype,
) -> dict:
    """All components of one (seed, n, embedding) on the all-class subsample."""
    components = set(settings.components)
    n_qubits = register_size(embedding, n, conventions)
    needs_qk = bool(
        components & {"dimension_geometry", "quantum_dataset", "engineered_qk"}
    )
    needs_labels = bool(components & {"quantum_dataset", "bound"})
    on_cuda = settings.device.startswith("cuda")
    if on_cuda:
        reset_peak_device_memory(settings.device)
    started = time.perf_counter()
    bank = build_quantum_bank(
        embedding,
        features,
        n,
        seed,
        settings,
        conventions,
        complex_dtype,
        want_kernel=needs_qk,
        want_labels=needs_labels,
    )
    record = {
        "seed": seed,
        "n": n,
        "embedding": embedding,
        "n_qubits": n_qubits,
        "quantum_kernel_available": bank.quantum_kernel is not None,
        "simulation_seconds": time.perf_counter() - started,
    }
    if on_cuda:
        record["simulation_peak_device_mb"] = (
            torch.cuda.max_memory_allocated(settings.device) / 2**20
        )
    if embedding in photonic_merlin.PHOTONIC_EMBEDDINGS:
        record.update(settings.photonic_layout(embedding, n))
    pqk_geometry = pqk_geometry_kernel(
        bank.pauli_features, n, settings.pqk_gamma_scale_geometry
    )
    model_random_state = int(
        stream_rng(seed, STREAM_MODELS, embedding, n).integers(2**31 - 1)
    )

    geometry_results = {}
    if (
        "dimension_geometry" in components
        or "engineered_pqk" in components
        or "engineered_qk" in components
    ):
        geometry_results["projected_quantum_kernel"] = (
            geometry.minimum_geometric_difference(
                pqk_geometry,
                classical.paper_spectra,
                lambdas=settings.lambdas,
                g_tra_threshold=settings.g_tra_threshold,
            )
        )
        if bank.quantum_kernel is not None:
            geometry_results["quantum_kernel"] = geometry.minimum_geometric_difference(
                bank.quantum_kernel,
                classical.paper_spectra,
                lambdas=settings.lambdas,
                g_tra_threshold=settings.g_tra_threshold,
            )
    if "dimension_geometry" in components:
        LOGGER.info(
            "EVALUATION_STARTED | task=dimension_geometry/%s | n=%d", embedding, n
        )
        record["dimension_geometry"] = {
            "d": {
                "projected_quantum_kernel": geometry.effective_dimension(pqk_geometry)
            },
            "g": {
                name: _geometry_record(result)
                for name, result in geometry_results.items()
            },
        }
        if bank.quantum_kernel is not None:
            record["dimension_geometry"]["d"]["quantum_kernel"] = (
                geometry.effective_dimension(bank.quantum_kernel)
            )
        if settings.include_polynomial_extension:
            record["dimension_geometry"]["g_with_extension"] = {
                "projected_quantum_kernel": _geometry_record(
                    geometry.minimum_geometric_difference(
                        pqk_geometry,
                        classical.extension_spectra,
                        lambdas=settings.lambdas,
                        g_tra_threshold=settings.g_tra_threshold,
                    )
                )
            }
            if bank.quantum_kernel is not None:
                record["dimension_geometry"]["g_with_extension"]["quantum_kernel"] = (
                    _geometry_record(
                        geometry.minimum_geometric_difference(
                            bank.quantum_kernel,
                            classical.extension_spectra,
                            lambdas=settings.lambdas,
                            g_tra_threshold=settings.g_tra_threshold,
                        )
                    )
                )
        LOGGER.info(
            "EVALUATION_COMPLETED | task=dimension_geometry/%s | n=%d | d=%s | g=%s",
            embedding,
            n,
            json.dumps(
                {k: round(v, 4) for k, v in record["dimension_geometry"]["d"].items()}
            ),
            json.dumps(
                {
                    k: round(v["g"], 4)
                    for k, v in record["dimension_geometry"]["g"].items()
                }
            ),
        )

    quantum_inputs, pqk_keys = pqk_model_kernels(
        bank.pauli_features, n, settings, "pqk"
    )
    quantum_key_map = {"projected_quantum_kernel": pqk_keys}
    if bank.quantum_kernel is not None:
        quantum_inputs["qk"] = bank.quantum_kernel
        quantum_key_map["quantum_kernel"] = [("qk", {})]
    inputs = {**classical.model_inputs, **quantum_inputs}

    def candidates_for(kind: str) -> list[models.Candidate]:
        candidates = classical_candidates(kind, classical, settings, model_random_state)
        for family, keys in quantum_key_map.items():
            candidates += models.kernel_family_candidates(
                family,
                kind,
                keys,
                grid_name=settings.grid,
                svr_epsilon=settings.svr_epsilon,
            )
        return candidates

    quantum_families = tuple(quantum_key_map)
    positions = np.arange(features.shape[0])

    if "quantum_dataset" in components:
        train_size = settings.quantum_dataset_train_size
        task = models.Task(
            "regression",
            bank.qnn_labels,
            positions[:train_size],
            positions[train_size:],
            inputs,
        )
        record["quantum_dataset"] = evaluate_task(
            task,
            candidates_for("regression"),
            settings,
            n,
            model_random_state,
            f"quantum_dataset/{embedding}",
            quantum_families,
        )
        record["quantum_dataset"]["label_std"] = float(np.std(bank.qnn_labels))

    if "bound" in components and n <= settings.bound_max_system_size:
        train = positions[: settings.quantum_dataset_train_size]
        train_kernels = {
            name: k[np.ix_(train, train)]
            for name, k in classical.geometry_kernels.items()
            if not name.startswith("poly_")
        }
        record["bound"] = geometry.minimum_prediction_error_bound(
            train_kernels, bank.qnn_labels[train], settings.lambdas
        )
        record["bound"]["train_size"] = int(train.size)

    for component, quantum_name, stream in (
        ("engineered_pqk", "projected_quantum_kernel", STREAM_ENGINEERED_PQK),
        ("engineered_qk", "quantum_kernel", STREAM_ENGINEERED_QK),
    ):
        if component not in components:
            continue
        if quantum_name not in geometry_results:
            record[component] = {
                "skipped": "quantum kernel above quantum_kernel_max_qubits"
            }
            continue
        reference = geometry_results[quantum_name]
        quantum_matrix = (
            pqk_geometry
            if quantum_name == "projected_quantum_kernel"
            else bank.quantum_kernel
        )
        constructed = geometry.engineered_labels(
            quantum_matrix,
            classical.geometry_kernels[reference.kernel],
            reference.lam,
            stream_rng(seed, stream, embedding, n),
            random_label_probability=settings.random_label_probability,
            threshold=settings.engineered_label_threshold,
        )
        component_record = {
            "construction": {
                "classical_kernel": reference.kernel,
                "lambda": reference.lam,
                "g_tra_threshold_met": reference.threshold_met,
                "g_gen": constructed.g_gen,
                "random_label_fraction_changed": constructed.flipped_fraction,
                "positive_fraction": float(np.mean(constructed.labels == 1)),
            },
            "by_train_size": {},
        }
        for train_size in settings.engineered_train_sizes:
            task = models.Task(
                "classification",
                constructed.labels,
                positions[:train_size],
                positions[train_size:],
                inputs,
            )
            component_record["by_train_size"][str(train_size)] = evaluate_task(
                task,
                candidates_for("classification"),
                settings,
                n,
                model_random_state,
                f"{component}/{embedding}/N={train_size}",
                quantum_families,
            )
        record[component] = component_record
    record["elapsed_seconds"] = time.perf_counter() - started
    return record


def run_classical_dataset_unit(
    pca_features: PCAFeatures,
    indices: np.ndarray,
    n: int,
    seed: int,
    settings: StudySettings,
    conventions: embeddings.EmbeddingConventions,
    complex_dtype,
) -> dict:
    """Dataset (C): dress vs shirt, every embedding's QK and PQK plus classical models."""
    started = time.perf_counter()
    features = pca_features.features(n, indices)
    labels = np.where(
        pca_features.labels[indices] == settings.classical_dataset_classes[0], 1, -1
    )
    classical = build_classical_inputs(features, n, settings)
    model_random_state = int(
        stream_rng(seed, STREAM_MODELS, None, n).integers(2**31 - 1)
    )
    inputs = dict(classical.model_inputs)
    candidates = classical_candidates(
        "classification", classical, settings, model_random_state
    )
    quantum_families = []
    for embedding in settings.embeddings:
        bank = build_quantum_bank(
            embedding,
            features,
            n,
            seed,
            settings,
            conventions,
            complex_dtype,
            want_kernel=True,
            want_labels=False,
        )
        pqk_inputs, pqk_keys = pqk_model_kernels(
            bank.pauli_features, n, settings, f"pqk_{embedding}"
        )
        inputs.update(pqk_inputs)
        family = f"projected_quantum_kernel_{embedding}"
        candidates += models.kernel_family_candidates(
            family,
            "classification",
            pqk_keys,
            grid_name=settings.grid,
            svr_epsilon=settings.svr_epsilon,
        )
        quantum_families.append(family)
        if bank.quantum_kernel is not None:
            inputs[f"qk_{embedding}"] = bank.quantum_kernel
            family = f"quantum_kernel_{embedding}"
            candidates += models.kernel_family_candidates(
                family,
                "classification",
                [(f"qk_{embedding}", {})],
                grid_name=settings.grid,
                svr_epsilon=settings.svr_epsilon,
            )
            quantum_families.append(family)
    positions = np.arange(features.shape[0])
    train_size = settings.classical_dataset_train_size
    task = models.Task(
        "classification", labels, positions[:train_size], positions[train_size:], inputs
    )
    record = {
        "seed": seed,
        "n": n,
        "embedding": None,
        "component": "classical_dataset",
        "classical_dataset": evaluate_task(
            task,
            candidates,
            settings,
            n,
            model_random_state,
            "classical_dataset",
            tuple(quantum_families),
        ),
    }
    record["elapsed_seconds"] = time.perf_counter() - started
    return record


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record per line (crash-safe partial results)."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_study(
    pca_features: PCAFeatures,
    settings: StudySettings,
    conventions: embeddings.EmbeddingConventions,
    complex_dtype,
    records_path: Path,
    on_unit_completed,
) -> list[dict]:
    """Run every (seed, n) unit and return all records.

    Parameters
    ----------
    pca_features : PCAFeatures
        Standardized PCA features of all images.
    settings : StudySettings
        Study settings.
    conventions : EmbeddingConventions
        Circuit conventions.
    complex_dtype : torch.dtype
        Simulation dtype.
    records_path : pathlib.Path
        JSONL file receiving each record as soon as it is computed.
    on_unit_completed : callable
        Called with ``(completed_units, total_units, record)`` after each record.
    """
    components = set(settings.components)
    per_embedding = bool(components - {"classical_dataset"})
    total_units = (
        len(settings.seeds)
        * len(settings.system_sizes)
        * (
            (len(settings.embeddings) if per_embedding else 0)
            + (1 if "classical_dataset" in components else 0)
        )
    )
    records, completed = [], 0
    for seed in settings.seeds:
        all_indices = subsample_indices(
            pca_features.labels,
            settings.subsample_size,
            stream_rng(seed, STREAM_SUBSAMPLE_ALL),
        )
        classical_indices = subsample_indices(
            pca_features.labels,
            settings.subsample_size,
            stream_rng(seed, STREAM_SUBSAMPLE_CLASSICAL),
            classes=list(settings.classical_dataset_classes),
        )
        for n in settings.system_sizes:
            if per_embedding:
                features = pca_features.features(n, all_indices)
                classical = build_classical_inputs(features, n, settings)
                for embedding in settings.embeddings:
                    record = run_embedding_unit(
                        embedding,
                        features,
                        classical,
                        n,
                        seed,
                        settings,
                        conventions,
                        complex_dtype,
                    )
                    record["subsample_indices_sha"] = _indices_digest(all_indices)
                    records.append(record)
                    append_jsonl(records_path, record)
                    completed += 1
                    on_unit_completed(completed, total_units, record)
            if "classical_dataset" in components:
                record = run_classical_dataset_unit(
                    pca_features,
                    classical_indices,
                    n,
                    seed,
                    settings,
                    conventions,
                    complex_dtype,
                )
                record["subsample_indices_sha"] = _indices_digest(classical_indices)
                records.append(record)
                append_jsonl(records_path, record)
                completed += 1
                on_unit_completed(completed, total_units, record)
    return records


def _indices_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[
        :16
    ]


# --------------------------------------------------------------------------- Fig. S2


def run_training_size_study(
    cfg: dict,
    pca_features: PCAFeatures,
    settings: StudySettings,
    conventions: embeddings.EmbeddingConventions,
    complex_dtype,
    records_path: Path,
    on_unit_completed,
) -> list[dict]:
    """Prediction error vs training size on Dataset (Q, E) (arXiv Fig. S2).

    The subsample holds ``max(train_sizes) + test_size`` points; every training
    set is a prefix of the same shuffled subsample (nested), and the test set is
    the same ``test_size`` points for every training size (AMB-5).
    """
    size_cfg = cfg["training_size"]
    train_sizes = tuple(int(v) for v in size_cfg["train_sizes"])
    test_size = int(size_cfg["test_size"])
    embedding = str(size_cfg["embedding"])
    total_units = len(settings.seeds) * len(settings.system_sizes)
    records, completed = [], 0
    for seed in settings.seeds:
        indices = subsample_indices(
            pca_features.labels,
            max(train_sizes) + test_size,
            stream_rng(seed, STREAM_SUBSAMPLE_ALL),
        )
        for n in settings.system_sizes:
            started = time.perf_counter()
            features = pca_features.features(n, indices)
            n_qubits = register_size(embedding, n, conventions)
            bank = build_quantum_bank(
                embedding,
                features,
                n,
                seed,
                settings,
                conventions,
                complex_dtype,
                want_kernel=True,
                want_labels=True,
            )
            classical = build_classical_inputs(features, n, settings)
            quantum_inputs, pqk_keys = pqk_model_kernels(
                bank.pauli_features, n, settings, "pqk"
            )
            key_map = {"projected_quantum_kernel": pqk_keys}
            if bank.quantum_kernel is not None:
                quantum_inputs["qk"] = bank.quantum_kernel
                key_map["quantum_kernel"] = [("qk", {})]
            inputs = {**classical.model_inputs, **quantum_inputs}
            model_random_state = int(
                stream_rng(seed, STREAM_MODELS, embedding, n).integers(2**31 - 1)
            )
            candidates = classical_candidates(
                "regression", classical, settings, model_random_state
            )
            for family, keys in key_map.items():
                candidates += models.kernel_family_candidates(
                    family,
                    "regression",
                    keys,
                    grid_name=settings.grid,
                    svr_epsilon=settings.svr_epsilon,
                )
            test_positions = np.arange(max(train_sizes), max(train_sizes) + test_size)
            record = {
                "seed": seed,
                "n": n,
                "embedding": embedding,
                "n_qubits": n_qubits,
                "component": "training_size",
                "by_train_size": {},
            }
            for train_size in train_sizes:
                task = models.Task(
                    "regression",
                    bank.qnn_labels,
                    np.arange(train_size),
                    test_positions,
                    inputs,
                )
                record["by_train_size"][str(train_size)] = evaluate_task(
                    task,
                    candidates,
                    settings,
                    n,
                    model_random_state,
                    f"training_size/{embedding}/N={train_size}",
                    tuple(key_map),
                )
            record["elapsed_seconds"] = time.perf_counter() - started
            records.append(record)
            append_jsonl(records_path, record)
            completed += 1
            on_unit_completed(completed, total_units, record)
    return records


__all__ = [
    "COMPONENTS",
    "StudySettings",
    "build_classical_inputs",
    "run_classical_dataset_unit",
    "run_embedding_unit",
    "run_study",
    "run_training_size_study",
    "stream_rng",
]
