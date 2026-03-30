"""Runtime entrypoints for the Nearest Centroid Classification reproduction."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torchvision
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

from runtime_lib.data_paths import paper_data_dir

from .classifier import MLQuantumNearestCentroid, QuantumNearestCentroid
from .synthetic_data import generate_synthetic_data
from .visualization import (
    create_label_comparison_table,
    plot_combined_figure,
    plot_confusion_matrices_comparison,
)

logger = logging.getLogger(__name__)


DATASETS: dict[str, dict[str, Any]] = {
    "mnist": {
        "description": "MNIST handwritten digits dataset (10 classes, 28x28).",
        "n_classes": 10,
    },
    "iris": {
        "description": "Iris flower dataset (3 classes, 4 features).",
        "n_classes": 3,
    },
    "synthetic": {
        "description": "Synthetic clustered data as described in paper Section III.B.a",
        "n_classes": 4,
    },
}


def compute_classical_distances(
    X_test: np.ndarray, centroids: np.ndarray
) -> list[float]:
    """Compute classical Euclidean distances for comparison."""
    distances: list[float] = []
    for x in X_test:
        for c in centroids:
            distances.append(float(np.linalg.norm(x - c)))
    return distances


def run_subset_experiment(
    X: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    max_samples: int | None = None,
    test_size: float = 0.5,
    n_repeats: int = 10,
    n_components: int = 8,
    n_shots: int = 1000,
    run_dir: Path | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """
    Run repeated classification experiments with PCA & classical/quantum classifiers.

    Both quantum classifiers (Cirq and MerLin) run as ideal simulators.
    They should produce identical results within shot noise variance.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    classes : Sequence[int]
        Which classes to include in this experiment
    max_samples : int, optional
        Maximum total samples to use. If None, use all available data.
    test_size : float
        Fraction of data to use for testing (default 0.5)
    n_repeats : int
        Number of times to repeat the experiment
    n_components : int
        Number of PCA components (= number of qubits)
    n_shots : int
        Number of quantum circuit repetitions
    run_dir : Path, optional
        Directory to save results
    collect_predictions : bool
        Whether to collect detailed predictions for the last repeat

    Returns
    -------
    Dict with accuracy statistics and optionally predictions
    """
    accs: list[float] = []
    ml_accs: list[float] = []
    c_accs: list[float] = []

    all_y_true: list[int] = []
    all_y_pred_classical: list[int] = []
    all_y_pred_cirq: list[int] = []
    all_y_pred_merlin: list[int] = []

    dist_ratios_cirq: list[float] = []
    dist_ratios_merlin: list[float] = []

    # Filter to selected classes
    mask = np.isin(y, classes)
    X_filtered, y_filtered = X[mask], y[mask]

    n_available = len(X_filtered)

    # Determine actual sample count
    if max_samples is not None and max_samples < n_available:
        n_to_use = max_samples
    else:
        n_to_use = n_available

    n_test = int(n_to_use * test_size)
    n_train = n_to_use - n_test

    logger.info(
        f"Experiment: {len(classes)} classes, {n_available} available, "
        f"using {n_to_use} ({n_train} train / {n_test} test), "
        f"{n_shots} shots, {n_components} components"
    )

    for r in range(n_repeats):
        # Subsample if max_samples is set
        if max_samples is not None and max_samples < n_available:
            X_sub, _, y_sub, _ = train_test_split(
                X_filtered,
                y_filtered,
                train_size=max_samples,
                stratify=y_filtered,
                random_state=r,
            )
        else:
            X_sub, y_sub = X_filtered, y_filtered

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=test_size, stratify=y_sub, random_state=r + 1000
        )

        # Apply PCA if we have more features than components
        if X_train.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
        else:
            X_train_pca = X_train
            X_test_pca = X_test

        # Scale to [0, 1] for quantum circuits
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        # Classical classifier
        c = NearestCentroid()
        c.fit(X_train_pca, y_train)

        # Quantum classifiers - both as IDEAL simulators (no noise, no error mitigation)
        clf_q = QuantumNearestCentroid(repetitions=n_shots)
        clf_ml = MLQuantumNearestCentroid(n=n_components, repetitions=n_shots)

        clf_q.fit(X_train_scaled, y_train)
        clf_ml.fit(X_train_scaled, y_train)

        # Evaluate
        c_acc = float(c.score(X_test_pca, y_test))
        acc_q = float(clf_q.score(X_test_scaled, y_test))
        acc_ml = float(clf_ml.score(X_test_scaled, y_test))

        c_accs.append(c_acc)
        accs.append(acc_q)
        ml_accs.append(acc_ml)

        if collect_predictions and r == n_repeats - 1:
            all_y_true = y_test.tolist()
            all_y_pred_classical = c.predict(X_test_pca).tolist()
            all_y_pred_cirq = clf_q.predict(X_test_scaled).tolist()
            all_y_pred_merlin = clf_ml.predict(X_test_scaled).tolist()

    result: dict[str, Any] = {
        "classes": list(classes),
        "n_available": n_available,
        "max_samples": max_samples,
        "n_used": n_to_use,
        "n_train": n_train,
        "n_test": n_test,
        "n_components": n_components,
        "n_shots": n_shots,
        "n_repeats": n_repeats,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "ml_acc_mean": float(np.mean(ml_accs)),
        "ml_acc_std": float(np.std(ml_accs)),
        "c_acc_mean": float(np.mean(c_accs)),
        "c_acc_std": float(np.std(c_accs)),
        "dist_ratios_cirq": dist_ratios_cirq if dist_ratios_cirq else [1.0],
        "dist_ratios_merlin": dist_ratios_merlin if dist_ratios_merlin else [1.0],
    }

    if collect_predictions:
        result["predictions"] = {
            "y_true": all_y_true,
            "y_pred_classical": all_y_pred_classical,
            "y_pred_cirq": all_y_pred_cirq,
            "y_pred_merlin": all_y_pred_merlin,
        }

    if run_dir is not None:
        out_file = run_dir / f"results_{'_'.join(map(str, classes))}.json"
        out_file.write_text(json.dumps(result, indent=2))

    return result


def _load_dataset(
    dataset_name: str, data_root: str | Path | None, cfg: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset by name, using runtime_lib for path resolution."""
    name = dataset_name.lower()

    # Use shared runtime data path resolution
    resolved_root = paper_data_dir("nearest_centroids_merlin", data_root)

    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
        mnist = torchvision.datasets.MNIST(
            root=str(resolved_root), train=True, download=True, transform=transform
        )
        X = mnist.data.numpy().reshape(len(mnist), -1)
        y = mnist.targets.numpy()
        return X, y

    if name == "iris":
        iris = load_iris()
        return iris.data, iris.target

    if name == "synthetic":
        dataset_cfg = cfg.get("dataset", {}) if cfg else {}
        X, y, _ = generate_synthetic_data(
            n_clusters=dataset_cfg.get("n_clusters", 4),
            n_dimensions=dataset_cfg.get("n_components", 8),
            n_points_per_cluster=dataset_cfg.get("n_points_per_cluster", 10),
            min_centroid_distance=dataset_cfg.get("min_centroid_distance", 0.3),
            gaussian_variance=dataset_cfg.get("gaussian_variance", 0.05),
            sphere_radius=dataset_cfg.get("sphere_radius", 1.0),
            seed=cfg.get("seed", 123) if cfg else 123,
        )
        return X, y

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _run_experiments(cfg: dict[str, Any], run_dir: Path) -> None:
    """Core experiment loop - run all configured experiments."""
    logger.info("Starting experiment with dataset: %s", cfg["dataset"]["name"])

    dataset_name = cfg["dataset"]["name"].lower()
    n_repeats = cfg["training"]["n_repeats"]
    n_components = cfg["dataset"]["n_components"]
    n_shots = cfg["training"].get("n_shots", 1000)
    test_size = cfg["training"].get("test_size", 0.5)
    default_max_samples = cfg["training"].get("max_samples", None)
    experiments: list[dict[str, Any]] = list(cfg.get("experiments", []))

    if not experiments:
        raise ValueError("No experiments specified in config")

    data_root = cfg.get("dataset", {}).get("root", None)
    X, y = _load_dataset(dataset_name, data_root, cfg)

    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")

    all_results: list[dict[str, Any]] = []

    for i, exp in enumerate(experiments):
        collect_predictions = i == len(experiments) - 1

        # Allow per-experiment overrides
        exp_n_shots = exp.get("n_shots", n_shots)
        exp_test_size = exp.get("test_size", test_size)
        exp_max_samples = exp.get("max_samples", default_max_samples)

        result = run_subset_experiment(
            X=X,
            y=y,
            classes=exp["classes"],
            max_samples=exp_max_samples,
            test_size=exp_test_size,
            n_repeats=n_repeats,
            n_components=n_components,
            n_shots=exp_n_shots,
            run_dir=run_dir,
            collect_predictions=collect_predictions,
        )
        all_results.append(result)
        logger.info(
            "Finished experiment %d/%d: classes=%s -> Classical: %.1f%%, Cirq: %.1f%%, Merlin: %.1f%%",
            i + 1,
            len(experiments),
            exp["classes"],
            result["c_acc_mean"] * 100,
            result["acc_mean"] * 100,
            result["ml_acc_mean"] * 100,
        )

    results_path = run_dir / "summary_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    logger.info("All experiments completed. Results saved to %s", results_path)

    _generate_visualizations(all_results, dataset_name, n_components, run_dir)


def _generate_visualizations(
    all_results: list[dict[str, Any]],
    dataset_name: str,
    n_components: int,
    run_dir: Path,
) -> None:
    """Generate all visualization figures."""
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info("Generating visualisations...")

    formatted_results: list[dict[str, Any]] = []
    for r in all_results:
        n_classes = len(r["classes"])
        n_shots = r.get("n_shots", 1000)
        n_used = r.get("n_used", r.get("n_train", 0) + r.get("n_test", 0))
        formatted_results.append(
            {
                "label": f"Nq={n_components}\nNc={n_classes}\nNs={n_shots}\nn={n_used}",
                "c_acc_mean": r["c_acc_mean"],
                "c_acc_std": r["c_acc_std"],
                "acc_mean": r["acc_mean"],
                "acc_std": r["acc_std"],
                "ml_acc_mean": r["ml_acc_mean"],
                "ml_acc_std": r["ml_acc_std"],
                "dist_ratios_cirq": r.get("dist_ratios_cirq", [1.0]),
                "dist_ratios_merlin": r.get("dist_ratios_merlin", [1.0]),
            }
        )

    try:
        plot_combined_figure(
            formatted_results,
            f"{dataset_name.upper()} Dataset Results",
            save_path=figures_dir / f"{dataset_name}_combined.png",
        )
    except Exception as exc:
        logger.warning("Failed to generate combined figure: %s", exc)

    for r in all_results:
        if "predictions" not in r:
            continue
        preds = r["predictions"]
        classes = r["classes"]
        try:
            plot_confusion_matrices_comparison(
                y_true=preds["y_true"],
                y_pred_classical=preds["y_pred_classical"],
                y_pred_quantum=preds["y_pred_merlin"],
                classes=classes,
                save_path=figures_dir
                / f"confusion_matrix_classes_{'_'.join(map(str, classes))}.png",
            )
        except Exception as exc:
            logger.warning("Failed to generate confusion matrix: %s", exc)

        try:
            create_label_comparison_table(
                sampling_labels=preds["y_true"],
                classical_labels=preds["y_pred_classical"],
                quantum_no_mit_labels=preds["y_pred_cirq"],
                quantum_mit_labels=preds["y_pred_merlin"],
                save_path=figures_dir
                / f"label_table_classes_{'_'.join(map(str, classes))}.txt",
            )
        except Exception as exc:
            logger.warning("Failed to generate label comparison table: %s", exc)

    logger.info("Figures saved to: %s", figures_dir)


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """
    Entry point required by the MerLin shared runtime.

    The shared runtime handles:
    - Seed setting
    - Run directory creation
    - Logging configuration
    - Config snapshot saving

    This function should focus only on the experiment logic.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (already loaded and merged with defaults)
    run_dir : Path
        Directory for saving results (already created by shared runtime)
    """
    run_dir = Path(run_dir)
    _run_experiments(cfg, run_dir)
    (run_dir / "done.txt").write_text("Completed")
    logger.info("Finished. Artifacts in: %s", run_dir)


__all__ = ["train_and_evaluate", "run_subset_experiment", "DATASETS"]
