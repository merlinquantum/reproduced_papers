#!/usr/bin/env python3
"""Run a small MerLin photonic fidelity-kernel experiment on CPU.

This is a photonic surrogate, not a reproduction of the paper's qubit BSP
feature map. It deliberately reuses the historical data split and preprocessing
used by the local classical baselines so that only the kernel changes.
"""

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import merlin as ML
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qve.core import normalize_train_and_cross_kernel_trace  # noqa: E402
from qve.process import data_prepare_cv  # noqa: E402
from scripts.classical_svm_c1_pca import evaluate, load_data, split_data  # noqa: E402


def compute_kernel(kernel, left, right=None):
    """Compute one MerLin Gram or cross-kernel matrix on CPU."""
    left_tensor = torch.as_tensor(left, dtype=torch.float32)
    right_tensor = (
        None if right is None else torch.as_tensor(right, dtype=torch.float32)
    )
    started = time.perf_counter()
    with torch.no_grad():
        matrix = kernel(left_tensor, right_tensor)
    elapsed = time.perf_counter() - started
    return matrix.detach().cpu().numpy(), elapsed


def validate_kernel(matrix, expected_shape, name, square=False):
    """Validate and summarize a fidelity-kernel matrix."""
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} has shape {matrix.shape}, expected {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")

    tolerance = 2e-4
    minimum = float(matrix.min())
    maximum = float(matrix.max())
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{name} values must be in [0, 1] within tolerance; "
            f"observed [{minimum}, {maximum}]"
        )

    summary = {
        "shape": list(matrix.shape),
        "minimum": minimum,
        "maximum": maximum,
        "tolerance": tolerance,
    }
    if square:
        symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
        diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
        minimum_eigenvalue = float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min())
        if symmetry_error > tolerance:
            raise ValueError(f"{name} is not symmetric: error={symmetry_error}")
        if diagonal_error > tolerance:
            raise ValueError(f"{name} diagonal differs from 1: error={diagonal_error}")
        if minimum_eigenvalue < -tolerance:
            raise ValueError(
                f"{name} is not positive semidefinite: "
                f"minimum eigenvalue={minimum_eigenvalue}"
            )
        summary.update(
            {
                "symmetry_error": symmetry_error,
                "diagonal_error": diagonal_error,
                "minimum_eigenvalue": minimum_eigenvalue,
            }
        )
    return summary


def normalize_kernel_pairs(
    train_validation,
    validation_cross,
    train_test,
    test_cross,
    normalization,
):
    """Normalize two matching train/cross kernel pairs.

    Parameters
    ----------
    train_validation, train_test : numpy.ndarray
        Square training kernels used for validation and test evaluation.
    validation_cross, test_cross : numpy.ndarray
        Associated rectangular cross-kernels.
    normalization : {"none", "train_trace"}
        Normalization protocol.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Kernels in the same order, normalized when requested.

    Raises
    ------
    ValueError
        If ``normalization`` is unsupported or a training trace is invalid.
    """
    if normalization == "none":
        return train_validation, validation_cross, train_test, test_cross
    if normalization != "train_trace":
        raise ValueError(f"Unsupported kernel normalization: {normalization}")

    train_validation, validation_cross = normalize_train_and_cross_kernel_trace(
        train_validation, validation_cross
    )
    train_test, test_cross = normalize_train_and_cross_kernel_trace(
        train_test, test_cross
    )
    return train_validation, validation_cross, train_test, test_cross


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("../../data/qsvm_medimage/pneumoniamnist_train.pkl"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pca_dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--circuit_seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--fix_leakage",
        action="store_true",
        help="Fit MinMaxScaler on train only (default: legacy train+held-out fit)",
    )
    parser.add_argument(
        "--kernel_normalization",
        choices=["none", "train_trace"],
        default="none",
        help="Kernel normalization protocol (default: none)",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.pca_dim <= 19:
        raise ValueError("MerLin FeatureMap.simple requires 1 <= pca_dim <= 19")

    features, labels, minority_label, class_names = load_data(args.data_path)
    train_x, validation_x, test_x, train_y, validation_y, test_y = split_data(
        features,
        labels,
        args.seed,
        args.max_samples,
    )

    train_for_validation, validation_pca = data_prepare_cv(
        args.pca_dim,
        train_x,
        validation_x,
        fix_leakage=args.fix_leakage,
        svd_solver="full",
    )
    train_for_test, test_pca = data_prepare_cv(
        args.pca_dim,
        train_x,
        test_x,
        fix_leakage=args.fix_leakage,
        svd_solver="full",
    )

    preprocessing_protocol = (
        "train_only" if args.fix_leakage else "legacy_train_plus_heldout"
    )
    minmax_fit = "train" if args.fix_leakage else "train+heldout"
    preprocessing_description = (
        f"StandardScaler(train) -> PCA(train) -> MinMaxScaler({minmax_fit})"
    )

    device = torch.device("cpu")
    torch.manual_seed(args.circuit_seed)
    feature_map = ML.FeatureMap.simple(
        input_size=args.pca_dim,
        dtype=torch.float32,
        device=device,
    )
    n_modes = args.pca_dim + 1
    input_state = [1 if mode % 2 == 0 else 0 for mode in range(n_modes)]
    kernel = ML.FidelityKernel(
        feature_map=feature_map,
        input_state=input_state,
        shots=None,
        force_psd=True,
        dtype=torch.float32,
        device=device,
    )
    kernel.requires_grad_(False)

    total_started = time.perf_counter()
    print("Computing MerLin train kernel for validation...")
    kernel_train_validation, train_validation_time = compute_kernel(
        kernel, train_for_validation
    )
    print("Computing MerLin validation cross-kernel...")
    kernel_validation, validation_time = compute_kernel(
        kernel, validation_pca, train_for_validation
    )
    print("Computing MerLin train kernel for test...")
    kernel_train_test, train_test_time = compute_kernel(kernel, train_for_test)
    print("Computing MerLin test cross-kernel...")
    kernel_test, test_time = compute_kernel(kernel, test_pca, train_for_test)

    checks = {
        "train_validation": validate_kernel(
            kernel_train_validation,
            (len(train_y), len(train_y)),
            "train-validation kernel",
            square=True,
        ),
        "validation": validate_kernel(
            kernel_validation,
            (len(validation_y), len(train_y)),
            "validation cross-kernel",
        ),
        "train_test": validate_kernel(
            kernel_train_test,
            (len(train_y), len(train_y)),
            "train-test kernel",
            square=True,
        ),
        "test": validate_kernel(
            kernel_test,
            (len(test_y), len(train_y)),
            "test cross-kernel",
        ),
    }

    (
        kernel_train_validation,
        kernel_validation,
        kernel_train_test,
        kernel_test,
    ) = normalize_kernel_pairs(
        kernel_train_validation,
        kernel_validation,
        kernel_train_test,
        kernel_test,
        args.kernel_normalization,
    )

    model_started = time.perf_counter()
    model = SVC(
        kernel="precomputed",
        C=1.0,
        probability=True,
        random_state=args.seed,
    )
    model.fit(kernel_train_test, train_y)
    train_time = time.perf_counter() - model_started

    validation_model = SVC(
        kernel="precomputed",
        C=1.0,
        probability=True,
        random_state=args.seed,
    )
    validation_model.fit(kernel_train_validation, train_y)

    row = {
        "seed": args.seed,
        "circuit_seed": args.circuit_seed,
        "pca_dim": args.pca_dim,
        "kernel": "merlin_fidelity",
        "best_c": 1.0,
        "c_values": "1.0",
        "n_modes": n_modes,
        "n_photons": sum(input_state),
        "input_state": ",".join(str(value) for value in input_state),
        "train_samples": len(train_y),
        "val_samples": len(validation_y),
        "test_samples": len(test_y),
        "fix_leakage": args.fix_leakage,
        "preprocessing_protocol": preprocessing_protocol,
        "kernel_normalization": args.kernel_normalization,
        "kernel_time_sec": (
            train_validation_time + validation_time + train_test_time + test_time
        ),
        "train_time_sec": train_time,
    }
    row.update(evaluate(model, kernel_train_test, train_y, "train", minority_label))
    row.update(evaluate(model, kernel_test, test_y, "test", minority_label))
    row.update(
        evaluate(
            validation_model,
            kernel_validation,
            validation_y,
            "val",
            minority_label,
        )
    )
    row["total_time_sec"] = time.perf_counter() - total_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_summary.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    with (args.output_dir / "dataset_info.json").open("w") as file:
        json.dump(
            {
                "scope": "merlin_photonic_adaptation",
                "data_path": str(args.data_path),
                "class_names": class_names,
                "minority_class": class_names[minority_label],
                "max_samples": args.max_samples,
                "seed": args.seed,
                "circuit_seed": args.circuit_seed,
                "pca_dim": args.pca_dim,
                "n_modes": n_modes,
                "input_state": input_state,
                "n_photons": sum(input_state),
                "shots": None,
                "force_psd": True,
                "kernel_normalization": args.kernel_normalization,
                "fix_leakage": args.fix_leakage,
                "preprocessing_protocol": preprocessing_protocol,
                "preprocessing": preprocessing_description,
                "kernel_checks": checks,
                "versions": {
                    "merlinquantum": version("merlinquantum"),
                    "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__,
                    "torch": torch.__version__,
                },
            },
            file,
            indent=2,
        )

    print(
        f"Saved: {metrics_path}\n"
        f"test minority F1: {row['test_minority_f1']:.4f}\n"
        f"kernel time: {row['kernel_time_sec']:.2f} s"
    )


if __name__ == "__main__":
    main()
