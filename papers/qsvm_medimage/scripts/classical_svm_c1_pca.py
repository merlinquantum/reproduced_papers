#!/usr/bin/env python3
"""Run CPU classical SVM baselines with the QSVM preprocessing."""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated in 1\.9",
    category=FutureWarning,
    module=r"sklearn\.svm\._base",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qve.process import data_prepare_cv  # noqa: E402

TARGET_COLUMNS = [
    "target",
    "label",
    "new_insurance_type",
    "insurance",
    "insurance_type",
]


def parse_values(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_data(data_path):
    if data_path.suffix in {".pkl", ".pickle"}:
        dataframe = pd.read_pickle(data_path)
    elif data_path.suffix == ".parquet":
        dataframe = pd.read_parquet(data_path)
    else:
        raise ValueError("Data must be a .pkl, .pickle, or .parquet file")

    target_column = next(
        (column for column in TARGET_COLUMNS if column in dataframe.columns),
        None,
    )
    if target_column is None or "embedding" not in dataframe.columns:
        raise ValueError("Data must contain 'embedding' and a supported target column")

    dataframe = dataframe.dropna(subset=[target_column, "embedding"])
    features = np.stack(
        dataframe["embedding"].map(lambda value: np.asarray(value, dtype=np.float64))
    ).reshape(len(dataframe), -1)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(dataframe[target_column].astype(str))
    if len(np.unique(labels)) != 2:
        raise ValueError("This baseline expects binary labels")
    label_counts = np.bincount(labels)
    minority_label = int(np.argmin(label_counts))
    return features, labels, minority_label, label_encoder.classes_.tolist()


def split_data(features, labels, seed, max_samples):
    if max_samples and max_samples < len(features):
        indices = np.random.RandomState(seed).permutation(len(features))[:max_samples]
        features = features[indices]
        labels = labels[indices]

    train_x, temporary_x, train_y, temporary_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        temporary_x,
        temporary_y,
        test_size=0.5,
        stratify=temporary_y,
        random_state=seed,
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def evaluate(model, features, labels, prefix, minority_label):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        f"{prefix}_accuracy": float(accuracy_score(labels, predictions)),
        f"{prefix}_f1": float(f1_score(labels, predictions, zero_division=0)),
        f"{prefix}_minority_f1": float(
            f1_score(
                labels,
                predictions,
                pos_label=minority_label,
                zero_division=0,
            )
        ),
        f"{prefix}_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(recall_score(labels, predictions, zero_division=0)),
        f"{prefix}_auc": float(roc_auc_score(labels, probabilities)),
    }


def select_c(
    train_x,
    train_y,
    validation_x,
    validation_y,
    kernel,
    c_values,
    seed,
    minority_label,
):
    best_c = c_values[0]
    best_f1 = -1.0
    for c_value in c_values:
        model = SVC(
            kernel=kernel,
            C=c_value,
            probability=True,
            random_state=seed,
        )
        model.fit(train_x, train_y)
        validation_f1 = f1_score(
            validation_y,
            model.predict(validation_x),
            pos_label=minority_label,
            zero_division=0,
        )
        if validation_f1 > best_f1:
            best_c = c_value
            best_f1 = validation_f1
    return best_c


def run_configuration(
    features,
    labels,
    seed,
    pca_dim,
    kernel,
    c_values,
    max_samples,
    minority_label,
    fix_leakage=False,
):
    train_x, validation_x, test_x, train_y, validation_y, test_y = split_data(
        features,
        labels,
        seed,
        max_samples,
    )
    train_for_validation, validation_pca = data_prepare_cv(
        pca_dim,
        train_x,
        validation_x,
        fix_leakage=fix_leakage,
        svd_solver="full",
    )
    train_for_test, test_pca = data_prepare_cv(
        pca_dim,
        train_x,
        test_x,
        fix_leakage=fix_leakage,
        svd_solver="full",
    )

    preprocessing_protocol = (
        "train_only" if fix_leakage else "legacy_train_plus_heldout"
    )
    best_c = select_c(
        train_for_validation,
        train_y,
        validation_pca,
        validation_y,
        kernel,
        c_values,
        seed,
        minority_label,
    )

    started = time.time()
    model = SVC(
        kernel=kernel,
        C=best_c,
        probability=True,
        random_state=seed,
    )
    model.fit(train_for_test, train_y)
    train_time = time.time() - started

    row = {
        "seed": seed,
        "pca_dim": pca_dim,
        "kernel": kernel,
        "best_c": best_c,
        "c_values": ",".join(str(value) for value in c_values),
        "train_samples": len(train_y),
        "val_samples": len(validation_y),
        "test_samples": len(test_y),
        "train_time_sec": train_time,
        "fix_leakage": fix_leakage,
        "preprocessing_protocol": preprocessing_protocol,
    }
    row.update(evaluate(model, train_for_test, train_y, "train", minority_label))
    row.update(evaluate(model, test_pca, test_y, "test", minority_label))

    validation_model = SVC(
        kernel=kernel,
        C=best_c,
        probability=True,
        random_state=seed,
    )
    validation_model.fit(train_for_validation, train_y)
    row.update(
        evaluate(
            validation_model,
            validation_pca,
            validation_y,
            "val",
            minority_label,
        )
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pca_dims", default="2")
    parser.add_argument("--kernels", default="linear")
    parser.add_argument("--c_values", default="1.0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--fix_leakage",
        action="store_true",
        help="Fit MinMaxScaler on train only (default: legacy train+held-out fit)",
    )
    args = parser.parse_args()

    pca_dims = parse_values(args.pca_dims, int)
    kernels = parse_values(args.kernels, str)
    c_values = parse_values(args.c_values, float)
    seeds = parse_values(args.seeds, int)
    if not pca_dims or not kernels or not c_values or not seeds:
        raise ValueError("PCA dimensions, kernels, C values, and seeds cannot be empty")

    features, labels, minority_label, class_names = load_data(args.data_path)
    rows = []
    for seed in seeds:
        for pca_dim in pca_dims:
            for kernel in kernels:
                row = run_configuration(
                    features,
                    labels,
                    seed,
                    pca_dim,
                    kernel,
                    c_values,
                    args.max_samples,
                    minority_label,
                    fix_leakage=args.fix_leakage,
                )
                rows.append(row)
                print(
                    f"seed={seed} q={pca_dim} kernel={kernel} "
                    f"C={row['best_c']} "
                    f"minority_f1={row['test_minority_f1']:.4f}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    with (args.output_dir / "dataset_info.json").open("w") as file:
        json.dump(
            {
                "data_path": str(args.data_path.resolve()),
                "features": features.shape[1],
                "classes": int(len(np.unique(labels))),
                "class_names": class_names,
                "minority_class": class_names[minority_label],
                "max_samples": args.max_samples,
                "seeds": seeds,
                "pca_dims": pca_dims,
                "kernels": kernels,
                "c_values": c_values,
                "fix_leakage": args.fix_leakage,
                "preprocessing_protocol": (
                    "train_only" if args.fix_leakage else "legacy_train_plus_heldout"
                ),
            },
            file,
            indent=2,
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
