"""Task 1: next-day direction classification, QNN vs ANN (§III-C, Tables IX-XI).

One run covers one ticker, one feature set and one seed. Every QNN and ANN
candidate is trained on all five walk-forward folds; selection then follows two
predeclared protocols on the same trained models:

- ``leak_free`` (primary, decision R3): highest mean validation AUC over folds
  1-4; fold 5's window (2020-2021) is used only as the test.
- ``paper``: highest mean validation AUC over folds 1-5 as in §III-C3, which
  includes the reported test window in selection (Fig. 4). Reported for
  comparison and labelled test-leaking.

Ties within ``selection.tie_tolerance`` go to the candidate with fewer
parameters, as in the paper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .. import data
from ..metrics import classification_metrics
from ..models import MLPClassifier, QNNClassifier, count_parameters
from ..photonic_merlin import PhotonicQNNClassifier
from ..training import fit_classifier, predict_proba

logger = logging.getLogger(__name__)

QNN_FAMILIES = {
    "SQ": {"hybrid": False, "multi_qubit_readout": False},
    "MQ": {"hybrid": False, "multi_qubit_readout": True},
    "Hybrid-SQ": {"hybrid": True, "multi_qubit_readout": False},
    "Hybrid-MQ": {"hybrid": True, "multi_qubit_readout": True},
}


def qnn_candidates(qnn_cfg: dict, n_features: int) -> list[dict]:
    """QNN grid of §III-C3: families x depths x qubit counts, skipping impossible encodings.

    With ``backend="merlin"`` (Phase 4 adaptation) the grid is families x
    modes x photons x re-uploads of :class:`PhotonicQNNClassifier`, named
    ``pqnn_<family>_m<modes>_n<photons>_L<layers>``.
    """
    backend = qnn_cfg["backend"]
    if backend == "merlin":
        return photonic_qnn_candidates(qnn_cfg, n_features)
    if backend != "statevector":
        raise ValueError(
            f"QNN backend must be 'statevector' or 'merlin', got {backend!r}"
        )
    candidates = []
    encoded_dim = (lambda q: q) if qnn_cfg["encoding"] == "angle" else (lambda q: 2**q)
    for family in qnn_cfg["families"]:
        for n_qubits in qnn_cfg["n_qubits"]:
            if (
                not QNN_FAMILIES[family]["hybrid"]
                and encoded_dim(n_qubits) != n_features
            ):
                continue
            for layers in qnn_cfg["layers"]:
                candidates.append(
                    {
                        "model": "qnn",
                        "family": family,
                        "n_qubits": n_qubits,
                        "layers": layers,
                        "name": f"qnn_{family}_q{n_qubits}_L{layers}",
                    }
                )
    if not candidates:
        raise ValueError(
            f"QNN grid is empty for {n_features} features and config {qnn_cfg}"
        )
    return candidates


def photonic_qnn_candidates(qnn_cfg: dict, n_features: int) -> list[dict]:
    """Photonic QNN grid; non-hybrid families need one mode per feature."""
    candidates = [
        {
            "model": "qnn",
            "family": family,
            "n_modes": n_modes,
            "n_photons": n_photons,
            "layers": layers,
            "name": f"pqnn_{family}_m{n_modes}_n{n_photons}_L{layers}",
        }
        for family in qnn_cfg["families"]
        for n_modes in qnn_cfg["n_modes"]
        if QNN_FAMILIES[family]["hybrid"] or n_modes == n_features
        for n_photons in qnn_cfg["n_photons"]
        for layers in qnn_cfg["layers"]
    ]
    if not candidates:
        raise ValueError(
            f"Photonic QNN grid is empty for {n_features} features and config {qnn_cfg}"
        )
    return candidates


def ann_candidates(ann_cfg: dict) -> list[dict]:
    """ANN width grid; names encode the hidden widths, e.g. ``ann_32-16``."""
    return [
        {
            "model": "ann",
            "hidden": list(widths),
            "name": "ann_" + "-".join(str(width) for width in widths),
        }
        for widths in ann_cfg["hidden_layer_options"]
    ]


def build_model(candidate: dict, n_features: int, models_cfg: dict) -> torch.nn.Module:
    """Instantiate the classifier a candidate describes."""
    if candidate["model"] == "qnn" and "n_modes" in candidate:
        family = QNN_FAMILIES[candidate["family"]]
        return PhotonicQNNClassifier(
            n_features=n_features,
            n_modes=candidate["n_modes"],
            n_photons=candidate["n_photons"],
            layers=candidate["layers"],
            hybrid=family["hybrid"],
            multi_mode_readout=family["multi_qubit_readout"],
            angle_scale=float(models_cfg["qnn"]["angle_scale"]),
        )
    if candidate["model"] == "qnn":
        qnn_cfg = models_cfg["qnn"]
        return QNNClassifier(
            n_features=n_features,
            n_qubits=candidate["n_qubits"],
            layers=candidate["layers"],
            encoding=qnn_cfg["encoding"],
            angle_scale=float(qnn_cfg["angle_scale"]),
            topology=qnn_cfg["topology"],
            **QNN_FAMILIES[candidate["family"]],
        )
    return MLPClassifier(n_features, candidate["hidden"])


def matched_ann_width(parameter_budget: int, n_features: int) -> int:
    """One-hidden-layer width whose parameter count ``w (d + 2) + 1`` is closest to the budget."""
    return max(1, int(round((parameter_budget - 1) / (n_features + 2))))


def select(
    candidate_rows: pd.DataFrame, folds: list[int], tie_tolerance: float
) -> pd.Series:
    """Highest mean validation AUC over ``folds``; ties go to fewer parameters, then grid order."""
    scores = (
        candidate_rows[candidate_rows["fold"].isin(folds)]
        .groupby("name", sort=False)
        .agg(
            mean_auc=("validation_auc", "mean"),
            parameters=("parameters", "first"),
            order=("order", "first"),
        )
    )
    tied = scores[scores["mean_auc"] >= scores["mean_auc"].max() - tie_tolerance]
    return tied.sort_values(["parameters", "order"]).iloc[0]


def run_directional(
    cfg: dict, section: dict, run_dir: Path, status: dict, report_progress
) -> dict:
    """Train every candidate on every fold, select, and evaluate on fold 5.

    Artifacts written to ``run_dir``: ``candidates.csv`` (one row per candidate
    and fold) and ``test_predictions.csv`` (fold-5 probabilities of the reported
    models).

    Returns
    -------
    dict
        Metrics for ``metrics.json``.
    """
    dataset_cfg, models_cfg, training_cfg = (
        section["dataset"],
        section["models"],
        section["training"],
    )
    device = torch.device(cfg["device"])
    dtype = cfg["torch_dtype"]
    seed = int(cfg["seed"])

    frame = data.build_directional_frame(dataset_cfg, cfg["data_root"])
    folds = data.walk_forward_folds(frame.index, dataset_cfg)
    values = frame.drop(columns="label").to_numpy(dtype=np.float64, copy=True)
    labels = frame["label"].to_numpy(copy=True)
    n_features = values.shape[1]
    status["dataset"] = {
        "name": f"yahoo_daily_prices:{dataset_cfg['ticker']}:{dataset_cfg['feature_set']}",
        "splits": {
            f"fold{fold.number}": {
                "train": [
                    str(frame.index[fold.train[0]].date()),
                    str(frame.index[fold.train[-1]].date()),
                ],
                "early_stop": [
                    str(frame.index[fold.early_stop[0]].date()),
                    str(frame.index[fold.early_stop[-1]].date()),
                ],
                "validation": [
                    str(frame.index[fold.validation[0]].date()),
                    str(frame.index[fold.validation[-1]].date()),
                ],
            }
            for fold in folds
        },
        "preprocessing": f"min-max fitted on each fold's training rows; feature_timing={dataset_cfg.get('feature_timing', 'n/a')}",
        "subset": "full",
    }
    logger.info(
        "DATASET_READY | ticker=%s | feature_set=%s | rows=%d | features=%d | first=%s | last=%s | positive_rate=%.4f",
        dataset_cfg["ticker"],
        dataset_cfg["feature_set"],
        len(frame),
        n_features,
        frame.index[0].date(),
        frame.index[-1].date(),
        labels.mean(),
    )

    candidates = qnn_candidates(models_cfg["qnn"], n_features) + ann_candidates(
        models_cfg["ann"]
    )
    fold_tensors = []
    for fold in folds:
        scaler = data.MinMaxScaler().fit(values[fold.train])
        scaled = scaler.transform(values)
        fold_tensors.append(
            {
                "scaled": torch.as_tensor(scaled, dtype=dtype, device=device),
                "raw": torch.as_tensor(values, dtype=dtype, device=device),
            }
        )
    label_tensor = torch.as_tensor(labels, dtype=dtype, device=device)

    def train_on_fold(
        candidate: dict, order: int, fold_index: int
    ) -> tuple[torch.nn.Module, dict]:
        fold = folds[fold_index]
        use_scaled = candidate["model"] == "qnn" or models_cfg["ann"]["minmax_inputs"]
        inputs = fold_tensors[fold_index]["scaled" if use_scaled else "raw"]
        torch.manual_seed(seed * 100_000 + order * 10 + fold.number)
        model = build_model(candidate, n_features, models_cfg).to(
            device=device, dtype=dtype
        )
        fit = fit_classifier(
            model,
            inputs[fold.train],
            label_tensor[fold.train],
            inputs[fold.early_stop],
            label_tensor[fold.early_stop],
            training_cfg,
            seed=seed * 100_000 + order * 10 + fold.number,
            context=f"candidate={candidate['name']} | fold={fold.number}",
        )
        probabilities = predict_proba(model, inputs[fold.validation])
        return model, {**fit, "probabilities": probabilities}

    logger.info(
        "TRAINING_STARTED | candidates=%d | folds=%d | max_epochs=%d | patience=%d | optimizer=adam | lr=%g",
        len(candidates),
        len(folds),
        training_cfg["max_epochs"],
        training_cfg["patience"],
        training_cfg["lr"],
    )
    rows, fold5_probabilities = [], {}
    for order, candidate in enumerate(candidates):
        for fold_index, fold in enumerate(folds):
            model, fit = train_on_fold(candidate, order, fold_index)
            validation_labels = labels[fold.validation]
            metrics = classification_metrics(validation_labels, fit["probabilities"])
            rows.append(
                {
                    "name": candidate["name"],
                    "order": order,
                    "model": candidate["model"],
                    "family": candidate.get("family", "ANN"),
                    "n_qubits": candidate.get("n_qubits"),
                    "layers": candidate.get("layers"),
                    "hidden": "-".join(map(str, candidate.get("hidden", []))),
                    "parameters": count_parameters(model),
                    "fold": fold.number,
                    "best_epoch": fit["best_epoch"],
                    "epochs_run": fit["epochs_run"],
                    "early_stop_auc": fit["best_early_stop_auc"],
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )
            report_progress(len(rows), len(candidates) * len(folds), "trainings")
            if fold.number == len(folds):
                fold5_probabilities[candidate["name"]] = fit["probabilities"]
            logger.info(
                "CANDIDATE_FOLD_EVALUATED | candidate=%s | fold=%d | parameters=%d | validation_auc=%.6f | validation_accuracy=%.6f",
                candidate["name"],
                fold.number,
                rows[-1]["parameters"],
                metrics["auc"],
                metrics["accuracy"],
            )
    candidate_table = pd.DataFrame(rows)
    candidate_table.to_csv(run_dir / "candidates.csv", index=False)
    logger.info(
        "ARTIFACT_WRITTEN | path=candidates.csv | rows=%d", len(candidate_table)
    )

    selection_cfg = section["selection"]
    tie_tolerance = float(selection_cfg["tie_tolerance"])
    test_fold = folds[-1]
    test_labels = labels[test_fold.validation]
    test_dates = frame.index[test_fold.validation]
    results: dict = {"selection": {}, "test": {}}
    predictions = {"date": test_dates.strftime("%Y-%m-%d"), "label": test_labels}

    def report(key: str, probabilities: np.ndarray, extra: dict) -> None:
        logger.info("EVALUATION_STARTED | split=test | model=%s", key)
        metrics = classification_metrics(test_labels, probabilities)
        results["test"][key] = {**metrics, **extra}
        predictions[key] = probabilities
        logger.info(
            "EVALUATION_COMPLETED | split=test | model=%s | auc=%.6f | accuracy=%.6f | precision=%.6f | recall=%.6f",
            key,
            metrics["auc"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
        )

    for protocol, selection_folds in selection_cfg["protocols"].items():
        for family in ("qnn", "ann"):
            chosen = select(
                candidate_table[candidate_table["model"] == family],
                selection_folds,
                tie_tolerance,
            )
            key = f"{protocol}:{family}"
            results["selection"][key] = {
                "candidate": chosen.name,
                "mean_validation_auc": float(chosen["mean_auc"]),
                "parameters": int(chosen["parameters"]),
                "folds": selection_folds,
            }
            logger.info(
                "MODEL_SELECTED | protocol=%s | family=%s | candidate=%s | mean_validation_auc=%.6f | parameters=%d",
                protocol,
                family,
                chosen.name,
                chosen["mean_auc"],
                chosen["parameters"],
            )
            report(
                key,
                fold5_probabilities[chosen.name],
                {"candidate": chosen.name, "parameters": int(chosen["parameters"])},
            )

        # Fair baseline: a one-hidden-layer ANN matched to the selected QNN's parameter count.
        qnn_parameters = results["selection"][f"{protocol}:qnn"]["parameters"]
        width = matched_ann_width(qnn_parameters, n_features)
        matched = {"model": "ann", "hidden": [width], "name": f"ann_matched_{width}"}
        model, fit = train_on_fold(matched, len(candidates) + 1, len(folds) - 1)
        report(
            f"{protocol}:ann_parameter_matched",
            fit["probabilities"],
            {
                "candidate": matched["name"],
                "parameters": count_parameters(model),
                "qnn_parameters": qnn_parameters,
            },
        )

    # Floors on the test fold, fitted on the fold-5 training plus early-stop rows.
    fit_rows = np.concatenate([test_fold.train, test_fold.early_stop])
    scaled = fold_tensors[-1]["scaled"].double().cpu().numpy()
    majority = float(labels[fit_rows].mean() >= 0.5)
    report("majority_class", np.full(len(test_labels), majority), {"parameters": 0})
    logistic = LogisticRegression(max_iter=1000).fit(scaled[fit_rows], labels[fit_rows])
    report(
        "logistic_regression",
        logistic.predict_proba(scaled[test_fold.validation])[:, 1],
        {"parameters": int(logistic.coef_.size + logistic.intercept_.size)},
    )
    previous_direction = labels[test_fold.validation - 1]
    report(
        "previous_day_direction", previous_direction.astype(float), {"parameters": 0}
    )

    pd.DataFrame(predictions).to_csv(run_dir / "test_predictions.csv", index=False)
    logger.info(
        "ARTIFACT_WRITTEN | path=test_predictions.csv | rows=%d", len(test_labels)
    )
    results["test_window"] = [str(test_dates[0].date()), str(test_dates[-1].date())]
    results["n_candidates"] = len(candidates)
    status["selection"] = {
        "name": "mean_validation_auc",
        "direction": "maximize",
        "split": "validation folds 1-4 (leak_free, primary); folds 1-5 (paper, test-leaking)",
        "checkpoint": "best early-stop AUC epoch",
    }
    return results
