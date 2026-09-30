"""Training, evaluation, and mutation-test-suite construction.

The mutation test suite follows Section IV-B of the paper: the *original*
model predicts a fixed set of test samples, and the samples it classifies
correctly become the suite against which mutants are later scored.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lib.data import DatasetSplits
from lib.photonic_qnn import PhotonicQnnClassifier
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Optimisation settings for one photonic QNN fit.

    Attributes
    ----------
    epochs : int
        Number of full-batch optimisation steps.
    lr : float
        Adam learning rate.
    optimizer : str
        Optimizer name; only ``adam`` is supported.
    log_every : int
        Epoch interval for INFO-level progress records.
    """

    epochs: int
    lr: float
    optimizer: str = "adam"
    log_every: int = 10


def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Return the configured optimizer.

    Raises
    ------
    ValueError
        If an unsupported optimizer name is requested.
    """

    if cfg.optimizer.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}")
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)


def accuracy(model: PhotonicQnnClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return analytic (shot-free) classification accuracy."""

    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return (predictions == y).float().mean().item()


def train_photonic_qnn(spec, splits, cfg, seed: int, model=None) -> tuple:
    """Train one photonic QNN and return the model plus its metrics.

    Parameters
    ----------
    spec : PhotonicQnnSpec
        Circuit description.
    splits : IrisSplits
        Prepared data splits.
    cfg : TrainingConfig
        Optimisation settings.
    seed : int
        Seed for the phase initialisation of this replicate.
    model : torch.nn.Module, optional
        Pre-instantiated model (e.g., MutatableQCNN). If None, builds from spec.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        Trained model and a metrics mapping.
    """

    torch.manual_seed(seed)

    if model is None:
        model = PhotonicQnnClassifier(spec)

    optimizer = _build_optimizer(model, cfg)

    loss_fn = nn.NLLLoss()

    logger.info(
        "REPLICATE_STARTED | seed=%d | epochs=%d | optimizer=%s | lr=%g "
        "| trainable_params=%d",
        seed,
        cfg.epochs,
        cfg.optimizer,
        cfg.lr,
        model.num_trainable_parameters(),
    )

    started = time.time()
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(splits.x_train)

        # if torch.isnan(outputs).any():
        #     print("\n!!! DÉTECTION DE NAN PENDANT LE FORWARD PASS !!!")
        #     print("1. Présence de NaN dans l'entrée :", torch.isnan(splits.x_train).any().item())
        #     print("2. Min/Max de l'entrée :", splits.x_train.min().item(), "/", splits.x_train.max().item())
        #     print("3. Normes (5 premiers) :", torch.linalg.norm(splits.x_train.view(splits.x_train.size(0), -1), dim=1)[:5].tolist())
        #     raise ValueError("Le modèle quantique a généré un NaN pendant le calcul.")

        # --- GESTION ROBUSTE DE LA PERTE ---
        # Le modèle d'origine (Iris) renvoie des probas normalisées [0, 1].
        # Le modèle QCNN (MNIST) renvoie des logits bruts [-inf, inf].
        # La fonction log_softmax de PyTorch s'adapte parfaitement aux deux
        # situations sans jamais générer de logarithme invalide (NaN).
        log_probs = F.log_softmax(outputs, dim=1)

        loss = loss_fn(log_probs, splits.y_train)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if epoch % cfg.log_every == 0 or epoch in (1, cfg.epochs):
            train_accuracy = accuracy(model, splits.x_train, splits.y_train)
            val_accuracy = accuracy(model, splits.x_val, splits.y_val)

            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                }
            )
            logger.info(
                "TRAIN_EPOCH_COMPLETED | seed=%d | epoch=%d/%d | loss=%.6f "
                "| train_acc=%.4f | val_acc=%.4f",
                seed,
                epoch,
                cfg.epochs,
                loss.item(),
                train_accuracy,
                val_accuracy,
            )

    wall_clock = time.time() - started
    metrics = {
        "seed": seed,
        "final_loss": float(loss.item()),
        "train_accuracy": accuracy(model, splits.x_train, splits.y_train),
        "val_accuracy": accuracy(model, splits.x_val, splits.y_val),
        "test_accuracy": accuracy(model, splits.x_test, splits.y_test),
        "train_wall_clock_s": wall_clock,
        "history": history,
    }

    logger.info(
        "EVALUATION_COMPLETED | seed=%d | train_acc=%.4f | val_acc=%.4f "
        "| test_acc=%.4f | wall_clock_s=%.2f",
        seed,
        metrics["train_accuracy"],
        metrics["val_accuracy"],
        metrics["test_accuracy"],
        wall_clock,
    )

    return model, metrics


def predict_with_shots(
    model: PhotonicQnnClassifier, x: torch.Tensor, shots: int
) -> torch.Tensor:
    """Return predicted labels using a finite shot budget.

    The paper runs every model, original and mutated, with 1,024 shots and
    keeps ``the most occurring result`` as the label. The photonic counterpart
    samples the outcome distribution, folds it into class groups, and takes the
    most frequent group.

    Parameters
    ----------
    model : PhotonicQnnClassifier
        Trained model.
    x : torch.Tensor
        Scaled features.
    shots : int
        Shot budget; ``0`` falls back to the analytic distribution.

    Returns
    -------
    torch.Tensor
        Predicted integer labels.
    """

    model.eval()
    with torch.no_grad():
        return model(x, shots=shots).argmax(dim=1)


def build_mutation_test_suite(
    model: PhotonicQnnClassifier,
    splits: DatasetSplits,
    shots: int,
) -> dict:
    """Derive the paper's mutation test suite from the original model.

    Parameters
    ----------
    model : PhotonicQnnClassifier
        Trained original (unmutated) model.
    splits : IrisSplits
        Prepared data splits; the suite is drawn from ``x_suite``.
    shots : int
        Shot budget used for the reference predictions.

    Returns
    -------
    dict
        Suite description with the kept sample positions, their reference
        labels, and the original model's accuracy on the candidate samples.
    """

    predictions = predict_with_shots(model, splits.x_suite, shots)
    correct = predictions == splits.y_suite
    kept = torch.nonzero(correct, as_tuple=True)[0]

    # The same suite under the noise-free protocol. The two can differ on
    # low-margin samples, and mixing them would let sampling noise "kill" an
    # unmutated circuit, so each protocol keeps its own suite.
    analytic_predictions = predict_with_shots(model, splits.x_suite, 0)
    analytic_correct = analytic_predictions == splits.y_suite
    analytic_kept = torch.nonzero(analytic_correct, as_tuple=True)[0]

    suite = {
        "n_candidate_samples": int(len(splits.y_suite)),
        "n_suite_samples": int(kept.numel()),
        "suite_positions": kept.tolist(),
        "suite_test_indices": splits.suite_indices[kept.numpy()].tolist(),
        "reference_labels": predictions[kept].tolist(),
        "true_labels": splits.y_suite[kept].tolist(),
        "original_accuracy_on_candidates": float(correct.float().mean().item()),
        "shots": shots,
        "n_suite_samples_analytic": int(analytic_kept.numel()),
        "suite_positions_analytic": analytic_kept.tolist(),
        "original_accuracy_on_candidates_analytic": float(
            analytic_correct.float().mean().item()
        ),
    }
    logger.info(
        "TEST_SUITE_BUILT | candidates=%d | kept_shots=%d | kept_analytic=%d "
        "| shots=%d | acc=%.4f",
        suite["n_candidate_samples"],
        suite["n_suite_samples"],
        suite["n_suite_samples_analytic"],
        shots,
        suite["original_accuracy_on_candidates"],
    )
    return suite
