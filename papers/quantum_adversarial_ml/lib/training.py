"""
Training Utilities
==================

Training loops, evaluation, and optimization for quantum classifiers.
"""

import logging
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data
        optimizer: Optimizer
        criterion: Loss function
        device: Torch device

    Returns:
        average_loss, accuracy
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data
        criterion: Loss function
        device: Torch device

    Returns:
        average_loss, accuracy
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


class Trainer:
    """Trainer class for quantum classifiers."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data
            test_loader: Test data
            config: Training configuration
            device: Torch device
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or torch.device("cpu")

        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        lr = config.get("learning_rate", 0.01)
        optimizer_name = config.get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler (optional)
        scheduler_name = config.get("scheduler", None)
        if scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.get("step_size", 10), gamma=0.5
            )
        elif scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.get("epochs", 50)
            )
        else:
            self.scheduler = None

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "epoch_time": [],
        }

    def train(
        self,
        epochs: int,
        verbose: bool = True,
        save_best: bool = True,
        save_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            epochs: Number of epochs
            verbose: Print progress
            save_best: Save best model
            save_path: Path to save model

        Returns:
            Training results
        """
        best_acc = 0.0
        best_model_state = None

        pbar = tqdm(range(epochs), disable=not verbose)

        for _ in pbar:
            start_time = time.time()

            # Training
            train_loss, train_acc = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )

            # Evaluation
            test_loss, test_acc = evaluate(
                self.model, self.test_loader, self.criterion, self.device
            )

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - start_time

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["epoch_time"].append(epoch_time)

            # Track best model
            if test_acc > best_acc:
                best_acc = test_acc
                if save_best:
                    best_model_state = self.model.state_dict().copy()

            pbar.set_description(
                f"Loss: {train_loss:.4f} | Train: {train_acc:.3f} | Test: {test_acc:.3f}"
            )

        # Save best model
        if save_best and save_path and best_model_state is not None:
            torch.save(best_model_state, save_path)
            logger.info(f"Saved best model to {save_path}")

        return {
            "history": self.history,
            "best_accuracy": best_acc,
            "final_accuracy": self.history["test_acc"][-1],
        }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device = None,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience function to train a model.

    Args:
        model: Model to train
        train_loader: Training data
        test_loader: Test data
        config: Training configuration
        device: Torch device
        save_path: Path to save model

    Returns:
        Training results
    """
    trainer = Trainer(model, train_loader, test_loader, config, device)
    return trainer.train(
        epochs=config.get("epochs", 50),
        verbose=config.get("verbose", True),
        save_best=config.get("save_best", True),
        save_path=save_path,
    )
