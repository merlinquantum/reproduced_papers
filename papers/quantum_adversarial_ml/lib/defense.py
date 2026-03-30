"""
Defense Strategies
==================

Implements defense strategies against adversarial attacks:
- Adversarial Training: Retrain with adversarial examples
- Robust Optimization: Min-max formulation

From the paper:
    min_Θ (1/N) Σ max_{U_δ ∈ Δ} L(h(U_δ|ψ_in); Θ), y)
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .attacks import bim_attack, fgsm_attack, mim_attack, pgd_attack

logger = logging.getLogger(__name__)


def adversarial_training_step(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    attack_method: str = "bim",
    epsilon: float = 0.1,
    num_iter: int = 3,
    alpha: float = None,
    mix_ratio: float = 0.5,
    device: torch.device = None,
) -> tuple[float, float, float]:
    """Single adversarial training step.

    Performs the inner maximization (generate adversarial examples)
    followed by the outer minimization (update model parameters).

    Args:
        model: Model to train
        data: Input batch
        labels: True labels
        optimizer: Optimizer
        criterion: Loss function
        attack_method: Attack method to use
        epsilon: Perturbation magnitude
        num_iter: Number of attack iterations
        alpha: Attack step size
        mix_ratio: Ratio of adversarial examples in batch (0.5 = 50%)
        device: Torch device

    Returns:
        loss, clean_accuracy, adversarial_accuracy
    """
    if device is None:
        device = data.device

    model.train()

    # Select attack function
    attack_fn = {
        "fgsm": lambda m, x, y, e: fgsm_attack(m, x, y, e),
        "bim": lambda m, x, y, e: bim_attack(
            m, x, y, e, alpha=alpha, num_iter=num_iter
        ),
        "pgd": lambda m, x, y, e: pgd_attack(
            m, x, y, e, alpha=alpha, num_iter=num_iter
        ),
        "mim": lambda m, x, y, e: mim_attack(
            m, x, y, e, alpha=alpha, num_iter=num_iter
        ),
    }.get(attack_method.lower())

    if attack_fn is None:
        raise ValueError(f"Unknown attack method: {attack_method}")

    # Step 1: Generate adversarial examples (inner maximization)
    model.eval()
    with torch.enable_grad():
        adv_data = attack_fn(model, data, labels, epsilon)
    model.train()

    # Step 2: Mix clean and adversarial data
    batch_size = data.size(0)
    n_adv = int(batch_size * mix_ratio)

    # Randomly select which samples to replace with adversarial
    indices = torch.randperm(batch_size)[:n_adv]
    mixed_data = data.clone()
    mixed_data[indices] = adv_data[indices]

    # Step 3: Update model (outer minimization)
    optimizer.zero_grad()
    outputs = model(mixed_data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Compute accuracies
    with torch.no_grad():
        clean_outputs = model(data)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_acc = (clean_preds == labels).float().mean().item()

        adv_outputs = model(adv_data)
        adv_preds = adv_outputs.argmax(dim=1)
        adv_acc = (adv_preds == labels).float().mean().item()

    return loss.item(), clean_acc, adv_acc


class AdversarialTrainer:
    """Adversarial trainer implementing robust optimization.

    Solves the min-max problem:
        min_Θ (1/N) Σ max_{δ ∈ Δ} L(h(x + δ; Θ), y)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device = None,
    ):
        """Initialize adversarial trainer.

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
        lr = config.get("learning_rate", 0.005)
        optimizer_name = config.get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Attack configuration
        self.attack_method = config.get("attack_method", "bim")
        self.epsilon = config.get("epsilon", 0.1)
        self.num_iter = config.get("num_iter", 3)
        self.alpha = config.get("alpha", None)
        self.mix_ratio = config.get("mix_ratio", 0.5)

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_adv_acc": [],
            "test_acc": [],
            "test_adv_acc": [],
        }

    def train(
        self, epochs: int, verbose: bool = True, eval_attack: bool = True
    ) -> dict[str, Any]:
        """Train the model with adversarial examples.

        Args:
            epochs: Number of epochs
            verbose: Print progress
            eval_attack: Evaluate attack resistance during training

        Returns:
            Training results
        """
        best_adv_acc = 0.0

        pbar = tqdm(range(epochs), disable=not verbose)

        for _ in pbar:
            # Training phase
            self.model.train()
            total_loss = 0.0
            total_clean_acc = 0.0
            total_adv_acc = 0.0
            n_batches = 0

            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                loss, clean_acc, adv_acc = adversarial_training_step(
                    self.model,
                    data,
                    labels,
                    self.optimizer,
                    self.criterion,
                    attack_method=self.attack_method,
                    epsilon=self.epsilon,
                    num_iter=self.num_iter,
                    alpha=self.alpha,
                    mix_ratio=self.mix_ratio,
                    device=self.device,
                )

                total_loss += loss
                total_clean_acc += clean_acc
                total_adv_acc += adv_acc
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_clean_acc = total_clean_acc / n_batches
            avg_adv_acc = total_adv_acc / n_batches

            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(avg_clean_acc)
            self.history["train_adv_acc"].append(avg_adv_acc)

            # Evaluation phase
            test_clean_acc, test_adv_acc = self._evaluate(eval_attack)
            self.history["test_acc"].append(test_clean_acc)
            self.history["test_adv_acc"].append(test_adv_acc)

            # Track best
            if test_adv_acc > best_adv_acc:
                best_adv_acc = test_adv_acc

            pbar.set_description(
                f"Loss: {avg_loss:.4f} | Clean: {test_clean_acc:.3f} | Adv: {test_adv_acc:.3f}"
            )

        return {
            "history": self.history,
            "best_adversarial_accuracy": best_adv_acc,
            "final_clean_accuracy": self.history["test_acc"][-1],
            "final_adversarial_accuracy": self.history["test_adv_acc"][-1],
        }

    def _evaluate(self, eval_attack: bool = True) -> tuple[float, float]:
        """Evaluate model on test set.

        Args:
            eval_attack: If True, also evaluate under attack

        Returns:
            clean_accuracy, adversarial_accuracy
        """
        self.model.eval()

        # Clean accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        clean_acc = correct / total

        # Adversarial accuracy
        if eval_attack:
            adv_correct = 0
            adv_total = 0

            attack_fn = {
                "fgsm": lambda m, x, y, e: fgsm_attack(m, x, y, e),
                "bim": lambda m, x, y, e: bim_attack(
                    m, x, y, e, alpha=self.alpha, num_iter=self.num_iter
                ),
                "pgd": lambda m, x, y, e: pgd_attack(
                    m, x, y, e, alpha=self.alpha, num_iter=self.num_iter
                ),
                "mim": lambda m, x, y, e: mim_attack(
                    m, x, y, e, alpha=self.alpha, num_iter=self.num_iter
                ),
            }.get(self.attack_method.lower())

            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Generate adversarial examples
                with torch.enable_grad():
                    adv_data = attack_fn(self.model, data, labels, self.epsilon)

                # Evaluate
                with torch.no_grad():
                    adv_outputs = self.model(adv_data)
                    adv_preds = adv_outputs.argmax(dim=1)
                    adv_correct += (adv_preds == labels).sum().item()
                    adv_total += labels.size(0)

            adv_acc = adv_correct / adv_total
        else:
            adv_acc = 0.0

        return clean_acc, adv_acc


def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    attack_methods: list[str] | None = None,
    epsilons: list[float] | None = None,
    num_iter: int = 10,
    device: torch.device = None,
) -> dict[str, dict[float, float]]:
    """Evaluate model robustness across multiple attacks and epsilon values.

    Args:
        model: Model to evaluate
        test_loader: Test data
        attack_methods: List of attack methods
        epsilons: List of epsilon values
        num_iter: Number of iterations for iterative attacks
        device: Torch device

    Returns:
        Dictionary mapping attack -> epsilon -> accuracy
    """
    if attack_methods is None:
        attack_methods = ["fgsm", "bim", "pgd"]
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2]

    if device is None:
        device = torch.device("cpu")

    model.eval()
    results = {}

    for attack_method in attack_methods:
        results[attack_method] = {}

        attack_fn = {
            "fgsm": lambda m, x, y, e: fgsm_attack(m, x, y, e),
            "bim": lambda m, x, y, e: bim_attack(m, x, y, e, num_iter=num_iter),
            "pgd": lambda m, x, y, e: pgd_attack(m, x, y, e, num_iter=num_iter),
            "mim": lambda m, x, y, e: mim_attack(m, x, y, e, num_iter=num_iter),
        }.get(attack_method.lower())

        for eps in epsilons:
            correct = 0
            total = 0

            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                # Generate adversarial examples
                with torch.enable_grad():
                    adv_data = attack_fn(model, data, labels, eps)

                # Evaluate
                with torch.no_grad():
                    outputs = model(adv_data)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            results[attack_method][eps] = correct / total

            logger.info(
                f"{attack_method.upper()} (ε={eps}): {results[attack_method][eps]:.3f}"
            )

    return results
