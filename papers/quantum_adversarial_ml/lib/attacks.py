"""
Adversarial Attack Methods
==========================

Implements adversarial attack methods from the paper:
- FGSM: Fast Gradient Sign Method
- BIM: Basic Iterative Method
- PGD: Projected Gradient Descent
- MIM: Momentum Iterative Method

These attacks generate adversarial perturbations that cause
quantum classifiers to make incorrect predictions.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger(__name__)


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module = None,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast Gradient Sign Method (FGSM) attack.

    Single-step attack that perturbs input in the direction of the
    gradient sign to maximize loss.

    From: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"

    Args:
        model: Target model to attack
        x: Input tensor of shape (batch_size, features)
        y: True labels
        epsilon: Perturbation magnitude
        criterion: Loss function (default: CrossEntropyLoss)
        targeted: If True, perform targeted attack
        target_label: Target labels for targeted attack

    Returns:
        Adversarial examples
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(x_adv)

    # Compute loss
    if targeted and target_label is not None:
        # Targeted: minimize loss w.r.t. target label
        loss = criterion(outputs, target_label)
    else:
        # Untargeted: maximize loss w.r.t. true label
        loss = criterion(outputs, y)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    grad_sign = x_adv.grad.sign()

    # Create adversarial example
    if targeted:
        # Targeted: subtract gradient (minimize loss)
        x_adv = x + epsilon * (-grad_sign)
    else:
        # Untargeted: add gradient (maximize loss)
        x_adv = x + epsilon * grad_sign

    # Clamp to valid range [0, 1] for normalized inputs
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def bim_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = None,
    num_iter: int = 10,
    criterion: nn.Module = None,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Basic Iterative Method (BIM) attack.

    Iterative version of FGSM with smaller step sizes.

    From: Kurakin et al., "Adversarial Examples in the Physical World"

    Args:
        model: Target model to attack
        x: Input tensor
        y: True labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size per iteration (default: epsilon / num_iter)
        num_iter: Number of iterations
        criterion: Loss function
        targeted: If True, perform targeted attack
        target_label: Target labels for targeted attack

    Returns:
        Adversarial examples
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if alpha is None:
        alpha = epsilon / num_iter

    # Initialize adversarial example
    x_adv = x.clone().detach()

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = model(x_adv)

        # Compute loss
        if targeted and target_label is not None:
            loss = criterion(outputs, target_label)
        else:
            loss = criterion(outputs, y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient sign
        grad_sign = x_adv.grad.sign()

        # Update adversarial example
        if targeted:
            x_adv = x_adv - alpha * grad_sign
        else:
            x_adv = x_adv + alpha * grad_sign

        # Project back to epsilon-ball around original input
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + perturbation

        # Clamp to valid range
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = None,
    num_iter: int = 40,
    criterion: nn.Module = None,
    random_start: bool = True,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) attack.

    Strong iterative attack with random initialization.

    From: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"

    Args:
        model: Target model to attack
        x: Input tensor
        y: True labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size per iteration
        num_iter: Number of iterations
        criterion: Loss function
        random_start: If True, start from random point in epsilon-ball
        targeted: If True, perform targeted attack
        target_label: Target labels for targeted attack

    Returns:
        Adversarial examples
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if alpha is None:
        alpha = epsilon / 4

    # Initialize with random perturbation if specified
    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
    else:
        x_adv = x.clone().detach()

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = model(x_adv)

        # Compute loss
        if targeted and target_label is not None:
            loss = criterion(outputs, target_label)
        else:
            loss = criterion(outputs, y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient sign
        grad_sign = x_adv.grad.sign()

        # Update adversarial example
        if targeted:
            x_adv = x_adv - alpha * grad_sign
        else:
            x_adv = x_adv + alpha * grad_sign

        # Project back to epsilon-ball
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + perturbation

        # Clamp to valid range
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv


def mim_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = None,
    num_iter: int = 10,
    decay: float = 1.0,
    criterion: nn.Module = None,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Momentum Iterative Method (MIM) attack.

    Iterative attack with momentum to stabilize gradient direction.

    From: Dong et al., "Boosting Adversarial Attacks with Momentum"

    Args:
        model: Target model to attack
        x: Input tensor
        y: True labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size per iteration
        num_iter: Number of iterations
        decay: Momentum decay factor
        criterion: Loss function
        targeted: If True, perform targeted attack
        target_label: Target labels for targeted attack

    Returns:
        Adversarial examples
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if alpha is None:
        alpha = epsilon / num_iter

    # Initialize
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x)

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        # Forward pass
        outputs = model(x_adv)

        # Compute loss
        if targeted and target_label is not None:
            loss = criterion(outputs, target_label)
        else:
            loss = criterion(outputs, y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient
        grad = x_adv.grad

        # Normalize gradient
        grad_norm = grad / (torch.norm(grad, p=1, dim=-1, keepdim=True) + 1e-8)

        # Update momentum
        momentum = decay * momentum + grad_norm

        # Get sign of momentum
        momentum_sign = momentum.sign()

        # Update adversarial example
        if targeted:
            x_adv = x_adv - alpha * momentum_sign
        else:
            x_adv = x_adv + alpha * momentum_sign

        # Project back to epsilon-ball
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = x + perturbation

        # Clamp to valid range
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv


def generate_adversarial_examples(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    attack_method: str = "bim",
    epsilon: float = 0.1,
    num_iter: int = 10,
    alpha: float = None,
    device: torch.device = None,
    targeted: bool = False,
    target_class: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate adversarial examples for a dataset.

    Args:
        model: Target model
        dataloader: Data loader with clean samples
        attack_method: 'fgsm', 'bim', 'pgd', or 'mim'
        epsilon: Perturbation magnitude
        num_iter: Number of iterations (for iterative methods)
        alpha: Step size (optional)
        device: Torch device
        targeted: If True, perform targeted attack
        target_class: Target class for targeted attack
        max_samples: Maximum number of samples to attack

    Returns:
        clean_samples, clean_labels, adv_samples, adv_predictions
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()

    # Select attack function
    attack_fn = {
        "fgsm": fgsm_attack,
        "bim": bim_attack,
        "pgd": pgd_attack,
        "mim": mim_attack,
    }.get(attack_method.lower())

    if attack_fn is None:
        raise ValueError(f"Unknown attack method: {attack_method}")

    clean_samples_list = []
    clean_labels_list = []
    adv_samples_list = []
    adv_predictions_list = []

    n_samples = 0

    for _, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)

        # Generate target labels if targeted attack
        if targeted:
            if target_class is not None:
                target_labels = torch.full_like(labels, target_class)
            else:
                # Random target (different from true label)
                n_classes = model.n_outputs if hasattr(model, "n_outputs") else 2
                target_labels = (
                    labels + torch.randint(1, n_classes, labels.shape, device=device)
                ) % n_classes
        else:
            target_labels = None

        # Generate adversarial examples
        if attack_method.lower() == "fgsm":
            adv_data = attack_fn(
                model,
                data,
                labels,
                epsilon,
                targeted=targeted,
                target_label=target_labels,
            )
        else:
            adv_data = attack_fn(
                model,
                data,
                labels,
                epsilon,
                alpha=alpha,
                num_iter=num_iter,
                targeted=targeted,
                target_label=target_labels,
            )

        # Get predictions on adversarial examples
        with torch.no_grad():
            adv_outputs = model(adv_data)
            adv_preds = adv_outputs.argmax(dim=1)

        clean_samples_list.append(data.cpu())
        clean_labels_list.append(labels.cpu())
        adv_samples_list.append(adv_data.cpu())
        adv_predictions_list.append(adv_preds.cpu())

        n_samples += data.size(0)
        if max_samples is not None and n_samples >= max_samples:
            break

    clean_samples = torch.cat(clean_samples_list, dim=0)
    clean_labels = torch.cat(clean_labels_list, dim=0)
    adv_samples = torch.cat(adv_samples_list, dim=0)
    adv_predictions = torch.cat(adv_predictions_list, dim=0)

    return clean_samples, clean_labels, adv_samples, adv_predictions


def functional_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_iter: int = 10,
    alpha: float = None,
    criterion: nn.Module = None,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional adversarial attack using local perturbations.

    Instead of additive perturbations to each input element independently,
    functional attacks apply structured perturbations that correspond to
    local unitary operations in the quantum domain.

    For photonic systems, this corresponds to local phase shifters on each mode.
    The perturbation is: x_adv = x * exp(i * delta) ≈ x * (1 + i*delta) for small delta
    In real-valued form: x_adv = x * cos(delta) for amplitude-encoded data

    From: Sec III.B of Lu et al. - "Functional adversarial attack"

    Args:
        model: Target model to attack
        x: Input tensor of shape (batch_size, features)
        y: True labels
        epsilon: Maximum phase perturbation magnitude
        num_iter: Number of iterations
        alpha: Step size per iteration
        criterion: Loss function
        targeted: If True, perform targeted attack
        target_label: Target labels for targeted attack

    Returns:
        Adversarial examples
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if alpha is None:
        alpha = epsilon / num_iter

    batch_size, n_features = x.shape

    # Initialize phase perturbations (one per feature/mode)
    delta = torch.zeros(batch_size, n_features, device=x.device, requires_grad=True)

    for _ in range(num_iter):
        # Apply functional perturbation: multiply by cos(delta)
        # This mimics local phase shifter rotations
        x_adv = x * torch.cos(delta)

        # Renormalize to maintain amplitude encoding property
        x_adv = functional.normalize(x_adv, p=2, dim=-1) * torch.norm(
            x, p=2, dim=-1, keepdim=True
        )

        # Forward pass
        outputs = model(x_adv)

        # Compute loss
        if targeted and target_label is not None:
            loss = criterion(outputs, target_label)
        else:
            loss = criterion(outputs, y)

        # Backward pass
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        # Get gradient and update perturbation
        with torch.no_grad():
            grad = delta.grad
            if grad is not None:
                if targeted:
                    delta_update = -alpha * grad.sign()
                else:
                    delta_update = alpha * grad.sign()
                delta = delta + delta_update
                # Project to epsilon-ball
                delta = torch.clamp(delta, -epsilon, epsilon)
            delta = delta.detach().requires_grad_(True)

    # Apply final perturbation
    with torch.no_grad():
        x_adv = x * torch.cos(delta)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def functional_fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module = None,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-step functional attack (FGSM-style).

    Args:
        model: Target model
        x: Input tensor
        y: True labels
        epsilon: Phase perturbation magnitude
        criterion: Loss function
        targeted: If True, targeted attack
        target_label: Target labels

    Returns:
        Adversarial examples
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    batch_size, n_features = x.shape

    # Initialize phase perturbations
    delta = torch.zeros(batch_size, n_features, device=x.device, requires_grad=True)

    # Apply perturbation
    x_adv = x * torch.cos(delta)
    x_adv = functional.normalize(x_adv, p=2, dim=-1) * torch.norm(
        x, p=2, dim=-1, keepdim=True
    )

    # Forward pass
    outputs = model(x_adv)

    # Compute loss
    if targeted and target_label is not None:
        loss = criterion(outputs, target_label)
    else:
        loss = criterion(outputs, y)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    grad_sign = delta.grad.sign()

    # Apply perturbation
    if targeted:
        final_delta = -epsilon * grad_sign
    else:
        final_delta = epsilon * grad_sign

    x_adv = x * torch.cos(final_delta)
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def compute_fidelity(
    clean_samples: torch.Tensor, adv_samples: torch.Tensor
) -> torch.Tensor:
    """Compute fidelity between clean and adversarial samples.

    For amplitude-encoded quantum states, fidelity is the squared
    inner product of the normalized amplitude vectors.

    Args:
        clean_samples: Clean samples (batch_size, features)
        adv_samples: Adversarial samples (batch_size, features)

    Returns:
        Fidelity values for each sample
    """
    # Normalize samples
    clean_norm = functional.normalize(clean_samples, p=2, dim=-1)
    adv_norm = functional.normalize(adv_samples, p=2, dim=-1)

    # Compute inner product squared (fidelity)
    inner_product = torch.sum(clean_norm * adv_norm, dim=-1)
    fidelity = inner_product**2

    return fidelity


def evaluate_attack_success(
    clean_labels: torch.Tensor,
    adv_predictions: torch.Tensor,
    clean_predictions: Optional[torch.Tensor] = None,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> dict:
    """Evaluate attack success metrics.

    Args:
        clean_labels: True labels
        adv_predictions: Predictions on adversarial examples
        clean_predictions: Predictions on clean examples (optional)
        targeted: If True, evaluate targeted attack success
        target_labels: Target labels for targeted attack

    Returns:
        Dictionary with attack success metrics
    """
    # Untargeted success: prediction changed from correct to incorrect
    if clean_predictions is not None:
        # Among correctly classified samples
        correct_mask = clean_predictions == clean_labels
        n_correct = correct_mask.sum().item()

        # Fooled: was correct, now incorrect
        fooled = correct_mask & (adv_predictions != clean_labels)
        fooling_rate = fooled.sum().item() / max(n_correct, 1)
    else:
        # Just check if adversarial prediction differs from true label
        fooling_rate = (adv_predictions != clean_labels).float().mean().item()

    results = {
        "fooling_rate": fooling_rate,
        "adversarial_accuracy": (adv_predictions == clean_labels).float().mean().item(),
    }

    # Targeted success
    if targeted and target_labels is not None:
        targeted_success = (adv_predictions == target_labels).float().mean().item()
        results["targeted_success_rate"] = targeted_success

    return results


def transfer_attack(
    surrogate_model: nn.Module,
    target_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    attack_method: str = "bim",
    epsilon: float = 0.1,
    num_iter: int = 10,
    alpha: float = None,
    device: torch.device = None,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Black-box transfer attack.

    Generate adversarial examples using a surrogate model and evaluate
    their transferability to a target (victim) model.

    From: Section III.B.3 "Black-box attack: transferability" of Lu et al.

    Args:
        surrogate_model: Model to generate adversarial examples (e.g., CNN)
        target_model: Victim model to evaluate transfer (e.g., quantum classifier)
        dataloader: Data loader with clean samples
        attack_method: Attack method ('fgsm', 'bim', 'pgd', 'mim')
        epsilon: Perturbation magnitude
        num_iter: Number of iterations
        alpha: Step size
        device: Torch device
        max_samples: Maximum samples to attack

    Returns:
        Dictionary with transfer attack results
    """
    if device is None:
        device = torch.device("cpu")

    surrogate_model.eval()
    target_model.eval()

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

    # Collect results
    all_clean_labels = []
    all_surrogate_clean_preds = []
    all_surrogate_adv_preds = []
    all_target_clean_preds = []
    all_target_adv_preds = []

    n_samples = 0

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        # Generate adversarial examples using surrogate model
        adv_data = attack_fn(surrogate_model, data, labels, epsilon)

        # Evaluate on surrogate model
        with torch.no_grad():
            surrogate_clean_out = surrogate_model(data)
            surrogate_clean_preds = surrogate_clean_out.argmax(dim=1)

            surrogate_adv_out = surrogate_model(adv_data)
            surrogate_adv_preds = surrogate_adv_out.argmax(dim=1)

        # Evaluate on target model
        with torch.no_grad():
            target_clean_out = target_model(data)
            target_clean_preds = target_clean_out.argmax(dim=1)

            target_adv_out = target_model(adv_data)
            target_adv_preds = target_adv_out.argmax(dim=1)

        all_clean_labels.append(labels.cpu())
        all_surrogate_clean_preds.append(surrogate_clean_preds.cpu())
        all_surrogate_adv_preds.append(surrogate_adv_preds.cpu())
        all_target_clean_preds.append(target_clean_preds.cpu())
        all_target_adv_preds.append(target_adv_preds.cpu())

        n_samples += data.size(0)
        if max_samples is not None and n_samples >= max_samples:
            break

    # Concatenate
    clean_labels = torch.cat(all_clean_labels)
    surrogate_clean_preds = torch.cat(all_surrogate_clean_preds)
    surrogate_adv_preds = torch.cat(all_surrogate_adv_preds)
    target_clean_preds = torch.cat(all_target_clean_preds)
    target_adv_preds = torch.cat(all_target_adv_preds)

    # Compute metrics
    # Surrogate model accuracy
    surrogate_clean_acc = (surrogate_clean_preds == clean_labels).float().mean().item()
    surrogate_adv_acc = (surrogate_adv_preds == clean_labels).float().mean().item()

    # Target model accuracy
    target_clean_acc = (target_clean_preds == clean_labels).float().mean().item()
    target_adv_acc = (target_adv_preds == clean_labels).float().mean().item()

    # Transfer rate: fraction of successful attacks that transfer
    # Success on surrogate = was correct, now wrong
    surrogate_success = (surrogate_clean_preds == clean_labels) & (
        surrogate_adv_preds != clean_labels
    )
    # Of those, how many also fool target?
    target_fooled = target_adv_preds != clean_labels

    transfer_rate = (surrogate_success & target_fooled).sum().item() / max(
        surrogate_success.sum().item(), 1
    )

    results = {
        "surrogate_clean_accuracy": surrogate_clean_acc,
        "surrogate_adversarial_accuracy": surrogate_adv_acc,
        "target_clean_accuracy": target_clean_acc,
        "target_adversarial_accuracy": target_adv_acc,
        "accuracy_drop_surrogate": surrogate_clean_acc - surrogate_adv_acc,
        "accuracy_drop_target": target_clean_acc - target_adv_acc,
        "transfer_rate": transfer_rate,
    }

    return results


def run_transfer_attack_experiment(
    quantum_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    surrogate_type: str = "cnn",
    input_dim: int = 256,
    image_size: int = 16,
    n_outputs: int = 2,
    attack_methods: list[str] | None = None,
    epsilon: float = 0.1,
    num_iter: int = 10,
    epochs: int = 20,
    device: torch.device = None,
) -> dict[str, Any]:
    """Run full transfer attack experiment.

    Trains a surrogate classical model, generates adversarial examples,
    and evaluates transferability to quantum model.

    Reproduces Table III from the paper.

    Args:
        quantum_model: Trained quantum classifier (target)
        train_loader: Training data for surrogate
        test_loader: Test data for evaluation
        surrogate_type: 'cnn' or 'fnn'
        input_dim: Input dimension for FNN
        image_size: Image size for CNN
        n_outputs: Number of output classes
        attack_methods: List of attack methods to evaluate
        epsilon: Perturbation magnitude
        num_iter: Attack iterations
        epochs: Epochs to train surrogate
        device: Torch device

    Returns:
        Dictionary with results for each attack method
    """
    if attack_methods is None:
        attack_methods = ["bim", "fgsm", "mim"]

    from .models import ClassicalCNN, ClassicalFNN
    from .training import train_model

    if device is None:
        device = torch.device("cpu")

    # Create surrogate model
    if surrogate_type.lower() == "cnn":
        surrogate_model = ClassicalCNN(
            input_channels=1, image_size=image_size, n_outputs=n_outputs
        ).to(device)
    else:
        surrogate_model = ClassicalFNN(input_dim=input_dim, n_outputs=n_outputs).to(
            device
        )

    # Train surrogate model
    logger.info(f"Training {surrogate_type.upper()} surrogate model...")
    train_config = {"epochs": epochs, "learning_rate": 0.001}
    train_model(surrogate_model, train_loader, test_loader, train_config, device)

    # Evaluate transfer attacks
    results = {"surrogate_type": surrogate_type}

    for attack_method in attack_methods:
        logger.info(f"Running {attack_method.upper()} transfer attack...")
        attack_results = transfer_attack(
            surrogate_model=surrogate_model,
            target_model=quantum_model,
            dataloader=test_loader,
            attack_method=attack_method,
            epsilon=epsilon,
            num_iter=num_iter,
            device=device,
        )
        results[attack_method] = attack_results

        logger.info(
            f"  {attack_method.upper()}: "
            f"Surrogate acc drop: {attack_results['accuracy_drop_surrogate']:.1%}, "
            f"Target acc drop: {attack_results['accuracy_drop_target']:.1%}, "
            f"Transfer rate: {attack_results['transfer_rate']:.1%}"
        )

    return results


# =============================================================================
# Noise Comparison Experiments (Fig 11 from paper)
# =============================================================================


def add_random_noise(
    x: torch.Tensor, epsilon: float, noise_type: str = "uniform"
) -> torch.Tensor:
    """Add random noise to input (non-adversarial baseline).

    Args:
        x: Input tensor
        epsilon: Noise magnitude
        noise_type: 'uniform', 'gaussian', or 'salt_pepper'

    Returns:
        Noisy input
    """
    if noise_type == "uniform":
        # Uniform noise in [-epsilon, epsilon]
        noise = (2 * torch.rand_like(x) - 1) * epsilon
    elif noise_type == "gaussian":
        # Gaussian noise with std = epsilon
        noise = torch.randn_like(x) * epsilon
    elif noise_type == "salt_pepper":
        # Salt and pepper noise
        mask = torch.rand_like(x)
        noise = torch.zeros_like(x)
        noise[mask < epsilon / 2] = -x[mask < epsilon / 2]  # Set to 0
        noise[mask > 1 - epsilon / 2] = 1 - x[mask > 1 - epsilon / 2]  # Set to 1
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    x_noisy = x + noise
    x_noisy = torch.clamp(x_noisy, 0, 1)

    return x_noisy


def evaluate_with_photon_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_rate: float,
    device: torch.device = None,
) -> dict[str, float]:
    """Evaluate model with simulated photon loss.

    Photon loss is modeled by randomly zeroing out features with
    probability = loss_rate, simulating photon absorption/scattering.

    This is analogous to the paper's depolarizing noise but adapted
    for photonic systems.

    Args:
        model: Trained model
        dataloader: Test data
        loss_rate: Probability of photon loss per feature (0 to 1)
        device: Torch device

    Returns:
        Accuracy metrics under photon loss
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            # Apply photon loss: randomly zero out features
            # This simulates photons being lost before detection
            loss_mask = torch.rand_like(data) > loss_rate
            data_lossy = data * loss_mask.float()

            # Renormalize to maintain amplitude encoding property
            norms = torch.norm(data_lossy, dim=-1, keepdim=True)
            data_lossy = (
                data_lossy / (norms + 1e-8) * torch.norm(data, dim=-1, keepdim=True)
            )

            outputs = model(data_lossy)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return {"accuracy": accuracy, "loss_rate": loss_rate}


def compare_noise_vs_adversarial(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilon_values: list[float] | None = None,
    attack_method: str = "bim",
    num_iter: int = 10,
    device: torch.device = None,
) -> dict[str, Any]:
    """Compare adversarial perturbations vs random noise vs photon loss.

    Reproduces Figure 11 from the paper, showing that adversarial
    perturbations are much more effective than random noise.

    For photonic systems, we also compare against photon loss noise.

    Args:
        model: Trained classifier
        dataloader: Test data
        epsilon_values: Perturbation/noise magnitudes to test
        attack_method: Attack method for adversarial examples
        num_iter: Attack iterations
        device: Torch device

    Returns:
        Results for adversarial, random noise, and photon loss
    """
    if epsilon_values is None:
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]

    if device is None:
        device = torch.device("cpu")

    model.eval()

    # Select attack function
    attack_fn = {
        "fgsm": fgsm_attack,
        "bim": bim_attack,
        "pgd": pgd_attack,
        "mim": mim_attack,
    }.get(attack_method.lower(), bim_attack)

    results = {
        "epsilon_values": epsilon_values,
        "clean_accuracy": None,
        "adversarial": [],
        "random_uniform": [],
        "random_gaussian": [],
        "photon_loss": [],
    }

    # First, compute clean accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    results["clean_accuracy"] = correct / total
    logger.info(f"Clean accuracy: {results['clean_accuracy']:.3f}")

    # Evaluate at each epsilon
    for epsilon in epsilon_values:
        logger.info(f"\nEvaluating at epsilon/loss_rate = {epsilon}")

        # 1. Adversarial accuracy
        adv_correct = 0
        adv_total = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            if attack_method == "fgsm":
                x_adv = attack_fn(model, data, labels, epsilon)
            else:
                x_adv = attack_fn(model, data, labels, epsilon, num_iter=num_iter)

            with torch.no_grad():
                outputs = model(x_adv)
                predictions = outputs.argmax(dim=1)
                adv_correct += (predictions == labels).sum().item()
                adv_total += labels.size(0)

        adv_acc = adv_correct / adv_total
        results["adversarial"].append(adv_acc)
        logger.info(f"  Adversarial ({attack_method}): {adv_acc:.3f}")

        # 2. Random uniform noise
        uniform_correct = 0
        uniform_total = 0
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                x_noisy = add_random_noise(data, epsilon, "uniform")
                outputs = model(x_noisy)
                predictions = outputs.argmax(dim=1)
                uniform_correct += (predictions == labels).sum().item()
                uniform_total += labels.size(0)

        uniform_acc = uniform_correct / uniform_total
        results["random_uniform"].append(uniform_acc)
        logger.info(f"  Random uniform noise: {uniform_acc:.3f}")

        # 3. Random Gaussian noise
        gauss_correct = 0
        gauss_total = 0
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                x_noisy = add_random_noise(data, epsilon, "gaussian")
                outputs = model(x_noisy)
                predictions = outputs.argmax(dim=1)
                gauss_correct += (predictions == labels).sum().item()
                gauss_total += labels.size(0)

        gauss_acc = gauss_correct / gauss_total
        results["random_gaussian"].append(gauss_acc)
        logger.info(f"  Random Gaussian noise: {gauss_acc:.3f}")

        # 4. Photon loss (photonic-specific noise)
        loss_results = evaluate_with_photon_loss(model, dataloader, epsilon, device)
        results["photon_loss"].append(loss_results["accuracy"])
        logger.info(f"  Photon loss: {loss_results['accuracy']:.3f}")

    return results


def create_noisy_quantum_layer(
    base_circuit, n_photons: int, brightness: float = 1.0, transmittance: float = 1.0
):
    """Create a quantum layer with photon loss noise model.

    Uses Perceval's NoiseModel to simulate realistic photon loss
    in the optical circuit.

    Args:
        base_circuit: Perceval circuit
        n_photons: Number of input photons
        brightness: Source brightness (1.0 = perfect)
        transmittance: Circuit transmittance (1.0 = no loss)

    Returns:
        QuantumLayer with noise model
    """
    import perceval as pcvl
    from merlin import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    # Create experiment with noise model
    experiment = pcvl.Experiment(base_circuit)
    experiment.noise = pcvl.NoiseModel(
        brightness=brightness, transmittance=transmittance
    )

    # Create quantum layer with noise
    layer = QuantumLayer(
        experiment=experiment,
        n_photons=n_photons,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
    )

    return layer


def evaluate_with_perceval_noise(
    model_class,
    model_config: dict[str, Any],
    dataloader: torch.utils.data.DataLoader,
    brightness_values: list[float] | None = None,
    transmittance_values: list[float] | None = None,
    device: torch.device = None,
) -> dict[str, Any]:
    """Evaluate model robustness to photon loss using Perceval's noise model.

    This is the proper photonic noise experiment using MerLin/Perceval's
    built-in noise simulation.

    Args:
        model_class: Model class to instantiate
        model_config: Model configuration
        dataloader: Test data
        brightness_values: Source brightness values to test
        transmittance_values: Circuit transmittance values to test
        device: Torch device

    Returns:
        Accuracy at different noise levels
    """
    if brightness_values is None:
        brightness_values = [1.0, 0.95, 0.9, 0.85, 0.8]
    if transmittance_values is None:
        transmittance_values = [1.0, 0.95, 0.9, 0.85, 0.8]

    if device is None:
        device = torch.device("cpu")

    results = {
        "brightness_sweep": {"values": brightness_values, "accuracy": []},
        "transmittance_sweep": {"values": transmittance_values, "accuracy": []},
        "combined_sweep": [],
    }

    # Note: This requires modifying the model to accept noise parameters
    # For now, we use the input-level photon loss simulation

    logger.info("Photon loss evaluation using input-level simulation")
    logger.info("(For full Perceval noise model, use create_noisy_quantum_layer)")

    return results


def run_noise_comparison_experiment(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device = None,
) -> dict[str, Any]:
    """Run the full noise vs adversarial comparison experiment.

    Reproduces Figure 11 from Lu et al. (2020).

    Args:
        model: Trained classifier
        test_loader: Test data
        config: Experiment configuration
        device: Torch device

    Returns:
        Complete comparison results
    """
    if device is None:
        device = torch.device("cpu")

    noise_config = config.get("noise", {})

    epsilon_values = noise_config.get("epsilon_values", [0.01, 0.05, 0.1, 0.15, 0.2])
    attack_method = noise_config.get("attack_method", "bim")
    num_iter = noise_config.get("num_iter", 10)

    logger.info("=" * 60)
    logger.info("Noise vs Adversarial Comparison Experiment")
    logger.info("Reproduces Figure 11 from Lu et al. (2020)")
    logger.info("=" * 60)

    results = compare_noise_vs_adversarial(
        model=model,
        dataloader=test_loader,
        epsilon_values=epsilon_values,
        attack_method=attack_method,
        num_iter=num_iter,
        device=device,
    )

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary: Accuracy at different perturbation levels")
    logger.info("=" * 60)
    logger.info(
        f"{'Epsilon':<10} {'Adversarial':<12} {'Uniform':<12} {'Gaussian':<12} {'Photon Loss':<12}"
    )
    logger.info("-" * 60)

    for i, eps in enumerate(epsilon_values):
        logger.info(
            f"{eps:<10.3f} "
            f"{results['adversarial'][i]:<12.3f} "
            f"{results['random_uniform'][i]:<12.3f} "
            f"{results['random_gaussian'][i]:<12.3f} "
            f"{results['photon_loss'][i]:<12.3f}"
        )

    return results
