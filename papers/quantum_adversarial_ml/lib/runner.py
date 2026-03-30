"""
Runner Module
=============

Main entry point for the quantum adversarial learning experiments.
Handles configuration, experiment execution, and results saving.

Experiments from the paper:
1. MNIST binary classification (digits 1 vs 9)
2. MNIST multi-class classification (digits 1, 3, 7, 9)
3. Ising model phase classification
4. Adversarial attacks and defense

Each config should represent ONE run. For figures requiring multiple runs,
use the shell scripts in scripts/ directory.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .attacks import (
    bim_attack,
    compute_fidelity,
    evaluate_attack_success,
    fgsm_attack,
    generate_adversarial_examples,
)
from .datasets import create_dataloaders
from .defense import AdversarialTrainer, evaluate_robustness
from .models import create_model
from .training import train_model

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def find_model_path(load_path: str) -> Optional[str]:
    """Find the actual model path, checking run subdirectories if needed.

    Models are saved in timestamped run directories like:
        results/train_quantum/run_20260120-123456/model.pt

    But configs typically specify:
        results/train_quantum/model.pt

    This function searches for the model in:
    1. The exact path specified
    2. The latest run_* subdirectory

    Args:
        load_path: Configured path to model (may or may not exist)

    Returns:
        Actual path to model.pt, or None if not found
    """
    if load_path is None:
        return None

    path = Path(load_path)

    # Check if exact path exists
    if path.exists():
        return str(path)

    # Check parent directory for run_* subdirectories
    parent = path.parent
    model_name = path.name  # usually "model.pt"

    if parent.exists():
        # Find all run_* directories
        run_dirs = sorted(
            parent.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        for run_dir in run_dirs:
            candidate = run_dir / model_name
            if candidate.exists():
                logger.info(f"Found model in run directory: {candidate}")
                return str(candidate)

    # Also check if the parent itself contains the model (for direct saves)
    direct_model = parent / model_name
    if direct_model.exists():
        return str(direct_model)

    logger.warning(f"Model not found at {load_path} or in run subdirectories")
    return None


def load_model(
    model_path: str, model_config: dict[str, Any], device: torch.device
) -> nn.Module:
    """Load a pre-trained model from disk.

    Args:
        model_path: Path to saved model weights (.pt file)
        model_config: Model configuration for architecture
        device: Torch device

    Returns:
        Loaded model
    """
    model = create_model(model_config).to(device)

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model


def save_model(model: nn.Module, save_path: str):
    """Save model weights to disk.

    Args:
        model: Model to save
        save_path: Path to save weights
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model to {save_path}")


def setup_logging(level: str = "info"):
    """Configure logging."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=level_map.get(level.lower(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_run_dir(base_dir: str = "results") -> Path:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_device(config: dict[str, Any]) -> torch.device:
    """Get torch device."""
    device_str = config.get("device", None)

    if device_str is not None and device_str not in ("auto", ""):
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def run_mnist_experiment(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run MNIST classification experiment.

    Reproduces Examples 1 (binary) and Figure 4-5 from the paper.
    """
    logger.info("Running MNIST Classification Experiment")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    options = config.get("options", {})

    # Determine binary or multi-class
    digits = dataset_config.get("digits", [1, 9])
    n_classes = len(digits)
    dataset_name = "mnist_binary" if n_classes == 2 else "mnist_multi"

    logger.info(f"Classifying digits: {digits}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        dataset_name,
        {**dataset_config, "batch_size": training_config.get("batch_size", 256)},
        seed=config.get("seed", 42),
    )

    # Determine input dimension
    image_size = dataset_config.get("image_size", 16)
    input_dim = image_size * image_size

    # Create model
    model_config["input_dim"] = input_dim
    model_config["n_outputs"] = n_classes
    model = create_model(model_config).to(device)

    logger.info(f"Model: {type(model).__name__}")

    # Train
    results = train_model(
        model,
        train_loader,
        test_loader,
        training_config,
        device,
        save_path=str(run_dir / "model.pt") if options.get("save_model") else None,
    )

    logger.info(f"Best accuracy: {results['best_accuracy']:.3f}")

    return {"training": results, "model": model, "test_loader": test_loader}


def run_attack_experiment(
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    model: torch.nn.Module = None,
    test_loader=None,
) -> dict[str, Any]:
    """Run adversarial attack experiment (single run).

    Evaluates one attack method at one epsilon value.
    For sweeps, use scripts/figure7_robustness.sh

    Reproduces Figures 6-10 from the paper.
    """
    logger.info("Running Adversarial Attack Experiment")

    attack_config = config.get("attack", {})
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    options = config.get("options", {})

    # Determine model and data dimensions
    digits = dataset_config.get("digits", [1, 9])
    image_size = dataset_config.get("image_size", 16)
    model_config["input_dim"] = image_size * image_size
    model_config["n_outputs"] = len(digits)

    # Load model: prefer explicit path, then run_dir, then create new
    if model is None:
        load_path = options.get("load_model")
        actual_path = find_model_path(load_path)
        if actual_path:
            model = load_model(actual_path, model_config, device)
        elif (run_dir / "model.pt").exists():
            model = load_model(str(run_dir / "model.pt"), model_config, device)
        else:
            logger.warning("No pre-trained model found. Creating new model.")
            model = create_model(model_config).to(device)

    if test_loader is None:
        dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"
        _, test_loader = create_dataloaders(
            dataset_name, dataset_config, seed=config.get("seed", 42)
        )

    model.eval()

    # Attack parameters (single values for this run)
    attack_method = attack_config.get("method", "bim")
    epsilon = attack_config.get("epsilon", 0.1)
    num_iter = attack_config.get("num_iter", 10)
    alpha = attack_config.get("alpha", None)

    logger.info(f"Attack: {attack_method.upper()}, ε={epsilon}, iterations={num_iter}")

    # Generate adversarial examples
    clean_samples, clean_labels, adv_samples, adv_preds = generate_adversarial_examples(
        model,
        test_loader,
        attack_method,
        epsilon=epsilon,
        num_iter=num_iter,
        alpha=alpha,
        device=device,
        max_samples=attack_config.get("max_samples", 1000),
    )

    # Get clean predictions
    with torch.no_grad():
        clean_outputs = model(clean_samples.to(device))
        clean_preds = clean_outputs.argmax(dim=1).cpu()

    # Compute metrics
    fidelities = compute_fidelity(clean_samples, adv_samples)
    avg_fidelity = fidelities.mean().item()

    attack_results = evaluate_attack_success(
        clean_labels, adv_preds, clean_predictions=clean_preds
    )

    logger.info(f"Fooling rate: {attack_results['fooling_rate']:.3f}")
    logger.info(f"Adversarial accuracy: {attack_results['adversarial_accuracy']:.3f}")
    logger.info(f"Average fidelity: {avg_fidelity:.3f}")

    results = {
        "attack_method": attack_method,
        "epsilon": epsilon,
        "fooling_rate": attack_results["fooling_rate"],
        "adversarial_accuracy": attack_results["adversarial_accuracy"],
        "average_fidelity": avg_fidelity,
    }

    # Evaluate robustness across epsilons
    if options.get("evaluate_robustness", True):
        epsilons = attack_config.get("epsilon_range", [0.01, 0.05, 0.1, 0.2])
        robustness = evaluate_robustness(
            model,
            test_loader,
            attack_methods=[attack_method],
            epsilons=epsilons,
            num_iter=num_iter,
            device=device,
        )
        results["robustness"] = robustness

    return results


def run_defense_experiment(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run adversarial training defense experiment.

    Reproduces Figure 16 from the paper.
    """
    logger.info("Running Adversarial Training Defense Experiment")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    defense_config = config.get("defense", {})

    # Create dataloaders
    digits = dataset_config.get("digits", [1, 9])
    dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"

    train_loader, test_loader = create_dataloaders(
        dataset_name, dataset_config, seed=config.get("seed", 42)
    )

    # Create fresh model
    image_size = dataset_config.get("image_size", 16)
    model_config["input_dim"] = image_size * image_size
    model_config["n_outputs"] = len(digits)
    model = create_model(model_config).to(device)

    # Adversarial training
    trainer = AdversarialTrainer(
        model, train_loader, test_loader, defense_config, device
    )

    epochs = defense_config.get("epochs", 100)
    results = trainer.train(epochs=epochs, verbose=True, eval_attack=True)

    logger.info(f"Final clean accuracy: {results['final_clean_accuracy']:.3f}")
    logger.info(
        f"Final adversarial accuracy: {results['final_adversarial_accuracy']:.3f}"
    )

    return results


def run_ising_experiment(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run Ising model phase classification experiment.

    Reproduces Section III.D from the paper.
    """
    logger.info("Running Ising Model Phase Classification Experiment")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    options = config.get("options", {})

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        "ising", dataset_config, seed=config.get("seed", 42)
    )

    # Create model
    n_spins = dataset_config.get("n_spins", 8)
    state_dim = 2**n_spins
    model_config["type"] = "ising_quantum"
    model_config["state_dim"] = state_dim
    model_config["n_outputs"] = 2

    model = create_model(model_config).to(device)

    # Train
    results = train_model(
        model,
        train_loader,
        test_loader,
        training_config,
        device,
        save_path=str(run_dir / "ising_model.pt")
        if options.get("save_model")
        else None,
    )

    logger.info(f"Best accuracy: {results['best_accuracy']:.3f}")

    return {"training": results, "model": model, "test_loader": test_loader}


def run_topological_experiment(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run QAH topological phase classification experiment.

    Reproduces Section III.C from the paper - "Quantum adversarial
    learning topological phases of matter".
    """
    logger.info("Running Topological Phase (QAH) Classification Experiment")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    options = config.get("options", {})

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        "qah", dataset_config, seed=config.get("seed", 42)
    )

    # Determine input dimension from momentum resolution
    momentum_res = dataset_config.get("momentum_resolution", 20)
    input_dim = momentum_res * momentum_res

    # Create model - use amplitude encoding
    model_config["type"] = model_config.get("type", "amplitude_quantum")
    if model_config["type"] in ("hybrid_quantum", "hybrid_photonic"):
        model_config["type"] = "amplitude_quantum"  # Use amplitude encoding
    model_config["input_dim"] = input_dim
    model_config["n_outputs"] = 2

    model = create_model(model_config).to(device)

    logger.info(f"Model: {type(model).__name__}, input_dim={input_dim}")

    # Train
    results = train_model(
        model,
        train_loader,
        test_loader,
        training_config,
        device,
        save_path=str(run_dir / "topological_model.pt")
        if options.get("save_model")
        else None,
    )

    logger.info(f"Best accuracy: {results['best_accuracy']:.3f}")

    return {"training": results, "model": model, "test_loader": test_loader}


def run_transfer_experiment(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run black-box transfer attack experiment.

    Reproduces Section III.B.3 "Black-box attack: transferability"
    and Table III from the paper.
    """
    from .attacks import run_transfer_attack_experiment

    logger.info("Running Black-box Transfer Attack Experiment")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    transfer_config = config.get("transfer", {})
    options = config.get("options", {})

    # Create dataloaders
    digits = dataset_config.get("digits", [1, 9])
    dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"

    train_loader, test_loader = create_dataloaders(
        dataset_name, dataset_config, seed=config.get("seed", 42)
    )

    # Determine dimensions
    image_size = dataset_config.get("image_size", 16)
    input_dim = image_size * image_size
    n_outputs = len(digits)

    # Load or create quantum model (target)
    target_model_path = options.get("target_model")

    if target_model_path and Path(target_model_path).exists():
        # Load pre-trained amplitude-encoded quantum model
        logger.info(f"Loading pre-trained quantum model from {target_model_path}")

        # Use amplitude_quantum config (same as train_quantum.json)
        quantum_config = {
            "type": "amplitude_quantum",
            "input_dim": input_dim,
            "n_outputs": n_outputs,
            "hidden_dims": model_config.get("hidden_dims", [128, 64]),
            "n_modes": model_config.get("n_modes", 8),
            "n_photons": model_config.get("n_photons", 2),
            "n_layers": model_config.get("n_layers", 2),
        }
        quantum_model = create_model(quantum_config).to(device)
        quantum_model.load_state_dict(
            torch.load(target_model_path, map_location=device, weights_only=True)
        )
        quantum_model.eval()

        train_results = {"loaded_from": str(target_model_path)}
    else:
        # Train new quantum model if no pre-trained model available
        logger.info("Training quantum classifier (target model)...")
        model_config["input_dim"] = input_dim
        model_config["n_outputs"] = n_outputs
        # Default to amplitude encoding for transfer experiments
        if model_config.get("type") in ("hybrid_quantum", "hybrid_photonic"):
            model_config["type"] = "amplitude_quantum"

        quantum_model = create_model(model_config).to(device)
        train_results = train_model(
            quantum_model, train_loader, test_loader, training_config, device
        )

    # Run transfer attacks with CNN and FNN surrogates
    all_results = {"quantum_training": train_results}

    for surrogate_type in transfer_config.get("surrogate_types", ["cnn", "fnn"]):
        logger.info(
            f"\n--- Transfer attack with {surrogate_type.upper()} surrogate ---"
        )

        transfer_results = run_transfer_attack_experiment(
            quantum_model=quantum_model,
            train_loader=train_loader,
            test_loader=test_loader,
            surrogate_type=surrogate_type,
            input_dim=input_dim,
            image_size=image_size,
            n_outputs=n_outputs,
            attack_methods=transfer_config.get(
                "attack_methods", ["bim", "fgsm", "mim"]
            ),
            epsilon=transfer_config.get("epsilon", 0.1),
            num_iter=transfer_config.get("num_iter", 10),
            epochs=transfer_config.get("surrogate_epochs", 20),
            device=device,
        )

        all_results[surrogate_type] = transfer_results

    # Save results table (similar to Table III)
    _save_transfer_results_table(all_results, run_dir)

    return all_results


def _save_transfer_results_table(results: dict[str, Any], run_dir: Path):
    """Save transfer attack results as a formatted table."""
    lines = [
        "Black-box Transfer Attack Results",
        "=" * 50,
        "",
        "Reproduces Table III from Lu et al. (2020)",
        "",
        f"{'Attack':<15} {'α_C^adv':<10} {'α_C - α_C^adv':<15} {'α_Q^adv':<10} {'α_Q - α_Q^adv':<15}",
        "-" * 65,
    ]

    for surrogate_type in ["cnn", "fnn"]:
        if surrogate_type not in results:
            continue

        lines.append(f"\n{surrogate_type.upper()} Surrogate:")

        for attack_name, attack_results in results[surrogate_type].items():
            if attack_name == "surrogate_type":
                continue

            ac_adv = attack_results["surrogate_adversarial_accuracy"]
            ac_drop = attack_results["accuracy_drop_surrogate"]
            aq_adv = attack_results["target_adversarial_accuracy"]
            aq_drop = attack_results["accuracy_drop_target"]

            lines.append(
                f"{attack_name.upper():<15} {ac_adv * 100:<10.1f} {ac_drop * 100:<15.1f} "
                f"{aq_adv * 100:<10.1f} {aq_drop * 100:<15.1f}"
            )

    with open(run_dir / "transfer_results.txt", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved transfer results to {run_dir / 'transfer_results.txt'}")


def run_noise_experiment(
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    model: nn.Module = None,
    test_loader: torch.utils.data.DataLoader = None,
) -> dict[str, Any]:
    """Run noise vs adversarial comparison experiment.

    NOTE: This runs a full sweep over multiple epsilon values.
    For single-run evaluation, use experiment=noise_eval instead.

    Reproduces Figure 11 from the paper, showing that adversarial
    perturbations are much more effective than random noise.

    For photonic systems, we also compare against photon loss noise.

    Args:
        config: Experiment configuration
        run_dir: Output directory
        device: Torch device
        model: Pre-trained model (optional)
        test_loader: Test data loader (optional)

    Returns:
        Comparison results
    """
    from .attacks import run_noise_comparison_experiment

    logger.info("Running Noise vs Adversarial Comparison Experiment")
    logger.info("Reproduces Figure 11 from Lu et al. (2020)")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    options = config.get("options", {})

    # Load or create model and data if not provided
    if model is None or test_loader is None:
        # Create dataloaders
        digits = dataset_config.get("digits", [1, 9])
        dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"

        train_loader, test_loader = create_dataloaders(
            dataset_name, dataset_config, seed=config.get("seed", 42)
        )

        # Try to load model first
        image_size = dataset_config.get("image_size", 16)
        input_dim = image_size * image_size
        model_config["input_dim"] = input_dim
        model_config["n_outputs"] = len(digits)

        load_path = options.get("load_model")
        actual_path = find_model_path(load_path)
        if actual_path:
            model = load_model(actual_path, model_config, device)
        else:
            model = create_model(model_config).to(device)
            logger.info("Training quantum classifier...")
            from .training import train_model

            train_model(model, train_loader, test_loader, training_config, device)

    # Run noise comparison
    results = run_noise_comparison_experiment(
        model=model, test_loader=test_loader, config=config, device=device
    )

    # Save results
    import json

    results_serializable = {
        k: v if not isinstance(v, list) else [float(x) for x in v]
        for k, v in results.items()
    }
    with open(run_dir / "noise_comparison_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    return results


def run_noise_eval(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Evaluate model with ONE noise type at ONE level (single run).

    Use this for shell script orchestration of Figure 11.

    Args:
        config: Experiment configuration with noise.epsilon and noise.type
        run_dir: Output directory
        device: Torch device

    Returns:
        Single evaluation result
    """
    from .attacks import add_random_noise, evaluate_with_photon_loss

    logger.info("Running Single Noise Evaluation")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    noise_config = config.get("noise", {})
    options = config.get("options", {})

    # Get noise parameters
    epsilon = noise_config.get("epsilon", 0.1)
    noise_type = noise_config.get("type", "uniform")

    logger.info(f"Noise type: {noise_type}, epsilon/rate: {epsilon}")

    # Load data
    digits = dataset_config.get("digits", [1, 9])
    dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"
    _, test_loader = create_dataloaders(
        dataset_name, dataset_config, seed=config.get("seed", 42)
    )

    # Load model
    image_size = dataset_config.get("image_size", 16)
    model_config["input_dim"] = image_size * image_size
    model_config["n_outputs"] = len(digits)

    load_path = options.get("load_model")
    if not load_path:
        raise ValueError("noise_eval requires options.load_model")

    actual_path = find_model_path(load_path)
    if not actual_path:
        raise FileNotFoundError(
            f"Model not found at {load_path} or in run subdirectories"
        )

    model = load_model(actual_path, model_config, device)
    model.eval()

    # Evaluate with noise
    if noise_type == "photon_loss":
        results = evaluate_with_photon_loss(model, test_loader, epsilon, device)
    else:
        # Random noise evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                x_noisy = add_random_noise(data, epsilon, noise_type)
                outputs = model(x_noisy)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        results = {
            "accuracy": correct / total,
            "epsilon": epsilon,
            "noise_type": noise_type,
        }

    logger.info(
        f"Accuracy with {noise_type} noise (ε={epsilon}): {results['accuracy']:.4f}"
    )

    # Save result
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_transfer_eval(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Evaluate transfer attack: generate adversarial from surrogate, test on target.

    Single run for one attack method. Use shell script for Table III.

    Args:
        config: Configuration with surrogate_model and target_model paths
        run_dir: Output directory
        device: Torch device

    Returns:
        Transfer attack results
    """
    from .attacks import transfer_attack

    logger.info("Running Transfer Attack Evaluation")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    attack_config = config.get("attack", {})
    options = config.get("options", {})

    # Load data
    digits = dataset_config.get("digits", [1, 9])
    dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"
    _, test_loader = create_dataloaders(
        dataset_name, dataset_config, seed=config.get("seed", 42)
    )

    # Get model paths
    surrogate_path = options.get("surrogate_model")
    target_path = options.get("target_model")

    if not surrogate_path or not target_path:
        raise ValueError(
            "transfer_eval requires options.surrogate_model and options.target_model"
        )

    # Determine surrogate model type from path
    image_size = dataset_config.get("image_size", 16)
    input_dim = image_size * image_size
    n_outputs = len(digits)

    # Load surrogate (infer type from config or path)
    surrogate_type = options.get("surrogate_type", "fnn")
    if surrogate_type == "cnn":
        surrogate_config = {
            "type": "cnn",
            "image_size": image_size,
            "n_outputs": n_outputs,
        }
    else:
        surrogate_config = {
            "type": "fnn",
            "input_dim": input_dim,
            "n_outputs": n_outputs,
        }

    surrogate = load_model(surrogate_path, surrogate_config, device)

    # Load target quantum model
    target_config = model_config.copy()
    target_config["input_dim"] = input_dim
    target_config["n_outputs"] = n_outputs
    target = load_model(target_path, target_config, device)

    # Run transfer attack
    attack_method = attack_config.get("method", "bim")
    epsilon = attack_config.get("epsilon", 0.1)
    num_iter = attack_config.get("num_iter", 10)

    logger.info(
        f"Transfer: {surrogate_type} -> quantum, attack={attack_method}, ε={epsilon}"
    )

    results = transfer_attack(
        surrogate_model=surrogate,
        target_model=target,
        dataloader=test_loader,
        attack_method=attack_method,
        epsilon=epsilon,
        num_iter=num_iter,
        device=device,
    )

    logger.info(f"Surrogate acc drop: {results['accuracy_drop_surrogate']:.3f}")
    logger.info(f"Target acc drop: {results['accuracy_drop_target']:.3f}")
    logger.info(f"Transfer rate: {results['transfer_rate']:.3f}")

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_comparison_experiment(
    config: dict[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Run photonic vs gate-based comparison experiment.

    Trains both MerLin (photonic) and PennyLane (gate-based) classifiers
    on the same dataset and compares their performance under adversarial attack.
    """
    logger.info("Running Photonic vs Gate-Based Comparison Experiment")

    dataset_config = config.get("dataset", {})
    training_config = config.get("training", {})
    attack_config = config.get("attack", {})
    comparison_config = config.get("comparison", {})
    options = config.get("options", {})

    # Create dataloaders
    digits = dataset_config.get("digits", [1, 9])
    dataset_name = "mnist_binary" if len(digits) == 2 else "mnist_multi"

    train_loader, test_loader = create_dataloaders(
        dataset_name, dataset_config, seed=config.get("seed", 42)
    )

    # Determine dimensions
    image_size = dataset_config.get("image_size", 16)
    input_dim = image_size * image_size
    n_outputs = len(digits)

    results = {"comparison": {}}

    # ==========================================================================
    # Train Photonic (MerLin) Model
    # ==========================================================================
    logger.info("\n=== Training Photonic (MerLin) Classifier ===")

    photonic_config = {
        "type": "hybrid_quantum",
        "input_dim": input_dim,
        "n_outputs": n_outputs,
        "hidden_dims": comparison_config.get("hidden_dims", [128, 64]),
        "n_modes": comparison_config.get("n_modes", 8),
        "n_photons": comparison_config.get("n_photons", 2),
        "n_layers": comparison_config.get("n_layers_photonic", 2),
    }

    photonic_model = create_model(photonic_config).to(device)
    photonic_results = train_model(
        photonic_model,
        train_loader,
        test_loader,
        training_config,
        device,
        save_path=str(run_dir / "photonic_model.pt")
        if options.get("save_model")
        else None,
    )
    results["photonic"] = {
        "training": photonic_results,
        "clean_accuracy": photonic_results["best_accuracy"],
    }

    # ==========================================================================
    # Train Gate-Based (PennyLane) Model
    # ==========================================================================
    logger.info("\n=== Training Gate-Based (PennyLane) Classifier ===")

    try:
        gate_config = {
            "type": "hybrid_gate",
            "input_dim": input_dim,
            "n_outputs": n_outputs,
            "hidden_dims": comparison_config.get("hidden_dims", [128, 64]),
            "n_qubits": comparison_config.get("n_qubits", 4),
            "n_layers": comparison_config.get("n_layers_gate", 6),
        }

        gate_model = create_model(gate_config).to(device)
        gate_results = train_model(
            gate_model,
            train_loader,
            test_loader,
            training_config,
            device,
            save_path=str(run_dir / "gate_model.pt")
            if options.get("save_model")
            else None,
        )
        results["gate"] = {
            "training": gate_results,
            "clean_accuracy": gate_results["best_accuracy"],
        }
    except ImportError as e:
        logger.warning(f"PennyLane not available, skipping gate-based model: {e}")
        results["gate"] = {"error": "PennyLane not available"}

    # ==========================================================================
    # Adversarial Attack Comparison
    # ==========================================================================
    if options.get("run_attack", True):
        logger.info("\n=== Adversarial Attack Comparison ===")

        attack_method = attack_config.get("method", "bim")
        epsilon = attack_config.get("epsilon", 0.1)
        num_iter = attack_config.get("num_iter", 10)

        # Attack photonic model
        logger.info("Attacking photonic model...")
        photonic_attack = _evaluate_model_under_attack(
            photonic_model, test_loader, attack_method, epsilon, num_iter, device
        )
        results["photonic"]["adversarial"] = photonic_attack

        # Attack gate model (if available)
        if "training" in results.get("gate", {}):
            logger.info("Attacking gate-based model...")
            gate_attack = _evaluate_model_under_attack(
                gate_model, test_loader, attack_method, epsilon, num_iter, device
            )
            results["gate"]["adversarial"] = gate_attack

    # ==========================================================================
    # Save Comparison Results
    # ==========================================================================
    _save_comparison_results(results, run_dir)

    return results


def _evaluate_model_under_attack(
    model: nn.Module,
    test_loader,
    attack_method: str,
    epsilon: float,
    num_iter: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model under adversarial attack."""
    from .attacks import mim_attack, pgd_attack

    model.eval()

    attack_fn = {
        "fgsm": lambda m, x, y, e: fgsm_attack(m, x, y, e),
        "bim": lambda m, x, y, e: bim_attack(m, x, y, e, num_iter=num_iter),
        "pgd": lambda m, x, y, e: pgd_attack(m, x, y, e, num_iter=num_iter),
        "mim": lambda m, x, y, e: mim_attack(m, x, y, e, num_iter=num_iter),
    }.get(attack_method.lower())

    clean_correct = 0
    adv_correct = 0
    total = 0

    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_out = model(data)
            clean_pred = clean_out.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()

        # Generate adversarial examples
        with torch.enable_grad():
            adv_data = attack_fn(model, data, labels, epsilon)

        # Adversarial accuracy
        with torch.no_grad():
            adv_out = model(adv_data)
            adv_pred = adv_out.argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

        total += labels.size(0)

    return {
        "clean_accuracy": clean_correct / total,
        "adversarial_accuracy": adv_correct / total,
        "accuracy_drop": (clean_correct - adv_correct) / total,
    }


def _save_comparison_results(results: dict[str, Any], run_dir: Path):
    """Save comparison results to file."""
    lines = [
        "Photonic vs Gate-Based Quantum Classifier Comparison",
        "=" * 55,
        "",
    ]

    # Photonic results
    if "photonic" in results:
        lines.append("PHOTONIC (MerLin) Model:")
        lines.append(
            f"  Clean Accuracy: {results['photonic'].get('clean_accuracy', 0):.1%}"
        )
        if "adversarial" in results["photonic"]:
            adv = results["photonic"]["adversarial"]
            lines.append(f"  Adversarial Accuracy: {adv['adversarial_accuracy']:.1%}")
            lines.append(f"  Accuracy Drop: {adv['accuracy_drop']:.1%}")
        lines.append("")

    # Gate-based results
    if "gate" in results and "error" not in results["gate"]:
        lines.append("GATE-BASED (PennyLane) Model:")
        lines.append(
            f"  Clean Accuracy: {results['gate'].get('clean_accuracy', 0):.1%}"
        )
        if "adversarial" in results["gate"]:
            adv = results["gate"]["adversarial"]
            lines.append(f"  Adversarial Accuracy: {adv['adversarial_accuracy']:.1%}")
            lines.append(f"  Accuracy Drop: {adv['accuracy_drop']:.1%}")
    elif "gate" in results:
        lines.append("GATE-BASED Model: Not available (PennyLane not installed)")

    with open(run_dir / "comparison_results.txt", "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved comparison results to {run_dir / 'comparison_results.txt'}")


def _plot_comparison_results(results: dict[str, Any], run_dir: Path):
    """Plot comparison bar chart."""
    import matplotlib.pyplot as plt

    models = []
    clean_accs = []
    adv_accs = []

    if "photonic" in results and "adversarial" in results["photonic"]:
        models.append("Photonic\n(MerLin)")
        clean_accs.append(results["photonic"]["clean_accuracy"])
        adv_accs.append(results["photonic"]["adversarial"]["adversarial_accuracy"])

    if "gate" in results and "adversarial" in results.get("gate", {}):
        models.append("Gate-Based\n(PennyLane)")
        clean_accs.append(results["gate"]["clean_accuracy"])
        adv_accs.append(results["gate"]["adversarial"]["adversarial_accuracy"])

    if not models:
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, clean_accs, width, label="Clean", color="steelblue")
    bars2 = ax.bar(x + width / 2, adv_accs, width, label="Adversarial", color="coral")

    ax.set_ylabel("Accuracy")
    ax.set_title("Photonic vs Gate-Based Quantum Classifiers")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(run_dir / "comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot to {run_dir / 'comparison_plot.png'}")


def main(config: dict[str, Any]) -> str:
    """Main entry point.

    Args:
        config: Full configuration dictionary

    Returns:
        Path to run directory
    """
    # Setup
    logging_config = config.get("logging", {})
    setup_logging(logging_config.get("level", "info"))

    seed = config.get("seed", 42)
    set_seed(seed)

    # Create run directory
    outdir = config.get("outdir", "results")
    run_dir = create_run_dir(outdir)
    logger.info(f"Output directory: {run_dir}")

    # Save config
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Device setup
    device = get_device(config)

    # Run experiment
    experiment = config.get("experiment", "mnist")
    results = {}

    if experiment == "mnist":
        results = run_mnist_experiment(config, run_dir, device)

        # Optionally run attack after training
        if config.get("options", {}).get("run_attack", True):
            attack_results = run_attack_experiment(
                config,
                run_dir,
                device,
                model=results.get("model"),
                test_loader=results.get("test_loader"),
            )
            results["attack"] = attack_results

    elif experiment == "attack":
        results = run_attack_experiment(config, run_dir, device)

    elif experiment == "defense":
        results = run_defense_experiment(config, run_dir, device)

    elif experiment == "ising":
        results = run_ising_experiment(config, run_dir, device)

        # Optionally run attack
        if config.get("options", {}).get("run_attack", True):
            attack_results = run_attack_experiment(
                config,
                run_dir,
                device,
                model=results.get("model"),
                test_loader=results.get("test_loader"),
            )
            results["attack"] = attack_results

    elif experiment == "topological" or experiment == "qah":
        results = run_topological_experiment(config, run_dir, device)

        # Optionally run attack
        if config.get("options", {}).get("run_attack", True):
            attack_results = run_attack_experiment(
                config,
                run_dir,
                device,
                model=results.get("model"),
                test_loader=results.get("test_loader"),
            )
            results["attack"] = attack_results

    elif experiment == "transfer":
        results = run_transfer_experiment(config, run_dir, device)

    elif experiment == "comparison":
        results = run_comparison_experiment(config, run_dir, device)

    elif experiment == "noise" or experiment == "noise_comparison":
        results = run_noise_experiment(config, run_dir, device)

    elif experiment == "noise_eval":
        results = run_noise_eval(config, run_dir, device)

    elif experiment == "transfer_eval":
        results = run_transfer_eval(config, run_dir, device)

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Save results summary
    summary = {"experiment": experiment}

    # Handle training results (might be nested or at top level)
    if "training" in results:
        summary["best_accuracy"] = results["training"].get("best_accuracy", 0)
        summary["final_accuracy"] = results["training"].get("final_accuracy", 0)
    elif "best_accuracy" in results:
        summary["best_accuracy"] = results.get("best_accuracy", 0)
        summary["final_accuracy"] = results.get("final_accuracy", 0)

    # Handle defense experiment results
    if "final_clean_accuracy" in results:
        summary["final_clean_accuracy"] = results.get("final_clean_accuracy", 0)
        summary["final_adversarial_accuracy"] = results.get(
            "final_adversarial_accuracy", 0
        )

    # Handle attack results (might be nested under "attack" or at top level)
    if "attack" in results:
        summary["fooling_rate"] = results["attack"].get("fooling_rate", 0)
        summary["adversarial_accuracy"] = results["attack"].get(
            "adversarial_accuracy", 0
        )
        summary["average_fidelity"] = results["attack"].get("average_fidelity", 0)
    elif "adversarial_accuracy" in results:
        # Standalone attack experiment
        summary["fooling_rate"] = results.get("fooling_rate", 0)
        summary["adversarial_accuracy"] = results.get("adversarial_accuracy", 0)
        summary["average_fidelity"] = results.get("average_fidelity", 0)

    with open(run_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Mark completion
    (run_dir / "done.txt").write_text(f"Completed at {datetime.now().isoformat()}")

    logger.info(f"Experiment complete! Results saved to {run_dir}")

    return str(run_dir)


def train_and_evaluate(config: dict[str, Any], run_dir: Path) -> None:
    """Entry point for the shared repository runner.

    Args:
        config: Configuration dictionary
        run_dir: Output directory
    """
    logging_config = config.get("logging", {})
    setup_logging(logging_config.get("level", "info"))

    logger.info(f"Output directory: {run_dir}")

    device = get_device(config)
    experiment = config.get("experiment", "mnist")

    results = {}

    if experiment == "mnist":
        results = run_mnist_experiment(config, run_dir, device)
        if config.get("options", {}).get("run_attack", True):
            attack_results = run_attack_experiment(
                config,
                run_dir,
                device,
                model=results.get("model"),
                test_loader=results.get("test_loader"),
            )
            results["attack"] = attack_results

    elif experiment == "attack":
        results = run_attack_experiment(config, run_dir, device)

    elif experiment == "defense":
        results = run_defense_experiment(config, run_dir, device)

    elif experiment == "ising":
        results = run_ising_experiment(config, run_dir, device)
        if config.get("options", {}).get("run_attack", True):
            attack_results = run_attack_experiment(
                config,
                run_dir,
                device,
                model=results.get("model"),
                test_loader=results.get("test_loader"),
            )
            results["attack"] = attack_results

    elif experiment == "topological" or experiment == "qah":
        results = run_topological_experiment(config, run_dir, device)
        if config.get("options", {}).get("run_attack", True):
            attack_results = run_attack_experiment(
                config,
                run_dir,
                device,
                model=results.get("model"),
                test_loader=results.get("test_loader"),
            )
            results["attack"] = attack_results

    elif experiment == "transfer":
        results = run_transfer_experiment(config, run_dir, device)

    elif experiment == "noise" or experiment == "noise_comparison":
        results = run_noise_experiment(config, run_dir, device)

    elif experiment == "noise_eval":
        results = run_noise_eval(config, run_dir, device)

    elif experiment == "transfer_eval":
        results = run_transfer_eval(config, run_dir, device)

    # Save summary with actual results
    summary = {"experiment": experiment, "completed": True}

    # Handle training results (might be nested or at top level)
    if "training" in results:
        summary["best_accuracy"] = results["training"].get("best_accuracy", 0)
        summary["final_accuracy"] = results["training"].get("final_accuracy", 0)
    elif "best_accuracy" in results:
        summary["best_accuracy"] = results.get("best_accuracy", 0)
        summary["final_accuracy"] = results.get("final_accuracy", 0)

    # Handle defense experiment results
    if "final_clean_accuracy" in results:
        summary["final_clean_accuracy"] = results.get("final_clean_accuracy", 0)
        summary["final_adversarial_accuracy"] = results.get(
            "final_adversarial_accuracy", 0
        )

    # Handle attack results (might be nested under "attack" or at top level)
    if "attack" in results:
        summary["fooling_rate"] = results["attack"].get("fooling_rate", 0)
        summary["adversarial_accuracy"] = results["attack"].get(
            "adversarial_accuracy", 0
        )
        summary["average_fidelity"] = results["attack"].get("average_fidelity", 0)
    elif "adversarial_accuracy" in results:
        # Standalone attack experiment
        summary["fooling_rate"] = results.get("fooling_rate", 0)
        summary["adversarial_accuracy"] = results.get("adversarial_accuracy", 0)
        summary["average_fidelity"] = results.get("average_fidelity", 0)

    with open(run_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    (run_dir / "done.txt").write_text(f"Completed at {datetime.now().isoformat()}")
    logger.info("Experiment complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = json.load(f)
        main(config)
    else:
        # Default example
        example_config = {
            "seed": 42,
            "experiment": "mnist",
            "dataset": {"digits": [1, 9], "image_size": 16},
            "model": {
                "type": "hybrid_quantum",
                "n_modes": 8,
                "n_photons": 2,
                "n_layers": 2,
            },
            "training": {"epochs": 10, "batch_size": 256, "learning_rate": 0.005},
            "attack": {"method": "bim", "epsilon": 0.1, "num_iter": 3},
            "options": {"run_attack": True},
        }
        main(example_config)
