"""
Runner Module
=============

Main entry point for the MerLin CLI. Handles configuration loading,
experiment execution, and results saving.

Supports both:
- MerLin (photonic): Primary implementation
- PennyLane (qubit): Reference for comparison
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .datasets import SpiralDataset, create_dataloaders
from .models import ClassicalBaseline, CQTransferModel, HybridModel
from .training import train_model
from .visualization import (
    plot_comparison,
    plot_image_predictions,
    plot_spiral_classification,
    plot_training_curves,
)

logger = logging.getLogger(__name__)


def _save_training_artifacts(results: Dict[str, Any], run_dir: Path, experiment: str) -> None:
    """Save training artifacts (losses CSV, best epoch info).

    Args:
        results: Training results dict
        run_dir: Output directory
        experiment: Experiment name
    """
    # Get the primary results (quantum if exists, otherwise direct results)
    primary_results = results.get("quantum", results)

    # Save losses CSV for detailed training history
    try:
        history = primary_results.get("history", {})

        # History is a dict with lists: {"train_loss": [...], "train_acc": [...], ...}
        if history and isinstance(history, dict):
            train_losses = history.get("train_loss", [])
            train_accs = history.get("train_acc", [])
            test_losses = history.get("test_loss", [])
            test_accs = history.get("test_acc", [])

            if train_losses:  # Only write if we have data
                csv_path = run_dir / "losses.csv"
                with csv_path.open("w") as f:
                    f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")
                    for i in range(len(train_losses)):
                        f.write(f"{i + 1},"
                                f"{train_losses[i]:.6f},"
                                f"{train_accs[i] if i < len(train_accs) else 0:.4f},"
                                f"{test_losses[i] if i < len(test_losses) else 0:.6f},"
                                f"{test_accs[i] if i < len(test_accs) else 0:.4f}\n")
                logger.info(f"Saved losses CSV: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write losses.csv: {e}")

    # Save best epoch metadata
    try:
        # Find best epoch from test accuracy history
        history = primary_results.get("history", {})
        test_accs = history.get("test_acc", []) if isinstance(history, dict) else []
        best_epoch = (test_accs.index(max(test_accs)) + 1) if test_accs else 0
        best_accuracy = primary_results.get("best_accuracy", 0)

        best_path = run_dir / "best_epoch.txt"
        best_path.write_text(
            f"best_epoch={best_epoch}\n"
            f"best_accuracy={best_accuracy:.6f}\n"
            f"experiment={experiment}\n"
        )
        logger.info(f"Saved best epoch info: {best_path}")
    except Exception as e:
        logger.warning(f"Failed to write best_epoch.txt: {e}")


def make_json_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to JSON-safe types.

    Handles torch.dtype, tuples, and other common non-serializable types
    that may be injected by the shared runtime.
    """
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        # Fallback: convert anything else to string
        return str(obj)


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get torch device with proper CUDA detection and diagnostics.

    Args:
        config: Configuration dictionary (may contain 'device' key)

    Returns:
        torch.device configured for best available hardware
    """
    device_str = config.get("device", None)

    # If device explicitly specified, use it
    if device_str is not None and device_str not in ("auto", ""):
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        return device

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU support
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        # Provide diagnostic info about why CUDA isn't available
        logger.warning("CUDA not available - running on CPU")
        logger.warning(f"  PyTorch version: {torch.__version__}")
        logger.warning(f"  PyTorch CUDA version: {torch.version.cuda}")
        if torch.version.cuda is None:
            logger.warning("  PyTorch was built WITHOUT CUDA support!")
            logger.warning("  Reinstall with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        else:
            logger.warning("  CUDA runtime may not be installed or GPU not detected")
            logger.warning("  Check: nvidia-smi")

    return device


def set_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def setup_logging(level: str = "info"):
    """Configure logging.

    Args:
        level: Logging level ('debug', 'info', 'warning', 'error')
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }

    logging.basicConfig(
        level=level_map.get(level.lower(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_run_dir(base_dir: str = "outdir") -> Path:
    """Create timestamped run directory.

    Args:
        base_dir: Base output directory

    Returns:
        Path to run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_spiral_experiment(
        config: Dict[str, Any],
        run_dir: Path,
        device: torch.device
) -> Dict[str, Any]:
    """Run Example 1: 2D spiral classification.

    Args:
        config: Experiment configuration
        run_dir: Output directory
        device: Torch device

    Returns:
        Experiment results
    """
    logger.info("Running Spiral Classification (Example 1)")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    options = config.get("options", {})

    # Determine backend
    backend = "merlin" if options.get("use_merlin", True) else "pennylane"
    logger.info(f"Using backend: {backend}")

    # Create dataset
    train_loader, test_loader = create_dataloaders(
        "spiral",
        {**dataset_config, "batch_size": training_config.get("batch_size", 10)},
        seed=config.get("seed", 42)
    )

    # Get raw data for visualization
    spiral_data = SpiralDataset(
        n_samples=dataset_config.get("n_samples", 2200),
        seed=config.get("seed", 42)
    )
    X_all, y_all = spiral_data.X, spiral_data.y
    n_train = dataset_config.get("n_train", 2000)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    results = {}

    # Train quantum model
    logger.info(f"Training dressed quantum circuit ({backend})...")
    quantum_model = HybridModel(
        n_inputs=model_config.get("n_inputs", 2),
        n_outputs=model_config.get("n_outputs", 2),
        n_qubits=model_config.get("n_qubits", model_config.get("n_modes", 4)),
        q_depth=model_config.get("q_depth", 5),
        backend=backend,
        n_photons=model_config.get("n_photons", 2),
        computation_space=model_config.get("computation_space", "unbunched"),
        merlin_depth=model_config.get("merlin_depth", 1),
        scale_type=model_config.get("scale_type", "learned")
    ).to(device)

    quantum_results = train_model(
        quantum_model, train_loader, test_loader,
        training_config, device,
        save_path=str(run_dir / "quantum_model.pt") if options.get("save_model") else None
    )
    results["quantum"] = quantum_results

    # Generate figures
    if options.get("generate_figures", True):
        plot_spiral_classification(
            quantum_model, X_train, y_train, X_test, y_test,
            quantum_results["best_accuracy"],
            title="Dressed Quantum Circuit",
            save_path=str(run_dir / "fig2_spiral_quantum.png"),
            device=device
        )

        plot_training_curves(
            quantum_results["history"],
            title="Spiral - Quantum",
            save_path=str(run_dir / "training_curves_quantum.png")
        )

    # Train classical baseline if requested
    if options.get("compare_classical", True):
        logger.info("Training classical baseline...")
        classical_config = config.get("classical_baseline", {})

        classical_model = ClassicalBaseline(
            n_inputs=model_config.get("n_inputs", 2),
            n_outputs=model_config.get("n_outputs", 2),
            hidden_sizes=classical_config.get("hidden_sizes", [4]),
            activation=classical_config.get("activation", "tanh")
        ).to(device)

        classical_results = train_model(
            classical_model, train_loader, test_loader,
            training_config, device
        )
        results["classical"] = classical_results

        if options.get("generate_figures", True):
            plot_spiral_classification(
                classical_model, X_train, y_train, X_test, y_test,
                classical_results["best_accuracy"],
                title="Classical Network",
                save_path=str(run_dir / "fig2_spiral_classical.png"),
                device=device
            )

            plot_comparison(
                results["quantum"], results["classical"],
                title="Spiral: Quantum vs Classical",
                save_path=str(run_dir / "comparison.png")
            )

    logger.info(f"Quantum accuracy: {quantum_results['best_accuracy']:.3f}")
    if "classical" in results:
        logger.info(f"Classical accuracy: {results['classical']['best_accuracy']:.3f}")

    return results


def run_transfer_experiment(
        config: Dict[str, Any],
        run_dir: Path,
        device: torch.device,
        experiment_name: str
) -> Dict[str, Any]:
    """Run CQ transfer learning experiment (Examples 2 & 3).

    Args:
        config: Experiment configuration
        run_dir: Output directory
        device: Torch device
        experiment_name: Name of the experiment

    Returns:
        Experiment results
    """
    logger.info(f"Running CQ Transfer Learning: {experiment_name}")

    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    feature_config = config.get("feature_extractor", {})
    training_config = config.get("training", {})
    options = config.get("options", {})

    # Determine backend
    backend = "merlin" if options.get("use_merlin", True) else "pennylane"
    logger.info(f"Using backend: {backend}")

    # Determine dataset type
    dataset_name = dataset_config.get("name", "hymenoptera")
    if dataset_name == "cifar10":
        dataset_key = "cifar10"
    else:
        dataset_key = "hymenoptera"

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        dataset_key,
        {**dataset_config, "batch_size": training_config.get("batch_size", 4)},
        seed=config.get("seed", 42)
    )

    # Create CQ transfer model
    logger.info(f"Creating CQ transfer model ({backend})...")
    model = CQTransferModel(
        n_outputs=model_config.get("n_outputs", 2),
        n_qubits=model_config.get("n_qubits", model_config.get("n_modes", 4)),
        q_depth=model_config.get("q_depth", 6),
        feature_extractor=feature_config.get("model", "resnet18"),
        pretrained=feature_config.get("pretrained", True),
        freeze_extractor=feature_config.get("freeze", True),
        backend=backend,
        n_photons=model_config.get("n_photons", 2),
        computation_space=model_config.get("computation_space", "unbunched"),
        merlin_depth=model_config.get("merlin_depth", 1),
        scale_type=model_config.get("scale_type", "learned")
    ).to(device)

    # Train
    logger.info("Training CQ model...")
    results = train_model(
        model, train_loader, test_loader,
        training_config, device,
        save_path=str(run_dir / f"{experiment_name}_model.pt") if options.get("save_model") else None
    )

    # Generate figures
    if options.get("generate_figures", True):
        plot_training_curves(
            results["history"],
            title=f"{experiment_name} - CQ Transfer Learning",
            save_path=str(run_dir / f"{experiment_name}_training.png")
        )

        # Get class names
        if dataset_key == "hymenoptera":
            class_names = ["ants", "bees"]
        else:
            cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck']
            classes = dataset_config.get("classes", [3, 5])
            class_names = [cifar_classes[c] for c in classes]

        plot_image_predictions(
            model, test_loader, class_names,
            n_images=4,
            title=f"{experiment_name} Predictions",
            save_path=str(run_dir / f"{experiment_name}_predictions.png"),
            device=device
        )

    logger.info(f"Best accuracy: {results['best_accuracy']:.3f}")

    return results


def main(config: Dict[str, Any]) -> str:
    """Main entry point for the runner.

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
    outdir = config.get("outdir", "outdir")
    run_dir = create_run_dir(outdir)
    logger.info(f"Output directory: {run_dir}")

    # Save config snapshot (best effort - may fail if config has non-serializable types)
    try:
        with open(run_dir / "config_snapshot.json", "w") as f:
            json.dump(make_json_serializable(config), f, indent=2)
    except (TypeError, ValueError) as e:
        logger.warning(f"Could not save config snapshot: {e}")

    # Device setup with proper auto-detection
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Run experiment
    experiment = config.get("experiment", "spiral")

    if experiment == "spiral":
        results = run_spiral_experiment(config, run_dir, device)
    elif experiment == "hymenoptera":
        results = run_transfer_experiment(config, run_dir, device, "hymenoptera")
    elif experiment == "cifar_dogs_cats":
        results = run_transfer_experiment(config, run_dir, device, "cifar_dogs_cats")
    elif experiment == "cifar_planes_cars":
        results = run_transfer_experiment(config, run_dir, device, "cifar_planes_cars")
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Save summary results
    summary = {
        "best_accuracy": results.get("quantum", results).get("best_accuracy",
                                                             results.get("best_accuracy", 0)),
        "final_accuracy": results.get("quantum", results).get("final_accuracy",
                                                              results.get("final_accuracy", 0)),
        "total_time": results.get("quantum", results).get("total_time",
                                                          results.get("total_time", 0))
    }

    with open(run_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save training artifacts (losses CSV, best epoch info)
    _save_training_artifacts(results, run_dir, experiment)

    # Mark completion
    (run_dir / "done.txt").write_text(f"Completed at {datetime.now().isoformat()}")

    logger.info(f"Experiment complete! Results saved to {run_dir}")

    return str(run_dir)


def train_and_evaluate(config: Dict[str, Any], run_dir: Path) -> None:
    """Entry point for the shared MerLin CLI runner.

    This wrapper adapts the main() function to the signature expected
    by the shared repository runner: (config, run_dir) -> None

    Args:
        config: Configuration dictionary (already merged with defaults)
        run_dir: Output directory created by the shared runner
    """
    # Setup logging
    logging_config = config.get("logging", {})
    setup_logging(logging_config.get("level", "info"))

    logger.info(f"Output directory: {run_dir}")

    # Device setup with proper auto-detection
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Run experiment
    experiment = config.get("experiment", "spiral")

    if experiment == "spiral":
        results = run_spiral_experiment(config, run_dir, device)
    elif experiment == "hymenoptera":
        results = run_transfer_experiment(config, run_dir, device, "hymenoptera")
    elif experiment == "cifar_dogs_cats":
        results = run_transfer_experiment(config, run_dir, device, "cifar_dogs_cats")
    elif experiment == "cifar_planes_cars":
        results = run_transfer_experiment(config, run_dir, device, "cifar_planes_cars")
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Save summary results
    summary = {
        "best_accuracy": results.get("quantum", results).get("best_accuracy",
                                                             results.get("best_accuracy", 0)),
        "final_accuracy": results.get("quantum", results).get("final_accuracy",
                                                              results.get("final_accuracy", 0)),
        "total_time": results.get("quantum", results).get("total_time",
                                                          results.get("total_time", 0))
    }

    with open(run_dir / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save training artifacts (losses CSV, best epoch info)
    _save_training_artifacts(results, run_dir, experiment)

    # Mark completion
    (run_dir / "done.txt").write_text(f"Completed at {datetime.now().isoformat()}")

    logger.info(f"Experiment complete! Results saved to {run_dir}")


if __name__ == "__main__":
    # For direct execution (testing)
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = json.load(f)
        main(config)
    else:
        # Run example config
        example_config = {
            "seed": 42,
            "experiment": "spiral",
            "dataset": {"n_samples": 220},
            "model": {"n_qubits": 4, "q_depth": 2},
            "training": {"epochs": 5, "batch_size": 10},
            "options": {"compare_classical": False, "generate_figures": False}
        }
        main(example_config)
