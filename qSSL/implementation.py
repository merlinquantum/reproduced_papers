# Main training script for Quantum Self-Supervised Learning (qSSL)
# This script implements the complete training pipeline for both classical and quantum SSL methods
# Based on "Quantum Self-Supervised Learning" by Jaderberg et al. (2022)

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from lib.config import deep_update, default_config, load_config
from lib.data_utils import load_finetuning_data, load_transformed_data
from lib.model import QSSL
from lib.training_utils import (
    get_results_dir,
    linear_evaluation,
    save_results_to_json,
    train,
)
from torchsummary import summary

# Command-line argument parser for configuring the experiment
parser = argparse.ArgumentParser(description="PyTorch Quantum self-sup training")

# ========== Dataset Configuration ==========
parser.add_argument(
    "-d", "--datadir", metavar="DIR", default="./data", help="path to dataset"
)
parser.add_argument("-cl", "--classes", type=int, default=2, help="Number of classes")
# ========== Training Configuration ==========
parser.add_argument(
    "-e", "--epochs", type=int, default=2, help="Number of epochs for SSL pre-training"
)
parser.add_argument(
    "-le",
    "--le-epochs",
    type=int,
    default=100,
    help="Number of epochs for linear evaluation",
)
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "-ckpt", "--ckpt-step", type=int, default=1, help="Epochs when the model is saved"
)
# ========== SSL Model Configuration ==========
parser.add_argument(
    "-bn",
    "--batch_norm",
    action="store_true",
    default=False,
    help="Set if we use BatchNorm after compression of the encoder",
)
# ========== Contrastive Loss Configuration ==========
parser.add_argument(
    "-ld", "--loss_dim", type=int, default=128, help="Dimension of the loss space"
)
parser.add_argument(
    "-tau",
    "--temperature",
    type=float,
    default=0.07,
    help="Temperature parameter for InfoNCE loss (lower = harder negatives)",
)
# ========== Quantum SSL Configuration (MerLin) ==========
parser.add_argument(
    "-w",
    "--width",
    type=int,
    default=8,
    help="Dimension of the features encoded in the quantum neural network",
)
parser.add_argument(
    "--merlin",
    action="store_true",
    default=False,
    help="Use Quantum SSL with MerLin photonic framework",
)
parser.add_argument(
    "-m",
    "--modes",
    type=int,
    default=10,
    help="Number of photonic modes in quantum circuit",
)
parser.add_argument(
    "-bunch",
    "--no_bunching",
    action="store_true",
    default=False,
    help="Disable photon bunching in quantum circuit",
)

# ========== Quantum SSL Configuration (Qiskit) ==========
# Based on implementation from https://github.com/bjader/QSSL
parser.add_argument(
    "--qiskit",
    action="store_true",
    default=False,
    help="Use Quantum SSL with Qiskit quantum computing framework",
)
parser.add_argument(
    "--layers",
    type=int,
    default=2,
    help="Number of layers in the test network (default: 2).",
)
parser.add_argument(
    "--q_backend",
    type=str,
    default="qasm_simulator",
    help="Type of backend simulator to run quantum circuits on (default: qasm_simulator)",
)

parser.add_argument(
    "--encoding",
    type=str,
    default="vector",
    help="Data encoding method (default: vector)",
)
parser.add_argument(
    "--q_ansatz",
    type=str,
    default="sim_circ_14_half",
    help="Variational ansatz method (default: sim_circ_14_half)",
)
parser.add_argument("--q_sweeps", type=int, default=1, help="Number of ansatz sweeeps.")
parser.add_argument(
    "--activation",
    type=str,
    default="null",
    help="Quantum layer activation function type (default: null)",
)
parser.add_argument(
    "--shots",
    type=int,
    default=100,
    help="Number of shots for quantum circuit evaluations.",
)
parser.add_argument(
    "--save-dhs",
    action="store_true",
    help="If enabled, compute the Hilbert-Schmidt distance of the quantum statevectors belonging to"
    " each class. Only works for -q and --classes 2.",
)
parser.add_argument("--config", type=str, default=None, help="Path to JSON config")


if __name__ == "__main__":
    args = parser.parse_args()

    # Helper to check if a CLI flag was explicitly provided
    def _provided(*names: str) -> bool:
        import sys as _sys

        return any(n in _sys.argv for n in names)

    # Load config if provided and map to args (config takes effect unless CLI overrides are explicitly passed)
    if args.config is not None and os.path.exists(args.config):
        cfg = deep_update(default_config(), load_config(Path(args.config)))

        # Backend resolution: CLI flags win if present, else config
        backend = cfg["model"].get("backend", "classical")
        if _provided("--merlin"):
            backend = "merlin"
        elif _provided("--qiskit"):
            backend = "qiskit"
        args.merlin = backend == "merlin"
        args.qiskit = backend == "qiskit"

        # Dataset
        if not _provided("-d", "--datadir"):
            args.datadir = cfg["dataset"]["root"]
        if not _provided("-cl", "--classes"):
            args.classes = cfg["dataset"]["classes"]
        if not _provided("-bs", "--batch_size"):
            args.batch_size = cfg["dataset"]["batch_size"]

        # Training
        if not _provided("-e", "--epochs"):
            args.epochs = cfg["training"]["epochs"]
        if not _provided("-ckpt", "--ckpt-step"):
            args.ckpt_step = cfg["training"]["ckpt_step"]
        if not _provided("-le", "--le-epochs"):
            args.le_epochs = cfg["training"]["le_epochs"]

        # Model (common)
        if not _provided("-w", "--width"):
            args.width = cfg["model"]["width"]
        if not _provided("-ld", "--loss_dim"):
            args.loss_dim = cfg["model"]["loss_dim"]
        if not _provided("-tau", "--temperature"):
            args.temperature = cfg["model"]["temperature"]
        # batch_norm and no_bunching are store_true; only set from config if the flag wasn't provided
        if not _provided("-bn", "--batch_norm"):
            args.batch_norm = cfg["model"]["batch_norm"]

        # Qiskit specific
        if not _provided("--layers"):
            args.layers = cfg["model"].get("layers", 2)
        if not _provided("--q_backend"):
            args.q_backend = cfg["model"].get("q_backend", "qasm_simulator")
        if not _provided("--encoding"):
            args.encoding = cfg["model"].get("encoding", "vector")
        if not _provided("--q_ansatz"):
            args.q_ansatz = cfg["model"].get("q_ansatz", "sim_circ_14_half")
        if not _provided("--q_sweeps"):
            args.q_sweeps = cfg["model"].get("q_sweeps", 1)
        if not _provided("--activation"):
            args.activation = cfg["model"].get("activation", "null")
        if not _provided("--shots"):
            args.shots = cfg["model"].get("shots", 100)
        if not _provided("--save-dhs"):
            # store_true default is False; config doesn't define it, keep False
            if not hasattr(args, "save_dhs"):
                args.save_dhs = False

        # MerLin specific
        if not _provided("-m", "--modes"):
            args.modes = cfg["model"].get("modes", 10)
        if not _provided("-bunch", "--no_bunching"):
            args.no_bunching = cfg["model"].get("no_bunching", False)

    results_dir = get_results_dir(args)
    # Save training arguments to JSON file for reproducibility
    args_dict = vars(args)
    with open(f"{results_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"Saved training arguments to {results_dir}/args.json")

    # ========== Phase 1: Self-Supervised Learning ==========
    # Load SSL training data with heavy augmentations for contrastive learning
    train_dataset = load_transformed_data(args)
    print(f"\n Loaded SSL training dataset with {len(train_dataset)} samples")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Initialize the QSSL model (quantum or classical based on args)
    model = QSSL(args)
    # Display model architecture summary for two CIFAR-10 images (query and key)
    summary(model, [(3, 32, 32), (3, 32, 32)])
    print(
        f"Total trainable parameters in SSL model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Train the SSL model using contrastive learning on augmented image pairs
    model, ssl_training_losses = train(model, train_loader, results_dir, args)

    # ========== Phase 2: Linear Evaluation ==========
    # Build model for linear evaluation with frozen feature extractor
    frozen_model = nn.Sequential(
        model.backbone,  # ResNet18 backbone (frozen)
        model.comp,  # Compression layer (frozen)
        model.representation_network,  # Quantum/classical rep network (frozen)
        nn.Linear(
            model.rep_net_output_size, args.classes
        ),  # Linear classifier (trainable)
    )
    print(
        f"Trainable parameters in linear evaluation: {sum(p.numel() for p in frozen_model[2].parameters() if p.requires_grad) + sum(p.numel() for p in frozen_model[-1].parameters() if p.requires_grad)}"
    )
    # Freeze all layers except the final linear classifier
    frozen_model.requires_grad_(False)
    frozen_model[-1].requires_grad_(True)

    # Load linear evaluation data with minimal augmentations
    train_dataset, eval_dataset = load_finetuning_data(args)
    print(
        f"\n Loaded linear evaluation datasets: {len(train_dataset)} train, {len(eval_dataset)} validation"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Perform linear evaluation to assess quality of learned representations
    model, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = (
        linear_evaluation(frozen_model, train_loader, val_loader, args, results_dir)
    )

    # ========== Save Experiment Results ==========
    # Save comprehensive results including SSL losses and linear evaluation metrics
    save_results_to_json(
        args,
        ssl_training_losses,
        ft_train_losses,
        ft_val_losses,
        ft_train_accs,
        ft_val_accs,
        results_dir,
    )
