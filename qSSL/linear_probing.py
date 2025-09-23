# Main training script for Quantum Self-Supervised Learning (qSSL)
# This script implements the complete training pipeline for both classical and quantum SSL methods
# Based on "Quantum Self-Supervised Learning" by Jaderberg et al. (2022)

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from data_utils import load_finetuning_data
from model import QSSL
from torchsummary import summary
from training_utils import linear_evaluation

# Command-line argument parser for configuring the experiment
parser = argparse.ArgumentParser(description="PyTorch Quantum self-sup training")

# ========== Dataset Configuration ==========
parser.add_argument(
    "-d", "--datadir", metavar="DIR", default="./data", help="path to dataset"
)
parser.add_argument(
    "-p",
    "--pretrained",
    type=str,
    default="./results/qiskit/20250819_182304",
    help="path to folder with trained models",
)
# ========== Training Configuration ==========
parser.add_argument(
    "-le",
    "--le-epochs",
    type=int,
    default=100,
    help="Number of epochs for linear evaluation",
)
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size")


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


def list_pth_files_pathlib(directory_path):
    """
    List all files ending in .pth using pathlib.

    Args:
        directory_path (str): Path to the directory to search

    Returns:
        list: List of .pth file names found in the directory
    """
    try:
        path = Path(directory_path)
        pth_files = [file.name for file in path.glob("*.pth") if file.is_file()]
        return pth_files
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return []


def extract_model_info(filename):
    """
    Extract classes and epoch from model filename.

    Args:
        filename (str): Filename in format "model-cl-{classes}-epoch-{epoch}.pth"

    Returns:
        tuple: (classes, epoch) or (None, None) if pattern doesn't match
    """
    pattern = r"model-cl-(\d+)-epoch-(\d+)\.pth"
    match = re.search(pattern, filename)

    if match:
        classes = int(match.group(1))
        epoch = int(match.group(2))
        return classes, epoch
    else:
        return None, None


def save_linear_probing_results(
    args, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs, epoch, results_dir
):
    """
    Save linear probing results to a JSON file.

    Args:
        args: Command line arguments
        ft_train_losses: Training losses from fine-tuning
        ft_val_losses: Validation losses from fine-tuning
        ft_train_accs: Training accuracies from fine-tuning
        ft_val_accs: Validation accuracies from fine-tuning
        epoch: The epoch from which the pretrained model was loaded
        results_dir: Directory to save results
    """
    import datetime

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_type": "Linear_Probing",
        "pretrained_epoch": epoch,
        "arguments": vars(args),
        "linear_evaluation": {
            "train_losses": ft_train_losses,
            "val_losses": ft_val_losses,
            "train_accuracies": ft_train_accs,
            "val_accuracies": ft_val_accs,
            "final_val_accuracy": ft_val_accs[-1] if ft_val_accs else 0.0,
            "best_val_accuracy": max(ft_val_accs) if ft_val_accs else 0.0,
        },
    }

    results_file = os.path.join(
        results_dir,
        f"linear_probing_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Linear probing results saved to: {results_file}")


class FactorMultiplication(nn.Module):
    """Normalizes phases to be within quantum-friendly range"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        out = x * self.factor
        # print(f"x from {torch.min(out)} to {torch.max(out)}")
        return out


if __name__ == "__main__":
    args = parser.parse_args()

    # define if we have a directory or a file
    pretrained_path = Path(args.pretrained)
    pretrained_model_file = True if os.path.isfile(pretrained_path) else False

    if pretrained_model_file:
        # Handle case where pretrained is a specific .pth file
        parent_dir = Path(os.path.dirname(pretrained_path))
        results_dir = parent_dir
        args_json_path = parent_dir / "args.json"
        with open(args_json_path) as f:
            saved_args_dict = json.load(f)
        # Create an argparse.Namespace object from the loaded arguments
        saved_args = argparse.Namespace(**saved_args_dict)
        saved_args.le_epochs = args.le_epochs
        ### data ###
        # Load linear evaluation data with minimal augmentations
        args.classes = saved_args.classes
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

        model = QSSL(saved_args)

        model.load_state_dict(torch.load(pretrained_path))
        print(f"\n - Model loaded from {pretrained_path}")
        summary(model, [(3, 32, 32), (3, 32, 32)])

        classes, epoch = extract_model_info(os.path.basename(pretrained_path))
        factor = torch.pi if not saved_args.merlin else 1 / torch.pi
        print(f"\n - Using a factor = {factor} \n")

        # ========== Phase 2: Linear Evaluation ==========
        # Build model for linear evaluation with frozen feature extractor
        frozen_model = nn.Sequential(
            model.backbone,  # ResNet18 backbone (frozen)
            model.comp,  # Compression layer (frozen)
            nn.Sigmoid(),
            FactorMultiplication(factor),
            model.representation_network,  # Quantum/classical rep network (frozen)
            nn.Linear(
                model.rep_net_output_size, saved_args.classes
            ),  # Linear classifier (trainable)
        )
        print(
            f"Trainable parameters in linear evaluation: {sum(p.numel() for p in frozen_model[2].parameters() if p.requires_grad) + sum(p.numel() for p in frozen_model[-1].parameters() if p.requires_grad)}"
        )
        # Freeze all layers except the final linear classifier
        frozen_model.requires_grad_(False)
        frozen_model[-1].requires_grad_(True)

        # Perform linear evaluation to assess quality of learned representations
        trained_model, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = (
            linear_evaluation(
                frozen_model, train_loader, val_loader, saved_args, str(results_dir)
            )
        )

        # Save results using new function
        save_linear_probing_results(
            saved_args,
            ft_train_losses,
            ft_val_losses,
            ft_train_accs,
            ft_val_accs,
            epoch,
            str(results_dir),
        )

    else:
        # Handle case where pretrained is a directory containing args.json
        results_dir = pretrained_path
        args_json_path = pretrained_path / "args.json"

        with open(args_json_path) as f:
            saved_args_dict = json.load(f)

        # Create an argparse.Namespace object from the loaded arguments
        saved_args = argparse.Namespace(**saved_args_dict)
        saved_args.le_epochs = args.le_epochs
        print(f"Loaded training arguments from {args_json_path}")

        ### data ###æ
        # Load linear evaluation data with minimal augmentations
        args.classes = saved_args.classes
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

        # Initialize model with the original training arguments
        # Find and load the model checkpoint
        pth_files = list_pth_files_pathlib(str(pretrained_path))
        print(f"Evaluating {len(pth_files)} models ({pth_files})")
        for pth_file in pth_files:
            model = QSSL(saved_args)
            model.load_state_dict(torch.load(os.path.join(results_dir, pth_file)))
            print(f"Loaded model weights from {pth_files}")
            # Extract epoch from model filename
            classes, epoch = extract_model_info(pth_file)
            print(f"Evaluating model on {epoch} epochs on {classes} classes")

            # ========== Phase 2: Linear Evaluation ==========
            # Build model for linear evaluation with frozen feature extractor
            factor = torch.pi if not saved_args.merlin else 1 / torch.pi
            print(f"\n - Using a factor = {factor} \n")

            frozen_model = nn.Sequential(
                model.backbone,  # ResNet18 backbone (frozen)
                model.comp,  # Compression layer (frozen)
                nn.Sigmoid(),
                FactorMultiplication(factor),
                model.representation_network,  # Quantum/classical rep network (frozen)
                nn.Linear(
                    model.rep_net_output_size, saved_args.classes
                ),  # Linear classifier (trainable)
            )
            print(
                f"Trainable parameters in linear evaluation: {sum(p.numel() for p in frozen_model[2].parameters() if p.requires_grad) + sum(p.numel() for p in frozen_model[-1].parameters() if p.requires_grad)}"
            )
            # Freeze all layers except the final linear classifier
            frozen_model.requires_grad_(False)
            frozen_model[-1].requires_grad_(True)

            # Perform linear evaluation to assess quality of learned representations
            (
                trained_model,
                ft_train_losses,
                ft_val_losses,
                ft_train_accs,
                ft_val_accs,
            ) = linear_evaluation(
                frozen_model, train_loader, val_loader, saved_args, str(results_dir)
            )

            # Save results using new function
            save_linear_probing_results(
                saved_args,
                ft_train_losses,
                ft_val_losses,
                ft_train_accs,
                ft_val_accs,
                epoch,
                str(results_dir),
            )
