import datetime
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm


class InfoNCELossFromPaper(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, out_1, out_2):
        out = torch.cat([out_1, out_2], dim=0)
        batch_size = out_1.shape[0]
        # InfoNCE Loss

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (
            torch.ones_like(sim_matrix)
            - torch.eye(2 * batch_size, device=sim_matrix.device)
        ).type(torch.bool)
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Efficient vectorized InfoNCE implementation.
        """
        z1 = f.normalize(z1, dim=1)
        z2 = f.normalize(z2, dim=1)
        batch_size = z1.shape[0]
        device = z1.device

        # Compute similarity matrices
        sim_11 = torch.matmul(z1, z1.T) / self.temperature
        sim_22 = torch.matmul(z2, z2.T) / self.temperature
        sim_12 = torch.matmul(z1, z2.T) / self.temperature

        # Positive pairs are on the diagonal of sim_12
        pos_sim = torch.diag(sim_12)

        # Negatives: all similarities except the positive pair and self-similarities
        # For z1: negatives are other z1s + all z2s except the corresponding one
        neg_sim_1 = torch.cat(
            [
                sim_11.masked_fill(
                    torch.eye(batch_size, device=device).bool(), float("-inf")
                ),
                sim_12.masked_fill(
                    torch.eye(batch_size, device=device).bool(), float("-inf")
                ),
            ],
            dim=1,
        )

        # For z2: negatives are other z2s + all z1s except the corresponding one
        neg_sim_2 = torch.cat(
            [
                sim_12.T.masked_fill(
                    torch.eye(batch_size, device=device).bool(), float("-inf")
                ),
                sim_22.masked_fill(
                    torch.eye(batch_size, device=device).bool(), float("-inf")
                ),
            ],
            dim=1,
        )

        # InfoNCE loss using logsumexp for numerical stability
        loss_1 = -pos_sim + torch.logsumexp(neg_sim_1, dim=1)
        loss_2 = -pos_sim + torch.logsumexp(neg_sim_2, dim=1)

        return ((loss_1.mean() + loss_2.mean()) / 2).unsqueeze(0)


def training_step(model, train_loader, optimizer):
    pbar = tqdm(train_loader)
    total_loss = 0.0
    for (x1, x2), _target in pbar:
        loss = model(x1, x2)

        # Check for NaN/inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss}")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss_scalar = loss.item() if loss.dim() == 0 else loss[0].item()
        total_loss += loss_scalar
        pbar.set_postfix({"Loss": f"{loss_scalar:.4f}"})

    return total_loss / len(train_loader), model


def get_results_dir(args):
    """Create and return results directory path based on training type"""
    if args.merlin:
        base_dir = "results/merlin"
    elif args.qiskit:
        base_dir = "results/qiskit"
    else:
        base_dir = "results/classical"

    # Create datetime subdirectory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def save_metrics_during_training(
    results_dir,
    epoch,
    ssl_loss=None,
    train_loss=None,
    val_loss=None,
    train_acc=None,
    val_acc=None,
):
    """Save metrics to JSON file during training"""
    metrics_file = os.path.join(results_dir, "training_metrics.json")

    # Load existing metrics or create new
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {
            "ssl_training_losses": [],
            "linear_evaluation": {
                "train_losses": [],
                "val_losses": [],
                "train_accuracies": [],
                "val_accuracies": [],
            },
        }

    # Update metrics
    if ssl_loss is not None:
        metrics["ssl_training_losses"].append({"epoch": epoch, "loss": ssl_loss})

    if train_loss is not None:
        metrics["linear_evaluation"]["train_losses"].append(
            {"epoch": epoch, "loss": train_loss}
        )

    if val_loss is not None:
        metrics["linear_evaluation"]["val_losses"].append(
            {"epoch": epoch, "loss": val_loss}
        )

    if train_acc is not None:
        metrics["linear_evaluation"]["train_accuracies"].append(
            {"epoch": epoch, "accuracy": train_acc}
        )

    if val_acc is not None:
        metrics["linear_evaluation"]["val_accuracies"].append(
            {"epoch": epoch, "accuracy": val_acc}
        )

    # Save updated metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


def train(model, train_loader, results_dir, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    training_losses = []

    # Create results directory
    print(f"Saving training results to: {results_dir}")

    torch.save(
        model.state_dict(),
        os.path.join(results_dir, f"model-cl-{args.classes}-epoch-0.pth"),
    )
    print(" - Initial model saved - ")

    for epoch in range(args.epochs):
        loss, model = training_step(model, train_loader, optimizer)
        print(f"epoch: {epoch + 1}/{args.epochs}, training loss: {loss}")
        training_losses.append(loss)

        # Save SSL training loss during training
        save_metrics_during_training(results_dir, epoch + 1, ssl_loss=loss)
        # Save model if required
        if (epoch + 1) % args.ckpt_step == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    results_dir, f"model-cl-{args.classes}-epoch-{epoch + 1}.pth"
                ),
            )
            print(f" - Model saved at epoch {epoch + 1}/{args.epochs} - ")

    torch.save(
        model.state_dict(),
        os.path.join(results_dir, f"model-cl-{args.classes}-epoch-{args.epochs}.pth"),
    )
    print(f" - Final model saved to: {results_dir} - ")

    return model, training_losses


def linear_evaluation(model, train_loader, val_loader, args, results_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(args.le_epochs):
        # training
        model.train()
        pbar = tqdm(train_loader)
        train_acc = 0
        train_loss_total = 0
        for img, target in pbar:
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item()
            train_acc += accuracy
            train_loss_total += loss.item()
            pbar.set_postfix(
                {
                    "Training Loss": f"{loss.item():.4f} - Training Accuracy: {accuracy:.4f}"
                }
            )

        # validation
        model.eval()
        pbar = tqdm(val_loader)
        val_acc = 0
        val_loss_total = 0
        with torch.no_grad():
            for img, target in pbar:
                output = model(img)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item()
                val_acc += accuracy
                val_loss_total += loss.item()
                pbar.set_postfix(
                    {
                        "Validation Loss": f"{loss.item():.4f} - Validation Accuracy: {accuracy:.4f}"
                    }
                )

        avg_train_acc = train_acc / len(train_loader.dataset)
        avg_val_acc = val_acc / len(val_loader.dataset)
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Save metrics during training
        save_metrics_during_training(
            results_dir,
            epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=avg_train_acc,
            val_acc=avg_val_acc,
        )

        print(
            f"Epoch {epoch + 1}/{args.le_epochs}: Train Acc = {avg_train_acc:.4f}, Val Acc = {avg_val_acc:.4f}"
        )

    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_loss(training_losses, args):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, args.epochs + 1),
        training_losses,
        "b-",
        linewidth=2,
        label="Training Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title = "Classical"
    if args.merlin:
        title = "Quantum_MerLin"
    if args.qiskit:
        title = "Quantum_Qiskit"
    plt.title(f"SSL Training Loss ({title} Network)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"ssl_training_loss_{title}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_evaluation_metrics(train_losses, val_losses, train_accs, val_accs, args):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
    title = "Classical"
    if args.merlin:
        title = "Quantum_MerLin"
    if args.qiskit:
        title = "Quantum_Qiskit"
    # Plot losses
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Linear Evaluation Losses ({title} Network)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", linewidth=2, label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", linewidth=2, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Linear Evaluation Accuracies ({title} Network)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"evaluation_metrics_{title}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def save_results_to_json(
    args,
    ssl_training_losses,
    ft_train_losses,
    ft_val_losses,
    ft_train_accs,
    ft_val_accs,
    results_dir,
):
    # Determine title based on quantum mode
    title = "Classical"
    if args.merlin:
        title = "Quantum_MerLin"
    if args.qiskit:
        title = "Quantum_Qiskit"

    # Save final summary to results directory
    summary_file = os.path.join(results_dir, "experiment_summary.json")

    # Create experiment entry
    experiment = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_type": title,
        "arguments": {
            "quantum-MerLin": args.merlin,
            "quantum-Qiskit": args.qiskit,
            "epochs": args.epochs,
            "le_epochs": args.le_epochs,
            "batch_size": args.batch_size,
            "classes": args.classes,
            "width": args.width,
            "loss_dim": args.loss_dim,
            "temperature": args.temperature,
            "modes": getattr(args, "modes", None),
            "no_bunching": getattr(args, "no_bunching", None),
            "datadir": args.datadir,
        },
        "ssl_training_losses": ssl_training_losses,
        "linear_evaluation": {
            "train_losses": ft_train_losses,
            "val_losses": ft_val_losses,
            "train_accuracies": ft_train_accs,
            "val_accuracies": ft_val_accs,
            "final_val_accuracy": ft_val_accs[-1] if ft_val_accs else 0.0,
            "best_val_accuracy": max(ft_val_accs) if ft_val_accs else 0.0,
        },
    }

    # Save experiment summary to results directory
    with open(summary_file, "w") as f:
        json.dump(experiment, f, indent=2)

    # Also save to the original location for backwards compatibility
    filename = f"{title}_results.json"
    try:
        with open(filename) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    # Append new experiment
    results.append(experiment)

    # Save updated results
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:")
    print(f"  - {summary_file}")
    print(f"  - {filename}")
    print(f"  - Training metrics: {os.path.join(results_dir, 'training_metrics.json')}")
    print(
        f"Final validation accuracy: {experiment['linear_evaluation']['final_val_accuracy']:.4f}"
    )
    print(
        f"Best validation accuracy: {experiment['linear_evaluation']['best_val_accuracy']:.4f}"
    )
