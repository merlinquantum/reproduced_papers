import datetime
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def get_qiskit_qnn(model):
    """Return the underlying Qiskit QuantumNetworkCircuit for the representation
    network, or None if the model does not use the Qiskit backend."""
    return getattr(model.representation_network, "qnn", None)


def disable_qiskit_statevector_capture(model):
    """Stop statevector capture for every QNet contained in a model.

    Parameters
    ----------
    model : torch.nn.Module
        QSSL model or frozen linear-evaluation model containing a QNet.
    """
    from .qnn.qnet import QNet

    for qiskit_network in model.modules():
        if isinstance(qiskit_network, QNet):
            qiskit_network.save_statevectors = False
            qiskit_network.qnn.statevectors.clear()


def compute_batch_hilbert_schmidt_metrics(statevectors):
    """
    Compute the Hilbert-Schmidt separation between positive and negative pairs
    for one batch of quantum statevectors, following Jaderberg et al. (2022),
    "Quantum Self-Supervised Learning" (arXiv:2103.14653), Fig. 4.

    `statevectors` must be ordered as [psi(x_1^1), ..., psi(x_B^1),
    psi(x_1^2), ..., psi(x_B^2)], i.e. the states produced by encoding view 1
    of the whole batch followed by the states produced by encoding view 2 -
    which is exactly the order QNet.qnn.statevectors accumulates in during a
    single QSSL.forward call.

    For each positive pair i = (x_i^1, x_i^2), rho_i is the ensemble density
    matrix of its two views, and sigma_i is the ensemble density matrix of
    every other statevector in the batch (i.e. all negative pairs w.r.t. i).
    D_HS(rho_i, sigma_i) = tr((rho_i - sigma_i)^2) measures how well the
    positive pair is separated from the rest of the batch in Hilbert space.

    Returns the batch-averaged tr(rho^2), tr(sigma^2), tr(rho*sigma) and
    D_HS, computed via a leave-one-pair-out sum rather than the naive O(B^2)
    reconstruction of sigma_i for every i, so this is safe to run on full
    batches (e.g. 256 samples) on GPU-scale training runs.
    """
    statevectors = np.asarray(statevectors)
    n_total = statevectors.shape[0]
    batch_size = n_total // 2
    aug_1 = statevectors[:batch_size]
    aug_2 = statevectors[batch_size:]

    # Sum of outer products (unnormalised density matrix) of every statevector
    # in the batch: sum_j |psi_j><psi_j|
    total_sum = np.einsum("bi,bj->ij", statevectors, np.conj(statevectors))

    rho_sq, sigma_sq, rho_sigma, dhs = [], [], [], []
    for v1, v2 in zip(aug_1, aug_2):
        pair_sum = np.outer(v1, np.conj(v1)) + np.outer(v2, np.conj(v2))
        rho = pair_sum / 2
        # Leave-one-pair-out: negatives are every statevector except this pair's two
        sigma = (total_sum - pair_sum) / (n_total - 2)

        # tr(A @ B) == sum(A * B.T) elementwise, avoids a full matrix product
        rho_sq.append(np.sum(rho * rho.T).real)
        sigma_sq.append(np.sum(sigma * sigma.T).real)
        rho_sigma.append(np.sum(rho * sigma.T).real)
        diff = rho - sigma
        dhs.append(np.sum(diff * diff.T).real)

    return {
        "rho_squared": float(np.mean(rho_sq)),
        "sigma_squared": float(np.mean(sigma_sq)),
        "rho_sigma": float(np.mean(rho_sigma)),
        "d_hs": float(np.mean(dhs)),
    }


def compute_batch_probability_hilbert_schmidt_metrics(probability_vectors):
    """Compute the Hilbert-Schmidt surrogate for photonic probability vectors.

    MerLin currently returns photon-count probabilities rather than complex
    amplitudes. Treating each probability vector as the diagonal of a density
    matrix gives the exact Hilbert-Schmidt distance between those diagonal
    density matrices, but discards optical phases and coherences.

    Parameters
    ----------
    probability_vectors : array-like
        State probabilities ordered as [view-1 batch, view-2 batch]. Every row
        must describe one normalized probability distribution.

    Returns
    -------
    dict
        Batch-averaged squared positive density, negative density, overlap,
        and probability-space Hilbert-Schmidt distance.

    Raises
    ------
    ValueError
        If the input does not contain two complete views or has invalid
        probability values.
    """
    probability_vectors = np.asarray(probability_vectors, dtype=float)
    if probability_vectors.ndim != 2:
        raise ValueError(
            "probability_vectors must have shape (2 * batch_size, dimension)"
        )
    if probability_vectors.shape[0] % 2 != 0 or probability_vectors.shape[0] < 4:
        raise ValueError("probability_vectors must contain at least two complete views")
    if not np.all(np.isfinite(probability_vectors)):
        raise ValueError("probability_vectors must contain only finite values")
    if np.any(probability_vectors < 0):
        raise ValueError("probability_vectors cannot contain negative probabilities")
    if not np.allclose(probability_vectors.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability vector must sum to one")

    batch_size = probability_vectors.shape[0] // 2
    aug_1 = probability_vectors[:batch_size]
    aug_2 = probability_vectors[batch_size:]
    total_sum = probability_vectors.sum(axis=0)

    rho_squared = []
    sigma_squared = []
    rho_sigma = []
    d_hs = []
    for probability_view_1, probability_view_2 in zip(aug_1, aug_2):
        pair_sum = probability_view_1 + probability_view_2
        rho = pair_sum / 2
        sigma = (total_sum - pair_sum) / (probability_vectors.shape[0] - 2)
        difference = rho - sigma

        rho_squared.append(np.dot(rho, rho))
        sigma_squared.append(np.dot(sigma, sigma))
        rho_sigma.append(np.dot(rho, sigma))
        d_hs.append(np.dot(difference, difference))

    return {
        "rho_squared": float(np.mean(rho_squared)),
        "sigma_squared": float(np.mean(sigma_squared)),
        "rho_sigma": float(np.mean(rho_sigma)),
        "d_hs": float(np.mean(d_hs)),
    }


def save_dhs_history(results_dir, dhs_history):
    """Save the full per-batch Hilbert-Schmidt metric history to JSON."""
    with open(os.path.join(results_dir, "hilbert_schmidt_metrics.json"), "w") as fp:
        json.dump(dhs_history, fp, indent=2)


def training_step(
    model,
    train_loader,
    optimizer,
    max_steps=None,
    args=None,
    dhs_history=None,
    batch_offset=0,
):
    pbar = tqdm(train_loader)
    total_loss = 0.0
    steps_run = 0

    compute_dhs = args is not None and getattr(args, "save_dhs", False)
    use_qiskit_states = compute_dhs and getattr(args, "qiskit", False)
    use_merlin_probabilities = compute_dhs and getattr(args, "merlin", False)
    qnn = get_qiskit_qnn(model) if use_qiskit_states else None
    if use_qiskit_states and qnn is None:
        print(
            "Warning: --save-dhs requires --qiskit with a Qiskit representation "
            "network; Hilbert-Schmidt tracking will be skipped."
        )
        compute_dhs = False
    if use_merlin_probabilities and not hasattr(model, "photonic_probability_vectors"):
        raise AttributeError(
            "MerLin Hilbert-Schmidt tracking requires photonic probability vectors"
        )
    dhs_freq = max(getattr(args, "dhs_freq", 1) or 1, 1) if compute_dhs else 1

    batch_index = batch_offset
    for (x1, x2), _target in pbar:
        if max_steps is not None and steps_run >= max_steps:
            break

        if use_qiskit_states:
            # Reset so this batch's forward pass only accumulates its own states
            qnn.statevectors = []
        elif use_merlin_probabilities:
            model.photonic_probability_vectors = []

        loss = model(x1, x2)

        # Check for NaN/inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss}")
            continue

        loss_scalar = loss.item() if loss.dim() == 0 else loss[0].item()

        metric_inputs = None
        metric_function = None
        if use_qiskit_states and qnn.statevectors:
            metric_inputs = qnn.statevectors
            metric_function = compute_batch_hilbert_schmidt_metrics
        elif use_merlin_probabilities and model.photonic_probability_vectors:
            metric_inputs = torch.cat(model.photonic_probability_vectors, dim=0).numpy()
            metric_function = compute_batch_probability_hilbert_schmidt_metrics

        if compute_dhs and metric_inputs is not None and batch_index % dhs_freq == 0:
            metrics = metric_function(metric_inputs)
            metrics["batch"] = batch_index
            metrics["loss"] = loss_scalar
            dhs_history.append(metrics)
        if use_qiskit_states:
            qnn.statevectors = []  # bound memory even when this batch was skipped
        elif use_merlin_probabilities:
            model.photonic_probability_vectors = []

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss_scalar
        steps_run += 1
        batch_index += 1
        pbar.set_postfix({"Loss": f"{loss_scalar:.4f}"})

    if steps_run == 0:
        return 0.0, model, batch_index
    return total_loss / steps_run, model, batch_index


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
    dhs_history = []
    batch_offset = 0

    # Create results directory
    print(f"Saving training results to: {results_dir}")

    if getattr(args, "save_dhs", False):
        if not getattr(args, "qiskit", False) and not getattr(args, "merlin", False):
            print(
                "Warning: --save-dhs requires --qiskit or --merlin "
                "(the representation network must expose quantum outputs); "
                "Hilbert-Schmidt tracking will be skipped."
            )
        elif (
            getattr(args, "qiskit", False)
            and getattr(args, "q_backend", "") != "statevector_simulator"
        ):
            print(
                "Warning: --save-dhs requires --q_backend statevector_simulator "
                "to access raw quantum statevectors; Hilbert-Schmidt tracking "
                "will be skipped."
            )

    torch.save(
        model.state_dict(),
        os.path.join(results_dir, f"model-cl-{args.classes}-epoch-0.pth"),
    )
    print(" - Initial model saved - ")

    for epoch in range(args.epochs):
        remaining_steps = None
        total_steps = getattr(args, "total_steps", None)
        if total_steps is not None:
            remaining_steps = total_steps - batch_offset
            if remaining_steps <= 0:
                break
            if args.max_steps is not None:
                remaining_steps = min(remaining_steps, args.max_steps)

        loss, model, batch_offset = training_step(
            model,
            train_loader,
            optimizer,
            remaining_steps if args.total_steps is not None else args.max_steps,
            args=args,
            dhs_history=dhs_history,
            batch_offset=batch_offset,
        )
        print(f"epoch: {epoch + 1}/{args.epochs}, training loss: {loss}")
        training_losses.append(loss)

        # Save SSL training loss during training
        save_metrics_during_training(results_dir, epoch + 1, ssl_loss=loss)
        if dhs_history:
            save_dhs_history(results_dir, dhs_history)
        # Save model if required
        if (epoch + 1) % args.ckpt_step == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    results_dir, f"model-cl-{args.classes}-epoch-{epoch + 1}.pth"
                ),
            )

        if total_steps is not None and batch_offset >= total_steps:
            break
            print(f" - Model saved at epoch {epoch + 1}/{args.epochs} - ")

    torch.save(
        model.state_dict(),
        os.path.join(results_dir, f"model-cl-{args.classes}-epoch-{args.epochs}.pth"),
    )
    print(f" - Final model saved to: {results_dir} - ")

    if dhs_history:
        plot_loss_and_hilbert_schmidt(training_losses, dhs_history, args, results_dir)

    return model, training_losses


def linear_evaluation(model, train_loader, val_loader, args, results_dir):
    disable_qiskit_statevector_capture(model)
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
        train_steps_run = 0
        for img, target in pbar:
            if args.le_max_steps is not None and train_steps_run >= args.le_max_steps:
                break
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item() / target.size(0)
            train_acc += accuracy
            train_loss_total += loss.item()
            train_steps_run += 1
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
            val_steps_run = 0
            for img, target in pbar:
                if args.le_max_steps is not None and val_steps_run >= args.le_max_steps:
                    break
                output = model(img)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item() / target.size(0)
                val_acc += accuracy
                val_loss_total += loss.item()
                val_steps_run += 1
                pbar.set_postfix(
                    {
                        "Validation Loss": (
                            f"{loss.item():.4f} - Validation Accuracy: {accuracy:.4f}"
                        )
                    }
                )

        train_div = train_steps_run or len(train_loader)
        val_div = val_steps_run or len(val_loader)
        avg_train_acc = train_acc / train_div
        avg_val_acc = val_acc / val_div
        avg_train_loss = train_loss_total / train_div
        avg_val_loss = val_loss_total / val_div

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


def plot_loss_and_hilbert_schmidt(training_losses, dhs_history, args, results_dir):
    """
    Reproduce Fig. 4 of Jaderberg et al. (2022): the SSL training loss recorded
    after each batch, with insets tracking the average Hilbert-Schmidt distance
    between positive and negative pairs (D_HS), the average positive pair
    clustering tr(rho^2), the average clustering of all negative pairs
    tr(sigma^2), and the ensemble inter-cluster overlap tr(rho*sigma).
    """
    if not dhs_history:
        return

    batches = [m["batch"] for m in dhs_history]
    losses = [m["loss"] for m in dhs_history]
    d_hs = [m["d_hs"] for m in dhs_history]
    rho_sq = [m["rho_squared"] for m in dhs_history]
    sigma_sq = [m["sigma_squared"] for m in dhs_history]
    rho_sigma = [m["rho_sigma"] for m in dhs_history]

    # Match the paper's wide composition: the loss occupies the full figure
    # and the four Hilbert-Schmidt metrics form a compact 2x2 block above it.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batches, losses, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Training batches")
    ax.set_ylabel("Loss")
    if not args.merlin:
        # Qiskit's loss starts higher than MerLin's. Use the paper-like upper
        # limit so the inset block does not hide the loss curve.
        ax.set_ylim(top=max(12.0, max(losses) * 1.3))

    inset_specs = [
        ((0.27, 0.67, 0.29, 0.27), d_hs, "tab:orange", r"$\bar{D}_{HS}$"),
        ((0.59, 0.67, 0.29, 0.27), rho_sq, "tab:green", r"$\overline{tr(\rho^2)}$"),
        (
            (0.27, 0.37, 0.29, 0.27),
            rho_sigma,
            "tab:red",
            r"$\overline{tr(\rho\sigma)}$",
        ),
        (
            (0.59, 0.37, 0.29, 0.27),
            sigma_sq,
            "tab:purple",
            r"$\overline{tr(\sigma^2)}$",
        ),
    ]
    for inset_index, (bbox, values, color, label) in enumerate(inset_specs):
        inset = ax.inset_axes(bbox)
        inset.plot(batches, values, color=color, linewidth=1.2)
        # Put the metric name inside the plotting area, as in the original
        # figure, so it cannot collide with a neighboring inset or the title.
        inset.text(
            0.94,
            0.12,
            label,
            transform=inset.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
        )
        is_right_column = inset_index % 2 == 1
        is_top_row = inset_index < 2
        inset.tick_params(
            labelsize=8,
            direction="in",
            top=True,
            right=True,
            left=True,
            labelleft=not is_right_column,
            labelright=is_right_column,
            labelbottom=not is_top_row,
        )

    title = "Quantum_MerLin" if args.merlin else "Quantum_Qiskit"
    fig.suptitle(
        f"SSL training with Hilbert-Schmidt tracking ({title})",
        y=0.98,
        fontsize=16,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.90)
    out_path = os.path.join(results_dir, "hilbert_schmidt_tracking.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Hilbert-Schmidt tracking figure to {out_path}")


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
            "datadir": str(args.datadir)
            if isinstance(args.datadir, Path)
            else args.datadir,
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
