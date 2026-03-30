# ruff: noqa: N812
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm

from lib.classical import count_svm_parameters, get_mlp_deep, get_mlp_wide
from lib.vqc import count_parameters, get_vqc

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class ExperimentArgs:
    m: int
    input_size: int
    initial_state: list[int]
    activation: str = "none"
    no_bunching: bool = False
    num_runs: int = 5
    n_epochs: int = 150
    batch_size: int = 30
    lr: float = 0.02
    alpha: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    circuit: str = "bs_mesh"
    scale_type: str = "learned"
    regu_on: str | None = "linear"
    log_wandb: bool = False
    wandb_project: str = "vqc_reproduction"
    wandb_entity: str | None = None
    device: str = "cpu"
    model_type: str = "vqc"
    dataset_name: str = ""
    requested_model_type: str = "vqc"

    def set_dataset_name(self, dataset: str) -> None:
        self.dataset_name = dataset

    def set_model_type(self, model_type: str) -> None:
        self.model_type = model_type


def _maybe_init_wandb(args: ExperimentArgs, tags: list[str]) -> bool:
    if not args.log_wandb:
        return False
    if wandb is None:  # pragma: no cover - optional dependency
        raise ImportError("wandb is not installed but log_wandb=True.")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset_name,
            "initial_state": args.initial_state,
            "learning_rate": args.lr,
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "activation": args.activation,
            "no_bunching": args.no_bunching,
            "alpha": args.alpha,
            "betas": args.betas,
            "circuit": args.circuit,
            "scale_type": args.scale_type,
            "regu_on": args.regu_on,
        },
        tags=tags,
    )
    return True


def _format_model_label(
    model_type: str, initial_state: list[int] | None, uppercase: bool = False
) -> str:
    label = model_type.upper() if uppercase else model_type
    if model_type == "vqc" and initial_state is not None:
        return f"{label} {initial_state}"
    return label


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model_name: str,
    args: ExperimentArgs,
) -> dict[str, list[float]]:
    """Train a torch model and return training metrics."""
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    criterion = nn.MSELoss()

    losses: list[float] = []
    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    wandb_active = (
        _maybe_init_wandb(args, [args.model_type, args.dataset_name])
        if args.log_wandb
        else False
    )

    pbar = tqdm(range(args.n_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        permutation = torch.randperm(x_train.size(0))
        total_loss = 0.0

        for i in range(0, x_train.size(0), args.batch_size):
            idx = permutation[i : i + args.batch_size]
            batch_x = x_train[idx].to(device)
            batch_y = y_train[idx].to(device)

            if args.activation == "softmax":
                batch_y = F.one_hot(batch_y.long(), num_classes=2).float()

            preds = model(batch_x)

            penalty = 0.0
            if args.regu_on and args.alpha > 0:
                if args.regu_on == "linear":
                    layers = model[-1] if args.activation == "none" else model[-2]
                    params = list(layers.parameters())
                elif args.regu_on == "all":
                    params = list(model.parameters())
                else:
                    raise NotImplementedError(f"Unknown regu_on '{args.regu_on}'")
                if params:
                    flat = torch.cat([p.view(-1) for p in params])
                    penalty = args.alpha * torch.linalg.vector_norm(flat) ** 2

            loss = criterion(preds.squeeze(), batch_y.squeeze()) + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (x_train.size(0) // args.batch_size)
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train.to(device))
            train_preds = (
                torch.argmax(train_outputs, dim=1)
                if args.activation == "softmax"
                else torch.round(train_outputs)
            )
            train_acc = accuracy_score(y_train.cpu().numpy(), train_preds.cpu().numpy())
            train_accuracies.append(train_acc)

            test_outputs = model(x_test.to(device))
            test_preds = (
                torch.argmax(test_outputs, dim=1)
                if args.activation == "softmax"
                else torch.round(test_outputs)
            )
            test_acc = accuracy_score(y_test.cpu().numpy(), test_preds.cpu().numpy())
            test_accuracies.append(test_acc)

            pbar.set_description(
                f"Training {model_name} - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
            )

        if wandb_active:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )

        model.train()

    if wandb_active:
        wandb.finish()

    return {
        "losses": losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "final_test_acc": test_accuracies[-1],
    }


def _instantiate_model(model_type: str, args: ExperimentArgs) -> nn.Module:
    if model_type == "vqc":
        return get_vqc(
            args.m,
            args.input_size,
            args.initial_state,
            no_bunching=args.no_bunching,
            activation=args.activation,
            circuit=args.circuit,
            scale_type=args.scale_type,
        )
    if model_type == "mlp_wide":
        return get_mlp_wide(args.input_size, activation=args.activation)
    if model_type == "mlp_deep":
        return get_mlp_deep(args.input_size, activation=args.activation)
    raise NotImplementedError(f"Unknown torch model type '{model_type}'")


def train_model_multiple_runs(
    model_type: str, args: ExperimentArgs, datasets: dict[str, dict[str, torch.Tensor]]
) -> tuple[dict[str, dict], list[dict]]:
    args.set_model_type(model_type)
    results: dict[str, dict] = {}
    best_models: list[dict] = []

    for dataset_name, data in datasets.items():
        args.set_dataset_name(dataset_name)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        model_runs: list[dict] = []
        models: list = []

        if model_type.startswith("svm"):
            kernel = "linear" if model_type == "svm_lin" else "rbf"
            print(f"\nTraining SVM ({kernel}) on {dataset_name} ({args.num_runs} runs)")
            for run in range(args.num_runs):
                model = SVC(kernel=kernel, gamma="scale")
                model.fit(x_train.cpu().numpy(), y_train.cpu().numpy())
                num_params = count_svm_parameters(model, kernel)
                print(f"  Run {run + 1}/{args.num_runs} — params: {num_params}")
                y_pred = model.predict(x_test.cpu().numpy())
                accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
                run_results = {
                    "losses": [0.0] * args.n_epochs,
                    "train_accuracies": [0.0] * args.n_epochs,
                    "test_accuracies": [0.0] * args.n_epochs,
                    "final_test_acc": accuracy,
                }
                models.append(model)
                model_runs.append(run_results)

        else:
            model_label = _format_model_label(
                model_type, args.initial_state, uppercase=False
            )
            print(f"\nTraining {model_label} on {dataset_name} ({args.num_runs} runs)")
            for run in range(args.num_runs):
                model = _instantiate_model(model_type, args)
                num_params = (
                    count_parameters(model)
                    if model_type == "vqc"
                    else sum(p.numel() for p in model.parameters() if p.requires_grad)
                )
                print(f"  Run {run + 1}/{args.num_runs} — params: {num_params}")
                run_label = _format_model_label(
                    model_type, args.initial_state, uppercase=False
                )
                history = train_model(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    f"{run_label}-run{run + 1}",
                    args,
                )
                models.append(model.cpu())
                model_runs.append(history)

        best_run = torch.argmax(
            torch.tensor([run["final_test_acc"] for run in model_runs])
        )
        best_model = models[best_run]
        best_acc = model_runs[best_run]["final_test_acc"]
        best_models.append(
            {
                "dataset": dataset_name,
                "model": best_model,
                "best_acc": float(best_acc),
                "model_type": model_type,
                "requested_model_type": args.requested_model_type,
                "activation": args.activation,
                "initial_state": list(args.initial_state)
                if model_type == "vqc"
                else None,
                "x_train": x_train.cpu(),
                "y_train": y_train.cpu(),
                "x_test": x_test.cpu(),
                "y_test": y_test.cpu(),
            }
        )

        results[dataset_name] = {
            "runs": model_runs,
            "avg_final_test_acc": sum(run["final_test_acc"] for run in model_runs)
            / len(model_runs),
        }

    return results, best_models


def summarize_results(results: dict[str, dict], args: ExperimentArgs) -> str:
    lines = ["----- Hyperparameters -----"]
    lines.append(f"m = {args.m}")
    lines.append(f"input_size = {args.input_size}")
    lines.append(f"initial_state = {args.initial_state}")
    lines.append("")
    lines.append("Training setup:")
    lines.append(
        f"activation = {args.activation}, no_bunching = {args.no_bunching}, num_runs = {args.num_runs}, epochs = {args.n_epochs}"
    )
    lines.append(
        f"batch_size = {args.batch_size}, lr = {args.lr}, alpha = {args.alpha}, betas = {args.betas}"
    )
    lines.append(
        f"circuit = {args.circuit}, scale_type = {args.scale_type}, regu_on = {args.regu_on}"
    )
    lines.append("")
    lines.append("----- Model Comparison Results -----")

    for dataset_name, data in results.items():
        finals = [run["final_test_acc"] for run in data["runs"]]
        avg = sum(finals) / len(finals)
        std = (sum((a - avg) ** 2 for a in finals) / len(finals)) ** 0.5
        label = _format_model_label(args.model_type, args.initial_state, uppercase=True)
        lines.append(
            f"{label} on {dataset_name}: {avg:.4f} ± {std:.4f} (min: {min(finals):.4f}, max: {max(finals):.4f})"
        )

    summary = "\n".join(lines)
    print(summary)
    return summary
