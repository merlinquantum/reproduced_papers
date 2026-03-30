from __future__ import annotations

import copy
from collections.abc import Iterable
from statistics import mean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0])


def train_single_run(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    cfg: dict,
    *,
    device: torch.device,
    desc: str,
) -> dict:
    epochs = int(cfg.get("epochs", 120))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 0.02))
    betas = cfg.get("betas", [0.9, 0.999])

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    criterion = nn.MSELoss()

    dataset = TensorDataset(train_x, train_y.unsqueeze(-1))
    losses: list[float] = []
    mses: list[float] = []

    for _epoch in range(epochs):
        # Shuffle each epoch by relying on DataLoader's sampling
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x).view(-1)
            targets = batch_y.view(-1)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        losses.append(avg_loss)

        with torch.no_grad():
            full_x = train_x.to(device)
            full_y = train_y.to(device)
            preds = model(full_x).view(-1)
            mse = torch.mean((preds - full_y) ** 2).item()
            mses.append(mse)

    return {"losses": losses, "train_mses": mses}


def train_models_multiple_runs(
    model_factory,
    initial_states: list[Iterable[int]],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    cfg: dict,
    *,
    device: torch.device,
    colors: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    num_runs = int(cfg.get("num_runs", 3))
    colors = colors or DEFAULT_COLORS

    results: dict[str, dict] = {}
    best_models: list[dict] = []

    for idx, initial_state in enumerate(initial_states):
        label = f"VQC_{initial_state}"
        color = colors[idx % len(colors)]
        run_histories: list[dict] = []
        best_mse = float("inf")
        best_model_state = None

        for run in range(num_runs):
            model = model_factory.build(initial_state)
            history = train_single_run(
                model,
                train_x,
                train_y,
                cfg,
                device=device,
                desc=f"{label} run {run + 1}/{num_runs}",
            )
            run_histories.append(history)

            final_mse = history["train_mses"][-1]
            if final_mse < best_mse:
                best_mse = final_mse
                best_model_state = copy.deepcopy(model.cpu().state_dict())

        best_model = model_factory.build(initial_state)
        if best_model_state:
            best_model.load_state_dict(best_model_state)
        best_models.append(
            {
                "label": label,
                "model": best_model,
                "color": color,
                "initial_state": initial_state,
            }
        )

        results[label] = {
            "initial_state": list(initial_state),
            "color": color,
            "runs": run_histories,
        }

    return results, best_models


def summarize_results(results: dict[str, dict]) -> str:
    lines = ["----- Model Comparison Results -----", ""]
    for label, data in results.items():
        finals = [run["train_mses"][-1] for run in data["runs"]]
        avg = mean(finals)
        variance = mean((m - avg) ** 2 for m in finals)
        std = variance**0.5
        lines.append(label)
        lines.append(
            f"  Final train MSE: {avg:.6f} ± {std:.6f} (min: {min(finals):.6f}, max: {max(finals):.6f})"
        )
        lines.append("")
    return "\n".join(lines)
