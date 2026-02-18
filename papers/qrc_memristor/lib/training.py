from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm
from utils.datasets import generate_narma


def train_sequence(
    model: torch.nn.Module,
    x_seq: torch.Tensor,
    y_seq: torch.Tensor,
    model_name: Optional[str] = None,
    n_epochs: int = 400,
    lr: float = 0.01,
    memory: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """
    Trains a model on a temporal sequence. This function iterates through the sequence time-step by time-step,
    accumulates predictions, computes the Mean Squared Error (MSE) against the target sequence,
    and updates the model parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained. It must implement
            a `reset_feedback()` method if `memory` is set to True.
        x_seq (torch.Tensor): The input sequence tensor. Expected shape is usually
            (Time, Batch, Features) or (Time, Features).
        y_seq (torch.Tensor): The target sequence tensor.
        model_name (Optional[str], optional): A label for the model to display
            in the progress bar. Defaults to None.
        n_epochs (int, optional): The number of training epochs. Defaults to 400.
        lr (float, optional): The learning rate for the Adam optimizer. Defaults to 0.01.
        memory (bool, optional): If True, calls `model.reset_feedback()` at the
            start of each epoch. Defaults to True.

    Returns:
        Dict[str, List[torch.Tensor]]: A dictionary containing training statistics.
            - 'losses': A list of loss values (0-d tensors) recorded at each epoch.
    """
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    pbar = tqdm(
        range(n_epochs), desc=f"Training {model_name}" if model_name else "Training"
    )
    losses = []

    for _epoch in pbar:
        model.train()
        if memory:
            if hasattr(model, "reset_feedback"):
                model.reset_feedback()
            else:
                raise AttributeError(
                    "Memory=True but model has no 'reset_feedback' method."
                )

        outs = []
        for t in range(x_seq.shape[0]):
            out_t = model(x_seq[t : t + 1])
            outs.append(out_t)

        outs = torch.cat(outs, dim=0)
        pred = outs[:, 0]
        loss = torch.mean((pred - y_seq.squeeze()) ** 2)
        loss = loss.real
        losses.append(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return {"losses": losses}


def R_to_theta(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    R = torch.clamp(R, eps, 1.0 - eps)
    return 2.0 * torch.acos(torch.sqrt(R))


def encode_phase(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)

    # For non-linear task
    # phi = 2.0 * torch.acos(torch.sqrt(x))
    # For NARMA
    phi = 2.0 * torch.acos(x)

    return phi


def extract_features_sequence(
    model: torch.nn.Module,
    x_seq: torch.Tensor,
    memory: bool = True,
    device: Optional[str] = None,
) -> torch.Tensor:
    model.eval()
    if memory:
        model.reset_feedback()

    outs = []
    with torch.no_grad():
        for t in range(x_seq.shape[0]):
            x_t = x_seq[t : t + 1]
            if device is not None:
                x_t = x_t.to(device)
            out_t = model(x_t)
            outs.append(out_t.detach().cpu())

    return torch.cat(outs, dim=0)


def fit_readout_mse(X_feat: torch.Tensor, y: torch.Tensor, split: float = 0.9) -> float:
    """
    X_feat: (N, D) torch tensor
    y: (N,)   torch tensor
    """
    N = y.shape[0]
    s = int(split * N)

    Xtr, Xte = X_feat[:s], X_feat[s:]
    ytr, yte = y[:s], y[s:]

    Xtr_ = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], dim=1)
    Xte_ = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], dim=1)

    w = torch.linalg.lstsq(Xtr_, ytr).solution

    y_pred_te = Xte_ @ w
    mse = torch.mean((y_pred_te - yte) ** 2).item()

    return mse


def get_param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def split_washout_train_test(
    x: torch.Tensor, y: torch.Tensor, washout: int = 20, train: int = 480
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    x,y are (N,1).

    Returns:
        washout slices, (x_train,y_train), (x_test,y_test).
    """
    x_wash, y_wash = x[:washout], y[:washout]
    x_train, y_train = x[washout : washout + train], y[washout : washout + train]
    x_test, y_test = x[washout + train :], y[washout + train :]

    return x_wash, y_wash, x_train, y_train, x_test, y_test


def collect_reservoir_features(
    model: torch.nn.Module, x_full: torch.Tensor
) -> torch.Tensor:
    """
    Runs the reservoir on the full input stream.
    Optimized to use internal model loops if available.
    """
    model.eval()
    model.reset_feedback()

    # Ensure x_full is (Batch, Time, Feat) or (Time, Feat)
    if x_full.dim() == 2:
        x_in = x_full.unsqueeze(0)  # Add batch dim -> (1, T, 1)
    else:
        x_in = x_full

    with torch.no_grad():
        # Call model once on the full sequence
        # The model I provided returns (Batch, Time, Output_Dim)
        out_seq = model(x_in)

        # Remove batch dim and return (Time, Output_Dim)
    return out_seq.squeeze(0).cpu()


def fit_readout_narma(
    R_tr: torch.Tensor,
    y_tr: torch.Tensor,
    R_te: torch.Tensor,
    y_te: torch.Tensor,
    lam: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
    """
    Ridge regression readout with bias.
    Inputs are torch tensors.
    Returns: w, b, train_mse, test_mse, yhat_tr, yhat_te
    """
    if y_tr.dim() == 1:
        y_tr = y_tr.view(-1, 1)
    if y_te.dim() == 1:
        y_te = y_te.view(-1, 1)

    device = R_tr.device
    dtype = R_tr.dtype

    # add bias
    Rtr_ = torch.cat(
        [R_tr, torch.ones(R_tr.shape[0], 1, device=device, dtype=dtype)], dim=1
    )
    Rte_ = torch.cat(
        [R_te, torch.ones(R_te.shape[0], 1, device=device, dtype=dtype)], dim=1
    )

    D = Rtr_.shape[1]
    identity_matrix = torch.eye(D, device=device, dtype=dtype)
    identity_matrix[-1, -1] = 0.0  # don't regularize bias

    w_full = torch.linalg.solve(
        Rtr_.T @ Rtr_ + lam * identity_matrix, Rtr_.T @ y_tr
    )  # (D,1)

    w = w_full[:-1, :]  # (D-1,1)
    b = w_full[-1, :]  # (1,)

    yhat_tr = (Rtr_ @ w_full).squeeze(1)
    yhat_te = (Rte_ @ w_full).squeeze(1)

    train_mse = torch.mean((yhat_tr - y_tr.squeeze(1)) ** 2).item()
    test_mse = torch.mean((yhat_te - y_te.squeeze(1)) ** 2).item()

    return w.squeeze(1), b.squeeze(0), train_mse, test_mse, yhat_tr, yhat_te


def narma_framework(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    washout: int = 20,
    train: int = 480,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    x_w, y_w, x_tr, y_tr, x_te, y_te = split_washout_train_test(
        x, y, washout=washout, train=train
    )

    R_all = collect_reservoir_features(model, x)

    R_tr = R_all[washout : washout + train]
    R_te = R_all[washout + train :]

    w, b, train_mse, test_mse, yhat_tr, yhat_te = fit_readout_narma(
        R_tr, y_tr, R_te, y_te
    )

    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test  MSE: {test_mse:.6f}")

    return (
        train_mse,
        test_mse,
        y_te.detach().cpu().numpy().flatten(),
        yhat_te.detach().cpu().numpy().flatten(),
    )


def run_narma_multiple(
    model_builder: Callable[[], torch.nn.Module],
    n_runs: int = 20,
    N: int = 1000,
    washout: int = 20,
    train: int = 480,
    base_seed: int = 0,
    device: str = "cpu",
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    test_mses = []

    best_mse = 1.0e9
    best_y_target = np.array([])
    best_y_pred = np.array([])

    for i in range(n_runs):
        seed = base_seed + i

        # 1) generate new NARMA sequence
        x, y, _ = generate_narma(data_size=N, seed=seed, device=device)

        # 2) fresh model (same architecture, same fixed phases)
        model = model_builder().to(device)

        # 3) run pipeline
        _, test_mse, y_target, y_pred = narma_framework(
            model, x, y, washout=washout, train=train
        )

        test_mses.append(test_mse)
        print(f"Run {i + 1:02d}/{n_runs} — test MSE = {test_mse:.6f}")

        if test_mse < best_mse:
            best_mse = test_mse
            best_y_target = y_target
            best_y_pred = y_pred

    test_mses_arr = np.array(test_mses)

    mean_mse = test_mses_arr.mean()
    std_mse = test_mses_arr.std()

    print("\n=== NARMA results ===")
    print(f"Runs: {n_runs}")
    print(f"Mean test MSE: {mean_mse:.6f}")
    print(f"Std  test MSE: {std_mse:.6f}")

    return mean_mse, std_mse, test_mses_arr, best_y_target, best_y_pred
