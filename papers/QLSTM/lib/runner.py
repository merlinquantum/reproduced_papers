from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lib.dataset import data as data_factory
from lib.model import build_model
from lib.rendering import save_losses_plot, save_pickle, save_simulation_plot
from runtime_lib.dtypes import DtypeSpec, coerce_dtype_spec, dtype_label, dtype_torch


def _resolve_model_dtype(
    requested: DtypeSpec | str | None, model_type: str
) -> torch.dtype:
    """Map config value to a torch dtype with sensible defaults per model type."""

    default = torch.float64 if "photonic" in model_type else torch.float32
    resolved = dtype_torch(requested)
    if resolved is not None:
        return resolved

    # Normalize common representations (e.g. {"label": "float64"}).
    spec = coerce_dtype_spec(requested)
    if spec is None or spec.label is None or spec.label == "auto":
        return default
    if spec.torch is not None:
        return spec.torch

    # Torch is required to return a torch.dtype.
    raise ValueError(
        f"Unsupported dtype '{spec.label}' for model_type '{model_type}'. "
        "Use one of float16/bfloat16/float32/float64 or 'auto'."
    )


def train_and_evaluate(cfg, run_dir: Path) -> None:
    log = logging.getLogger(__name__)

    exp = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    global_dtype: DtypeSpec | None = cfg.get("dtype")
    requested_model_dtype = model_cfg.get("dtype", global_dtype)
    model_dtype = _resolve_model_dtype(requested_model_dtype, model_cfg["type"])

    if exp["generator"] == "csv" and exp.get("csv_path"):
        gen = data_factory.get("csv", path=exp["csv_path"])  # type: ignore[arg-type]
    else:
        gen = data_factory.get(exp["generator"])

    dataset_name = getattr(gen, "name", exp["generator"])
    dataset_extra = ""
    if exp["generator"] == "csv" and exp.get("csv_path"):
        dataset_extra = f" (path={exp['csv_path']})"
    log.info(
        "Dataset: %s [key=%s]%s | seq_length=%d train_split=%.2f",
        dataset_name,
        exp["generator"],
        dataset_extra,
        exp["seq_length"],
        exp["train_split"],
    )

    x, y = gen.get_data(seq_len=exp["seq_length"], max_samples=exp.get("max_samples"))
    x = x.to(dtype=model_dtype)
    y = y.to(dtype=model_dtype)
    n_train = int(exp["train_split"] * len(x))
    log.info(
        "Dataset samples=%d (train=%d, test=%d)", len(x), n_train, len(x) - n_train
    )
    dtype_source = (
        "model.dtype"
        if dtype_label(model_cfg.get("dtype")) is not None
        else "cfg.dtype"
        if dtype_label(global_dtype) is not None
        else "auto"
    )
    log.info("Using model dtype %s (source=%s)", model_dtype, dtype_source)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    x_train_in = x_train.unsqueeze(2)
    x_test_in = x_test.unsqueeze(2)

    device = torch.device(cfg["device"])
    model = build_model(
        model_cfg["type"],
        1,
        model_cfg["hidden_size"],
        model_cfg["vqc_depth"],
        1,
        use_preencoders=bool(model_cfg.get("use_preencoders", False)),
        shots=int(model_cfg.get("shots", 0)),
        use_photonic_head=bool(model_cfg.get("use_photonic_head", False)),
        dtype=model_dtype,
    ).to(device=device, dtype=model_dtype)
    opt = torch.optim.RMSprop(model.parameters(), lr=train_cfg["lr"])
    mse = nn.MSELoss()

    train_losses: list[float] = []
    test_losses: list[float] = []
    best_test: float = float("inf")
    best_epoch: int = 0
    best_train_at_best: float = float("nan")

    snapshot_set = set(exp.get("snapshot_epochs", []) or [])
    snapshot_set.add(train_cfg["epochs"])
    for epoch in range(train_cfg["epochs"]):
        model.train()
        perm = torch.randperm(x_train_in.size(0))
        batch_losses: list[float] = []
        for i in range(0, x_train_in.size(0), train_cfg["batch_size"]):
            idx = perm[i : i + train_cfg["batch_size"]]
            xb = x_train_in[idx].to(device)
            yb = y_train[idx].to(device)
            pred, _ = model(xb)
            pred_last = pred.transpose(0, 1)[-1].view(-1)
            loss = mse(pred_last, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        tr_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
        with torch.no_grad():
            model.eval()
            ptest, _ = model(x_test_in.to(device))
            te_loss = mse(ptest.transpose(0, 1)[-1].view(-1), y_test.to(device)).item()
        train_losses.append(tr_loss)
        test_losses.append(float(te_loss))
        ep = epoch + 1
        if te_loss < best_test:
            best_test = float(te_loss)
            best_epoch = ep
            best_train_at_best = float(tr_loss)
            best_ckpt = run_dir / "model_best.pth"
            torch.save(model.state_dict(), best_ckpt)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info("Epoch %d: train %.6f test %.6f", epoch + 1, tr_loss, te_loss)

        if ep in snapshot_set:
            log.info("Snapshot Epoch %d: train %.6f test %.6f", ep, tr_loss, te_loss)
            save_losses_plot(train_losses, str(run_dir), prefix="train")
            save_losses_plot(test_losses, str(run_dir), prefix="test")
            with torch.no_grad():
                model.eval()
                full_pred, _ = model(x.unsqueeze(2).to(device))
                full_last = (
                    full_pred.transpose(0, 1)[-1].view(-1).detach().cpu().numpy()
                )
                save_simulation_plot(
                    y.detach().cpu().numpy(),
                    full_last,
                    str(run_dir),
                    prefix=f"simulation_e{ep}",
                    vline_x=n_train,
                    width=exp.get("plot_width"),
                    title=f"Epoch {ep} — train MSE {tr_loss:.6f}, test MSE {te_loss:.6f}",
                )
                np.savez(
                    run_dir / f"simulation_data_e{ep}.npz",
                    y=y.detach().cpu().numpy(),
                    y_pred=full_last,
                    n_train=n_train,
                    epoch=ep,
                    train_mse=tr_loss,
                    test_mse=te_loss,
                )

    ckpt_path = run_dir / "model_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    save_pickle(train_losses, str(run_dir), "last_TRAINING_LOSS")
    save_pickle(test_losses, str(run_dir), "last_TESTING_LOSS")
    try:
        meta_path = run_dir / "best_epoch.txt"
        meta_path.write_text(
            f"best_epoch={best_epoch}\nbest_test_mse={best_test:.8f}\n"
            f"train_mse_at_best={best_train_at_best:.8f}\n",
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover
        logging.getLogger("train").warning("Failed to write best_epoch.txt: %s", exc)
    try:
        csv_path = run_dir / "losses.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("epoch,train_mse,test_mse\n")
            for i, (tr, te) in enumerate(zip(train_losses, test_losses), start=1):
                handle.write(f"{i},{tr:.8f},{te:.8f}\n")
        log.info("Wrote losses CSV: %s", csv_path)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to write losses.csv: %s", exc)
    log.info("Saved final checkpoint: %s", ckpt_path)
