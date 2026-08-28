"""Training loop for QVT reproduction.

Paper setup: Adam, lr=1e-3, decay ×0.1 at epochs 50 & 75, 100 epochs, batch 32.

Logs per epoch:
  - train loss / acc
  - val loss / acc / auc
  - gradient norm (total and per-component)
  - sector mass (Model D only)
  - wall-clock time
  - learning rate

All metrics are saved to results.json for post-hoc figure generation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _auc(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Compute ROC AUC.

    Avoid importing scikit-learn on the hot path (it is expensive to import and
    dominates smoke-test runtime). For binary classification we use a small
    NumPy implementation; for multi-class we fall back to scikit-learn when
    available.
    """

    try:
        labels = np.asarray(labels).astype(int)
        probs = np.asarray(probs)
        if n_classes == 2:
            scores = probs[:, 1].astype(float)
            # Rank-based AUC (equivalent to Mann-Whitney U statistic).
            order = np.argsort(scores, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
            pos = labels == 1
            n_pos = int(pos.sum())
            n_neg = int((~pos).sum())
            if n_pos == 0 or n_neg == 0:
                return 0.0
            sum_ranks_pos = float(ranks[pos].sum())
            auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            return float(auc)

        # Multi-class: keep the reference implementation when sklearn is present.
        from sklearn.metrics import roc_auc_score  # type: ignore
        from sklearn.preprocessing import label_binarize  # type: ignore

        lb = label_binarize(labels, classes=list(range(n_classes)))
        return float(roc_auc_score(lb, probs, multi_class="ovr", average="macro"))
    except Exception:
        return 0.0


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def _cast(batch, device):
    """Cast images to default dtype and labels to long on device."""
    imgs = batch[0].to(device=device, dtype=torch.get_default_dtype())
    labs = batch[1].to(device).squeeze().long()
    return imgs, labs


def _atomic_write_json(payload: dict, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _atomic_torch_save(payload, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _write_progress_snapshot(
    *,
    path: str,
    model_type: str,
    config: dict,
    history: list[dict],
    best_auc: float,
    best_epoch: int,
    epoch: int,
    elapsed_time_s: float,
    resumed_from_checkpoint: bool,
) -> None:
    payload = {
        "model_type": model_type,
        "circuit_family": config.get("circuit_family", "generic"),
        "data_regime": config.get("data_regime", "standard"),
        "dataset": config.get("dataset", "?"),
        "seed": config.get("seed", "?"),
        "epoch": epoch,
        "epochs": config.get("epochs", 100),
        "best_val_auc": None if best_epoch == 0 else best_auc,
        "best_epoch": best_epoch,
        "elapsed_time_s": round(elapsed_time_s, 1),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "config": config,
        "history": history,
    }
    _atomic_write_json(payload, path)


def _resume_config_differences(current: dict, previous: dict) -> list[str]:
    ignored_keys = {"epochs"}
    diffs = []
    for key in sorted(set(current) | set(previous)):
        if key in ignored_keys:
            continue
        if current.get(key) != previous.get(key):
            diffs.append(key)
    return diffs


def _load_resume_checkpoint(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    device: torch.device,
    config: dict,
    strict: bool,
) -> tuple[int, float, int, list[dict], float, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        if strict:
            raise
        logger.warning(
            "Resume checkpoint at %s is incompatible with the current model shape; "
            "starting a fresh run instead. Details: %s",
            checkpoint_path,
            exc,
        )
        return 1, float("-inf"), 0, [], 0.0, False

    opt.load_state_dict(checkpoint["optimizer_state_dict"])
    sched.load_state_dict(checkpoint["scheduler_state_dict"])
    _optimizer_to_device(opt, device)
    _restore_rng_state(checkpoint.get("rng_state"))

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_auc = float(checkpoint.get("best_val_auc", float("-inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    history = list(checkpoint.get("history", []))
    elapsed_before_resume = float(checkpoint.get("elapsed_time_s", 0.0))

    checkpoint_cfg = checkpoint.get("config")
    if checkpoint_cfg is not None:
        diff_keys = _resume_config_differences(config, checkpoint_cfg)
        if diff_keys:
            logger.warning(
                "Resume checkpoint config differs from current config on keys: %s; continuing with current config values",
                ", ".join(diff_keys),
            )

    return start_epoch, best_auc, best_epoch, history, elapsed_before_resume, True


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_probs, all_labels = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        imgs, labs = _cast(batch, device)
        logits = model(imgs)
        total_loss += criterion(logits, labs).item() * imgs.shape[0]
        all_probs.append(torch.softmax(logits, -1).cpu())
        all_labels.append(labs.cpu())
        n += imgs.shape[0]
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = float((probs.argmax(1) == labels).mean())
    return {
        "acc": acc,
        "auc": _auc(probs, labels, n_classes),
        "loss": total_loss / max(n, 1),
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    n_classes,
    config,
    outdir,
    device,
    resume_checkpoint: str | None = None,
    resume_strict: bool = False,
):
    epochs = config.get("epochs", 100)
    lr = config.get("lr", 1e-3)
    lr_quantum = config.get("lr_quantum", None)  # separate lr for interferometers
    milestones = config.get("lr_milestones", [50, 75])
    gamma = config.get("lr_gamma", 0.1)
    model_type = config.get("model_type", "?")

    if lr_quantum is not None:
        # split into quantum (attn_layers) and classical (everything else)
        quantum_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "attn_layers." in n
        ]
        classical_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "attn_layers." not in n
        ]
        param_groups = [
            {"params": quantum_params, "lr": lr_quantum},
            {"params": classical_params, "lr": lr},
        ]
        opt = torch.optim.Adam(param_groups)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=milestones, gamma=gamma
    )
    criterion = nn.CrossEntropyLoss()
    os.makedirs(outdir, exist_ok=True)
    best_path = os.path.join(outdir, "best.pt")
    last_path = os.path.join(outdir, "last.pt")
    progress_path = os.path.join(outdir, "progress.json")
    results_path = os.path.join(outdir, "results.json")

    n_batches = len(train_loader)
    best_auc, best_epoch = float("-inf"), 0
    history: list[dict] = []
    start_epoch = 1
    resumed_from_checkpoint = False
    elapsed_before_resume = 0.0

    if resume_checkpoint is not None:
        (
            start_epoch,
            best_auc,
            best_epoch,
            history,
            elapsed_before_resume,
            resumed_from_checkpoint,
        ) = _load_resume_checkpoint(
            model=model,
            opt=opt,
            sched=sched,
            checkpoint_path=resume_checkpoint,
            device=device,
            config=config,
            strict=resume_strict,
        )

    # ── header ──
    param_counts = (
        model.count_trainable_params()
        if hasattr(model, "count_trainable_params")
        else {}
    )
    lr_str = f"lr={lr}"
    if lr_quantum is not None:
        lr_str += f"  lr_quantum={lr_quantum}"
    logger.info("=" * 65)
    logger.info(
        f"Model {model_type}  |  dataset={config.get('dataset', '?')}  "
        f"|  epochs={epochs}  {lr_str}"
    )
    logger.info(f"Params: {json.dumps(param_counts)}")
    logger.info(f"Batches/epoch: {n_batches}  |  device={device}")
    if resumed_from_checkpoint:
        logger.info(
            f"Resuming from {resume_checkpoint}  |  next epoch={start_epoch}  |  best val AUC {best_auc:.4f} @ ep {best_epoch}"
        )
    logger.info("=" * 65)

    t_start = time.time() - elapsed_before_resume

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        tloss, correct, n = 0.0, 0, 0
        sector_masses = []

        for bi, batch in enumerate(train_loader):
            imgs, labs = _cast(batch, device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            opt.step()

            tloss += loss.item() * imgs.shape[0]
            correct += (logits.argmax(-1) == labs).sum().item()
            n += imgs.shape[0]

            # collect sector mass for Model D
            if hasattr(model, "sector_masses") and model.sector_masses:
                for sm in model.sector_masses:
                    if isinstance(sm, dict):
                        # full_sector mode: sm["sector_masses"] is a dict
                        sector_masses.append(sm.get("sector_masses", {}))
                    elif isinstance(sm, torch.Tensor):
                        # cross_only mode: sm is a [B] tensor
                        sector_masses.append({"cross": sm.mean().item()})
                    else:
                        sector_masses.append({"cross": float(sm)})

            # batch progress every 25%
            if n_batches > 4 and (bi + 1) % max(1, n_batches // 4) == 0:
                logger.debug(f"  batch {bi + 1}/{n_batches}  loss={loss.item():.4f}")

        sched.step()
        gn = _grad_norm(model)
        t_epoch = time.time() - t0

        # validate
        val = evaluate(model, val_loader, device, n_classes)

        rec = {
            "epoch": epoch,
            "train_loss": tloss / n,
            "train_acc": correct / n,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_auc": val["auc"],
            "grad_norm": gn,
            "lr": opt.param_groups[0]["lr"],
            "lr_quantum": opt.param_groups[0]["lr"] if lr_quantum is not None else None,
            "lr_classical": opt.param_groups[1]["lr"]
            if lr_quantum is not None
            else None,
            "time_s": round(t_epoch, 2),
        }
        if sector_masses:
            # sector_masses is a list of dicts with varying keys:
            # D cross_only: {"cross"}, D full: {"cross","pp","ff"},
            # E: {"pp"}, F: {"triple_cross","rpp"}
            for key in ("cross", "pp", "ff", "triple_cross", "rpp"):
                vals = [s.get(key) for s in sector_masses if key in s]
                if vals:
                    rec[f"sector_mass_{key}"] = float(np.mean(vals))
            # backward compat alias
            if "sector_mass_cross" in rec:
                rec["sector_mass_mean"] = rec["sector_mass_cross"]

        history.append(rec)

        if best_epoch == 0 or val["auc"] >= best_auc:
            best_auc = val["auc"]
            best_epoch = epoch
            _atomic_torch_save(model.state_dict(), best_path)

        elapsed_time = time.time() - t_start
        _write_progress_snapshot(
            path=progress_path,
            model_type=model_type,
            config=config,
            history=history,
            best_auc=best_auc,
            best_epoch=best_epoch,
            epoch=epoch,
            elapsed_time_s=elapsed_time,
            resumed_from_checkpoint=resumed_from_checkpoint,
        )
        _atomic_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "best_val_auc": best_auc,
                "best_epoch": best_epoch,
                "history": history,
                "elapsed_time_s": elapsed_time,
                "config": config,
                "rng_state": _capture_rng_state(),
            },
            last_path,
        )

        # log every epoch (concise) + extra detail every 10
        parts = [
            f"Ep {epoch:3d}/{epochs}",
            f"loss {rec['train_loss']:.4f}",
            f"acc {rec['train_acc']:.3f}",
            f"| val {val['acc']:.3f}/{val['auc']:.3f}",
            f"| gnorm {gn:.2e}",
            f"| {t_epoch:.1f}s",
        ]
        _sector_abbrev = {
            "sector_mass_cross": "C",
            "sector_mass_pp": "PP",
            "sector_mass_ff": "FF",
            "sector_mass_triple_cross": "TC",
            "sector_mass_rpp": "RPP",
        }
        sector_keys = [k for k in _sector_abbrev if k in rec]
        if sector_keys:
            sector_str = " ".join(
                f"{_sector_abbrev[k]}={rec[k]:.2f}" for k in sector_keys
            )
            parts.append(f"| sectors {sector_str}")
        logger.info("  ".join(parts))

        if epoch % 10 == 0:
            elapsed = timedelta(seconds=int(time.time() - t_start))
            logger.info(
                f"  ── {elapsed} elapsed  |  best val AUC {best_auc:.4f} @ ep {best_epoch}"
            )

    # ── test ──
    logger.info("-" * 65)
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    else:
        logger.warning("best.pt not found; evaluating current in-memory model weights")
    test = evaluate(model, test_loader, device, n_classes)
    total_time = time.time() - t_start

    results = {
        "model_type": model_type,
        "circuit_family": config.get("circuit_family", "generic"),
        "data_regime": config.get("data_regime", "standard"),
        "dataset": config.get("dataset", "?"),
        "seed": config.get("seed", "?"),
        "test_acc": test["acc"],
        "test_auc": test["auc"],
        "test_loss": test["loss"],
        "best_val_auc": None if best_epoch == 0 else best_auc,
        "best_epoch": best_epoch,
        "total_time_s": round(total_time, 1),
        "param_counts": param_counts,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "last_completed_epoch": history[-1]["epoch"] if history else 0,
        "config": config,
        "history": history,
    }

    _atomic_write_json(results, results_path)

    logger.info(
        f"Test: acc={test['acc']:.4f}  auc={test['auc']:.4f}  "
        f"(best val AUC {results['best_val_auc']:.4f} @ ep {best_epoch})"
    )
    logger.info(
        f"Total time: {timedelta(seconds=int(total_time))}  |  saved → {results_path}"
    )
    logger.info("=" * 65)

    return results
