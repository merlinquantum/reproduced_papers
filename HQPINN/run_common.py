import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .layer_merlin import make_merlin_processor
from .utils import get_latest_checkpoint, load_model


def _resolve_checkpoint(
    *,
    ckpt_dir: str,
    case_prefix: str,
    use_models_subdir: bool,
) -> str | None:
    # SEE/DEE checkpoints live in `<ckpt_dir>/models`, while DHO uses `ckpt_dir` directly.
    checkpoint_root = os.path.join(ckpt_dir, "models") if use_models_subdir else ckpt_dir
    ckpt_path = get_latest_checkpoint(checkpoint_root, case_prefix)
    if ckpt_path is None:
        return None
    return ckpt_path


def _load_model_for_mode(
    *,
    mode: str,
    backend: str,
    ckpt_path: str,
    model_factory: Callable[[object], nn.Module],
    warn_if_backend_ignored: bool = True,
    print_remote_header: bool = True,
) -> nn.Module:
    # `run` always loads a local model instance.
    # `remote` rebuilds the model with a Merlin remote processor attached.
    if mode == "run":
        if warn_if_backend_ignored and backend.lower() != "local":
            print(
                f"Backend '{backend}' is not used in run mode; use mode='remote'."
            )
        model = load_model(ckpt_path, lambda processor=None: model_factory(processor))

    elif mode == "remote":
        if print_remote_header:
            print("=== REMOTE MODE ===")
        # Force explicit cloud/simulator backend selection in remote mode.
        if backend.lower() == "local":
            raise ValueError(
                "backend='local' is not allowed in remote mode. "
                "Use: sim:ascella."
            )
        processor = make_merlin_processor(backend)
        model = load_model(
            ckpt_path,
            lambda processor=processor: model_factory(processor),
            processor=processor,
        )

    else:
        raise ValueError("mode must be 'run' or 'remote'")

    model.eval()
    return model


def run_density_inference_mode(
    *,
    mode: str,
    backend: str,
    ckpt_dir: str,
    case_prefix: str,
    n_photons: int,
    timestamp: str,
    model_factory: Callable[[object], nn.Module],
    save_plot_fn: Callable[..., str],
) -> str | None:
    """
    Shared helper for SEE/DEE inference modes (`run` and `remote`).
    """
    ckpt_path = _resolve_checkpoint(
        ckpt_dir=ckpt_dir, case_prefix=case_prefix, use_models_subdir=True
    )
    if ckpt_path is None:
        return None

    model = _load_model_for_mode(
        mode=mode,
        backend=backend,
        ckpt_path=ckpt_path,
        model_factory=model_factory,
        warn_if_backend_ignored=True,
        print_remote_header=True,
    )

    # Delegate plotting to the caller so each domain keeps its own plot implementation.
    png_path = save_plot_fn(
        model=model,
        ckpt_dir=ckpt_dir,
        case_prefix=case_prefix,
        n_photons=n_photons,
        timestamp=timestamp,
        backend=backend,
    )
    print(f"Figure saved to: {png_path}")
    return png_path


def run_series_inference_mode(
    *,
    mode: str,
    backend: str,
    ckpt_dir: str,
    case_prefix: str,
    model_factory: Callable[[object], nn.Module],
    make_time_grid: Callable[[], torch.Tensor],
    exact_fn: Callable[[np.ndarray], np.ndarray],
    plot_fn: Callable[[np.ndarray, np.ndarray, torch.Tensor], None],
) -> None:
    """
    Shared helper for DHO inference modes (`run` and `remote`).
    """
    ckpt_path = _resolve_checkpoint(
        ckpt_dir=ckpt_dir, case_prefix=case_prefix, use_models_subdir=False
    )
    if ckpt_path is None:
        return

    model = _load_model_for_mode(
        mode=mode,
        backend=backend,
        ckpt_path=ckpt_path,
        model_factory=model_factory,
        warn_if_backend_ignored=False,
        print_remote_header=True,
    )

    if mode == "remote":
        print(f"Executed remote model on simulator from checkpoint: {ckpt_path}")

    with torch.no_grad():
        t = make_time_grid()
        u_pred = model(t)
        # Normalize output to a flat NumPy vector for downstream plotting.
        if isinstance(u_pred, torch.Tensor):
            u_pred = u_pred.detach().cpu().numpy().flatten()
        else:
            u_pred = np.asarray(u_pred).flatten()
        u_ex = exact_fn(t.cpu().numpy().flatten())

    plot_fn(u_pred, u_ex, t)
