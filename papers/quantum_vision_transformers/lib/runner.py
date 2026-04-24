from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from runtime_lib.dtypes import dtype_label, dtype_torch
from runtime_lib.seed import seed_everything

from .config import validate_run_config
from .data import get_medmnist_loaders
from .models import QVTModel
from .training import train


def _resolve_runtime_dtype(cfg: dict) -> torch.dtype:
    """Resolve the runtime real dtype from the normalized config."""

    precision_mode = cfg.get("precision_mode", "baseline")
    if precision_mode == "gpu_friendly":
        return torch.float32

    dtype_value = cfg.get("dtype")
    resolved = dtype_torch(dtype_value)
    if resolved is not None:
        return resolved

    label = dtype_label(dtype_value) or "float64"
    return getattr(torch, label)


def _list_models() -> None:
    models = {
        "A": "Orthogonal Patch-wise (no attention)",
        "B": "Quantum Orthogonal Transformer (overlap attention)",
        "C": "Direct Quantum Attention (pragmatic hybrid)",
        "D": "Compound Transformer (2-photon compound matrix)",
        "E": "Multi-sector Attention (shared circuit, 1ph+2ph)",
        "F": "Hierarchical Compound (3-photon, region+patch+feature)",
        "VisionTransformer": "Classical baseline from the paper appendix",
        "OrthoFNN": "Quantum fully connected baseline from the paper",
    }
    for key, value in models.items():
        print(f"  {key}  {value}")


def _build_model(cfg: dict, sample: torch.Tensor, nc: int, runtime_dtype: torch.dtype,
                 device: torch.device) -> QVTModel:
    return QVTModel(
        model_type=cfg.get("model_type", "B"),
        img_size=sample.shape[2],
        in_channels=sample.shape[1],
        patch_size=cfg.get("patch_size", 7),
        embed_dim=cfg.get("embed_dim", 16),
        n_layers=cfg.get("n_layers", 4),
        n_classes=nc,
        use_cls_token=cfg.get("use_cls_token", True),
        use_pos_embed=cfg.get("use_pos_embed", True),
        image_embed_grayscale=cfg.get("image_embed_grayscale", False),
        compound_readout=cfg.get("compound_readout", "cross_only"),
        circuit_family=cfg.get("circuit_family", "generic"),
        n_regions_per_side=cfg.get("n_regions_per_side", 2),
        n_patches_per_side=cfg.get("n_patches_per_side", 2),
        use_rpp_attention=cfg.get("use_rpp_attention", True),
        device=device,
    ).to(device=device, dtype=runtime_dtype)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    log = logging.getLogger("QVT")

    cfg = validate_run_config(dict(cfg))

    seed_value = int(cfg.get("seed", 42))
    seed_everything(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if cfg.get("list_models"):
        _list_models()
        return

    runtime_dtype = _resolve_runtime_dtype(cfg)
    torch.set_default_dtype(runtime_dtype)

    device_cfg = str(cfg.get("device", "cpu"))
    if device_cfg.startswith("cuda") and not torch.cuda.is_available():
        log.warning("Config requests %s but CUDA is not available; falling back to CPU", device_cfg)
        device_cfg = "cpu"
    device = torch.device(device_cfg)

    ds = cfg.get("dataset", "retinamnist")
    log.info("Loading %s", ds)
    trn, val, tst, nc = get_medmnist_loaders(
        ds,
        cfg.get("batch_size", 32),
        cfg.get("data_root", "data/QVT"),
        num_workers=cfg.get("num_workers", 2),
        seed=seed_value,
        train_subset_size=cfg.get("train_subset_size"),
        train_subset_seed=cfg.get("train_subset_seed"),
        train_subset_mode=cfg.get("train_subset_mode", "stratified"),
    )
    sample = next(iter(trn))[0]

    model = _build_model(cfg, sample, nc, runtime_dtype, device)
    param_counts = model.count_trainable_params()
    log.info("Params: %s", json.dumps(param_counts))
    if cfg.get("count_params"):
        return

    outdir = Path(run_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / "last.pt"
    has_resume_checkpoint = checkpoint_path.exists()
    resume_mode = str(cfg.get("resume", "auto"))
    resume_checkpoint: Path | None
    if resume_mode == "must" and not has_resume_checkpoint:
        raise FileNotFoundError(
            f"--resume=must was requested, but no checkpoint exists at {checkpoint_path}"
        )
    if resume_mode == "never" or not has_resume_checkpoint:
        resume_checkpoint = None
    else:
        resume_checkpoint = checkpoint_path
        log.info("Found checkpoint at %s; training will resume from it", checkpoint_path)

    train(
        model,
        trn,
        val,
        tst,
        nc,
        cfg,
        str(outdir),
        device,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        resume_strict=(resume_mode == "must"),
    )


__all__ = ["train_and_evaluate"]
