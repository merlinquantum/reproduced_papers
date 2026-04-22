#!/usr/bin/env python3
"""
Quantum Vision Transformers — Photonic Reproduction (MerLin native)

Usage:
    python implementation.py --config configs/model_b_retina.json
    python implementation.py --config configs/model_b_retina.json --seed 42
"""

import argparse, json, logging, os, random, sys
from datetime import datetime
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.config import validate_run_config
from lib.models import QVTModel
from lib.data import get_medmnist_loaders
from lib.training import train


def resolve_runtime_dtype(cfg: dict, cli_dtype: str | None) -> torch.dtype:
    """Resolve the real-valued training dtype from config + CLI overrides."""
    if cli_dtype is not None:
        return getattr(torch, cli_dtype)
    if cfg.get("precision_mode", "baseline") == "gpu_friendly":
        return torch.float32
    return getattr(torch, cfg.get("dtype", "float64"))


def main():
    parser = argparse.ArgumentParser(description="QVT photonic reproduction")
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--resume", choices=("auto", "never", "must"), default="auto")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--count-params", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("QVT")

    if args.list_models:
        for k, v in {"A": "Orthogonal Patch-wise (no attention)",
                      "B": "Quantum Orthogonal Transformer (overlap attention)",
                      "C": "Direct Quantum Attention (pragmatic hybrid)",
                      "D": "Compound Transformer (2-photon compound matrix)",
                      "E": "Multi-sector Attention (shared circuit, 1ph+2ph)",
                      "F": "Hierarchical Compound (3-photon, region+patch+feature)",
                      "VisionTransformer": "Classical baseline from the paper appendix",
                      "OrthoFNN": "Quantum fully connected baseline from the paper"}.items():
            print(f"  {k}  {v}")
        return

    if not args.config:
        parser.error("the following arguments are required: --config")

    with open(args.config) as f:
        cfg = json.load(f)
    if "circuit_family" not in cfg:
        cfg["circuit_family"] = "generic"
    for k, v in [("seed", args.seed), ("device", args.device), ("data_root", args.data_root)]:
        if v is not None:
            cfg[k] = v
    cfg = validate_run_config(cfg)

    seed = cfg.get("seed", 42)
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    dtype = resolve_runtime_dtype(cfg, args.dtype)
    torch.set_default_dtype(dtype)
    device_cfg = cfg.get("device", "cuda")
    if device_cfg.startswith("cuda") and not torch.cuda.is_available():
        log.warning(f"Config requests {device_cfg} but CUDA not available — falling back to CPU")
        device_cfg = "cpu"
    device = torch.device(device_cfg)

    ds = cfg.get("dataset", "retinamnist")
    log.info(f"Loading {ds}")
    trn, val, tst, nc = get_medmnist_loaders(ds, cfg.get("batch_size", 32),
                                              cfg.get("data_root", "data/QVT"),
                                              num_workers=cfg.get("num_workers", 2),
                                              seed=seed,
                                              train_subset_size=cfg.get("train_subset_size"),
                                              train_subset_seed=cfg.get("train_subset_seed"),
                                              train_subset_mode=cfg.get("train_subset_mode", "stratified"))
    sample = next(iter(trn))[0]

    model = QVTModel(
        model_type=cfg.get("model_type", "B"),
        img_size=sample.shape[2], in_channels=sample.shape[1],
        patch_size=cfg.get("patch_size", 7),
        embed_dim=cfg.get("embed_dim", 16),
        n_layers=cfg.get("n_layers", 4),
        n_classes=nc,
        use_cls_token=cfg.get("use_cls_token", True),
        use_pos_embed=cfg.get("use_pos_embed", True),
        image_embed_grayscale=cfg.get("image_embed_grayscale", False),
        compound_readout=cfg.get("compound_readout", "cross_only"),
        circuit_family=cfg.get("circuit_family"),
        n_regions_per_side=cfg.get("n_regions_per_side", 2),
        n_patches_per_side=cfg.get("n_patches_per_side", 2),
        use_rpp_attention=cfg.get("use_rpp_attention", True),
        device=device,
    ).to(device)

    log.info(f"Params: {json.dumps(model.count_trainable_params())}")
    if args.count_params:
        return

    outdir = args.outdir or cfg.get("outdir", f"outdir/run_{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(outdir, exist_ok=True)
    resume_checkpoint = os.path.join(outdir, "last.pt")
    has_resume_checkpoint = os.path.exists(resume_checkpoint)
    if args.resume == "must" and not has_resume_checkpoint:
        raise FileNotFoundError(f"--resume=must was requested, but no checkpoint exists at {resume_checkpoint}")
    if args.resume == "never":
        resume_checkpoint = None
    elif not has_resume_checkpoint:
        resume_checkpoint = None
    else:
        log.info(f"Found checkpoint at {resume_checkpoint}; training will resume from it")

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train(
        model,
        trn,
        val,
        tst,
        nc,
        cfg,
        outdir,
        device,
        resume_checkpoint=resume_checkpoint,
        resume_strict=(args.resume == "must"),
    )


if __name__ == "__main__":
    main()
