#!/usr/bin/env python3
"""
Benchmark full-size Retina configs across devices and precision modes.

Usage:
    python scripts/benchmarks/benchmark_device_profile.py
    python scripts/benchmarks/benchmark_device_profile.py --devices cpu cuda:0
    python scripts/benchmarks/benchmark_device_profile.py --models A D E F --precision-mode gpu_friendly

This runs the real training pipeline for the selected full Retina configs,
typically for 1 epoch, and writes a JSON/CSV report under outdir/.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pathlib
import queue
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import validate_run_config
from lib.data import get_medmnist_loaders
from lib.models import QVTModel
from lib.runner import resolve_runtime_dtype
from lib.training import train

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only
    resource = None


MODEL_CONFIGS = {
    "A": "configs/model_a_retina.json",
    "B": "configs/model_b_retina.json",
    "C": "configs/model_c_retina.json",
    "D": "configs/model_d_retina.json",
    "D_full": "configs/model_d_full_retina.json",
    "E": "configs/model_e_retina.json",
    "F": "configs/model_f_retina.json",
}


def _build_config(
    *,
    model: str,
    device: str,
    precision_mode: str,
    epochs: int,
    data_root: str,
    num_workers: int,
    circuit_family: str,
    seed: int,
) -> dict:
    cfg_path = ROOT / MODEL_CONFIGS[model]
    cfg = json.loads(cfg_path.read_text())
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["epochs"] = epochs
    cfg["data_root"] = data_root
    cfg["num_workers"] = num_workers
    cfg["precision_mode"] = precision_mode
    cfg["circuit_family"] = circuit_family

    if model == "D_full":
        cfg["model_type"] = "D"
        cfg["compound_readout"] = "full_sector"
    else:
        cfg["model_type"] = model

    if circuit_family == "butterfly":
        if model in {"D", "D_full"}:
            cfg["use_cls_token"] = False
        if model == "E":
            cfg["use_cls_token"] = False
        if model in {"D", "D_full", "E"}:
            cfg["embed_dim"] = 16

    return validate_run_config(cfg)


def _prepare_model(cfg: dict, device: torch.device) -> tuple[QVTModel, tuple[int, int, int, int]]:
    trn, val, tst, nc = get_medmnist_loaders(
        cfg.get("dataset", "retinamnist"),
        cfg.get("batch_size", 32),
        cfg.get("data_root", "data/QVT"),
        num_workers=cfg.get("num_workers", 2),
        seed=cfg.get("seed", 42),
    )
    sample = next(iter(trn))[0]
    model = QVTModel(
        model_type=cfg.get("model_type", "B"),
        img_size=sample.shape[2],
        in_channels=sample.shape[1],
        patch_size=cfg.get("patch_size", 7),
        embed_dim=cfg.get("embed_dim", 16),
        n_layers=cfg.get("n_layers", 4),
        n_classes=nc,
        use_cls_token=cfg.get("use_cls_token", True),
        use_pos_embed=cfg.get("use_pos_embed", True),
        compound_readout=cfg.get("compound_readout", "cross_only"),
        circuit_family=cfg.get("circuit_family"),
        n_regions_per_side=cfg.get("n_regions_per_side", 2),
        n_patches_per_side=cfg.get("n_patches_per_side", 2),
        use_rpp_attention=cfg.get("use_rpp_attention", True),
        device=device,
    ).to(device)
    return model, (trn, val, tst, nc)


def _run_single_benchmark(
    *,
    model: str,
    device_name: str,
    precision_mode: str,
    epochs: int,
    out_root: str,
    data_root: str,
    num_workers: int,
    circuit_family: str,
    seed: int,
    result_queue,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = _build_config(
        model=model,
        device=device_name,
        precision_mode=precision_mode,
        epochs=epochs,
        data_root=data_root,
        num_workers=num_workers,
        circuit_family=circuit_family,
        seed=seed,
    )
    dtype = resolve_runtime_dtype(cfg, None)
    torch.set_default_dtype(dtype)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device_name}' requested but CUDA is not available.")

    device = torch.device(device_name)
    model_obj, loaders = _prepare_model(cfg, device)
    trn, val, tst, nc = loaders

    run_dir = pathlib.Path(out_root) / f"{model}_{circuit_family}_{precision_mode}_{device_name.replace(':', '_')}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    max_cuda_memory_bytes = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    wall_start = time.perf_counter()
    results = train(model_obj, trn, val, tst, nc, cfg, str(run_dir), device, resume_checkpoint=None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        max_cuda_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    wall_time_s = time.perf_counter() - wall_start

    max_rss_kb = None
    if resource is not None:
        max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    first_epoch = results["history"][0] if results.get("history") else {}
    result_queue.put(
        {
            "model": model,
            "device": device_name,
            "precision_mode": precision_mode,
            "circuit_family": circuit_family,
            "dtype": str(dtype).replace("torch.", ""),
            "epochs": epochs,
            "batch_size": cfg.get("batch_size"),
            "n_layers": cfg.get("n_layers"),
            "embed_dim": cfg.get("embed_dim"),
            "epoch_time_s": first_epoch.get("time_s"),
            "reported_total_time_s": results.get("total_time_s"),
            "wall_time_s": round(wall_time_s, 2),
            "max_rss_kb": max_rss_kb,
            "max_cuda_memory_bytes": max_cuda_memory_bytes,
            "outdir": str(run_dir),
        }
    )


def _launch_benchmark(**kwargs) -> dict:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_run_single_benchmark, kwargs={**kwargs, "result_queue": result_queue})
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Benchmark failed for model={kwargs['model']} device={kwargs['device_name']} "
            f"precision_mode={kwargs['precision_mode']}."
        )
    try:
        return result_queue.get_nowait()
    except queue.Empty as exc:  # pragma: no cover - defensive
        raise RuntimeError("Benchmark process exited without producing results.") from exc


def _write_reports(rows: list[dict], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_summary.json"
    csv_path = out_dir / "benchmark_summary.csv"

    json_path.write_text(json.dumps(rows, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    print(
        f"{'Model':<8} {'Device':<8} {'Prec':<12} {'Epoch(s)':>9} "
        f"{'Total(s)':>9} {'RSS(MB)':>9} {'CUDA(MB)':>10}"
    )
    print("-" * 74)
    for row in rows:
        rss_mb = "-" if row["max_rss_kb"] is None else f"{row['max_rss_kb'] / 1024:.0f}"
        cuda_mb = "-"
        if row["max_cuda_memory_bytes"] is not None:
            cuda_mb = f"{row['max_cuda_memory_bytes'] / (1024 ** 2):.0f}"
        print(
            f"{row['model']:<8} {row['device']:<8} {row['precision_mode']:<12} "
            f"{row['epoch_time_s']:>9} {row['reported_total_time_s']:>9} {rss_mb:>9} {cuda_mb:>10}"
        )

    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], {})[row["device"]] = row

    comparisons = []
    for model, per_device in grouped.items():
        if "cpu" in per_device:
            cpu_row = per_device["cpu"]
            for device, row in per_device.items():
                if device == "cpu":
                    continue
                speedup = None
                if row["reported_total_time_s"] and cpu_row["reported_total_time_s"]:
                    speedup = cpu_row["reported_total_time_s"] / row["reported_total_time_s"]
                comparisons.append((model, device, speedup))

    if comparisons:
        print("\nCPU -> accelerator speedups")
        print(f"{'Model':<8} {'Device':<8} {'Speedup':>8}")
        print("-" * 28)
        for model, device, speedup in comparisons:
            value = "-" if speedup is None else f"{speedup:.2f}x"
            print(f"{model:<8} {device:<8} {value:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QVT full Retina configs across devices.")
    parser.add_argument("--models", nargs="+", default=["A", "B", "C", "D", "D_full", "E", "F"])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--precision-mode", default="baseline", choices=["baseline", "gpu_friendly"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--circuit-family", default="generic", choices=["generic", "butterfly"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/QVT")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--outdir", default="outdir/device_profile")
    args = parser.parse_args()

    rows = []
    for device_name in args.devices:
        for model in args.models:
            print(
                f"Running {model} on {device_name} "
                f"(precision_mode={args.precision_mode}, family={args.circuit_family}, epochs={args.epochs})"
            )
            row = _launch_benchmark(
                model=model,
                device_name=device_name,
                precision_mode=args.precision_mode,
                epochs=args.epochs,
                out_root=str(ROOT / args.outdir),
                data_root=str(ROOT / args.data_root),
                num_workers=args.num_workers,
                circuit_family=args.circuit_family,
                seed=args.seed,
            )
            rows.append(row)

    rows.sort(key=lambda row: (row["model"], row["device"]))
    _write_reports(rows, ROOT / args.outdir)
    _print_summary(rows)


if __name__ == "__main__":
    main()
