"""Runtime entry point for the QML passive-sonar reproduction.

The runner exposes three task families dispatched via ``cfg["task"]``:

  - ``"sav_detection"``: runs SAV (and optionally DEMON) on the
    configured dataset and writes detection statistics.
  - ``"classification"``: trains a classical CNN and/or HQ-CNN on the
    configured dataset and reports per-epoch metrics, confusion matrix,
    and the train/test generalisation gap.
  - ``"merlin_classification"``: same as ``"classification"`` but with the
    photonic HQ-CNN variant — currently raises a clear ImportError when the
    ``merlinquantum`` package is missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from lib.data import (
    FrameSpec,
    build_dataset,
    ensure_deepship_github_sample,
    make_dataloaders,
    split_dataset,
)
from lib.demon import compute_demon
from lib.models import HQCNN, CNNClassifier
from lib.sav import compute_sav
from lib.training import confusion_matrix, evaluate, train_loop

logger = logging.getLogger(__name__)


def _resolve_device(cfg_device) -> torch.device:
    requested = str(cfg_device or "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _run_sav_detection(cfg: dict, run_dir: Path, data_root: Path) -> dict:
    dataset_name = cfg["dataset"]["name"]
    frame_spec = FrameSpec(
        duration_s=cfg["dataset"].get("frame_duration_s", 60.0),
        overlap=cfg["dataset"].get("frame_overlap", 0.25),
        target_sr=cfg["dataset"].get("target_sr", 8000),
    )

    images, labels, class_names = build_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        spectrogram_method="stft",  # not used for the report itself
        frame_spec=frame_spec,
        image_size=cfg["dataset"].get("image_size", 64),
        samples_per_class_synthetic=cfg["dataset"].get("samples_per_class_synthetic", 8),
        duration_synthetic_s=cfg["dataset"].get("duration_synthetic_s", 30.0),
        seed=cfg.get("seed", 1337),
    )
    logger.info("Built dataset %s: %d frames across %d classes", dataset_name, len(labels), len(class_names))

    # Re-generate raw waveforms (without any frame extraction) so the SAV/DEMON
    # algorithms see continuous signals of meaningful duration. We piggy-back
    # on ``build_dataset`` only for label / class metadata above.
    from lib.data import _generate_synthetic_signal

    tonal_sets_shipear = [(47, 91), (110, 175), (63, 130, 220), (38, 80, 150, 305), ()]
    tonal_sets_deepship = [(52, 100, 190), (120, 240), (40, 78, 160), (90, 180, 270)]
    if dataset_name == "shipear":
        tonal_sets = [tuple(float(x) for x in t) for t in tonal_sets_shipear]
        bg = 4
    else:
        tonal_sets = [tuple(float(x) for x in t) for t in tonal_sets_deepship]
        bg = None

    rng = np.random.default_rng(cfg.get("seed", 1337))
    duration = cfg["dataset"].get("duration_synthetic_s", 30.0)
    sr = frame_spec.target_sr
    eta = float(cfg["dataset"].get("threshold_eta", 2.0))
    tol_hz = float(cfg["dataset"].get("peak_tolerance_hz", 3.0))

    sav_records, demon_records = [], []
    traces: dict[str, np.ndarray] = {}
    sav_hits = sav_total = demon_hits = demon_total = 0
    sav_false = demon_false = 0
    # Recall/FA tallies use the union of expected frequencies as ground truth.
    for class_idx, class_name in enumerate(class_names):
        if class_idx == bg:
            continue
        signal = _generate_synthetic_signal(rng, class_idx, duration, sr, list(tonal_sets), bg)
        # Use short segments on synthetic clips so the algorithm sees several
        # variance estimates even at 30 s total. The paper's 45 s segments
        # apply to multi-minute real recordings.
        seg = max(2.0, min(8.0, duration / 4.0))
        sav = compute_sav(signal, sr=sr, segment_duration=seg, stacking=2, threshold_eta=eta)
        demon = compute_demon(
            signal,
            sr=sr,
            band=(20.0, min(0.45 * sr, 3000.0)),
            window_duration=min(1.0, duration / 8.0),
            threshold_eta=eta,
        )

        sav_peak_hz = sav.frequencies[sav.peak_indices]
        demon_peak_hz = demon.frequencies[demon.peak_indices]
        expected = np.asarray(tonal_sets[class_idx], dtype=float)

        # detection: each expected tonal counted as hit if any detected peak is within tol_hz.
        def _hits(detected: np.ndarray, expected: np.ndarray) -> tuple[int, int]:
            if expected.size == 0:
                return 0, 0
            if detected.size == 0:
                return 0, int(expected.size)
            d = np.abs(detected[:, None] - expected[None, :]).min(axis=0)
            return int((d <= tol_hz).sum()), int(expected.size)

        def _false(detected: np.ndarray, expected: np.ndarray) -> int:
            if detected.size == 0:
                return 0
            if expected.size == 0:
                return int(detected.size)
            d = np.abs(detected[:, None] - expected[None, :]).min(axis=1)
            return int((d > tol_hz).sum())

        s_hit, s_total = _hits(sav_peak_hz, expected)
        d_hit, d_total = _hits(demon_peak_hz, expected)
        sav_hits += s_hit
        sav_total += s_total
        demon_hits += d_hit
        demon_total += d_total
        sav_false += _false(sav_peak_hz, expected)
        demon_false += _false(demon_peak_hz, expected)

        sav_records.append(
            {
                "class": class_name,
                "threshold": sav.threshold,
                "num_peaks": int(sav.peak_indices.size),
                "peak_freqs_hz": sav_peak_hz.tolist(),
                "expected_freqs_hz": expected.tolist(),
                "hits": s_hit,
                "expected_count": s_total,
            }
        )
        demon_records.append(
            {
                "class": class_name,
                "threshold": demon.threshold,
                "num_peaks": int(demon.peak_indices.size),
                "peak_freqs_hz": demon_peak_hz.tolist(),
                "expected_freqs_hz": expected.tolist(),
                "hits": d_hit,
                "expected_count": d_total,
            }
        )
        traces[f"sav_{class_name}_freqs"] = sav.frequencies
        traces[f"sav_{class_name}_vstack"] = sav.v_stack
        traces[f"demon_{class_name}_freqs"] = demon.frequencies
        traces[f"demon_{class_name}_spectrum"] = demon.spectrum

    def _safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    metrics = {
        "sav_detection_rate": _safe_div(sav_hits, sav_total),
        "demon_detection_rate": _safe_div(demon_hits, demon_total),
        "sav_false_peaks": sav_false,
        "demon_false_peaks": demon_false,
        "threshold_eta": eta,
        "peak_tolerance_hz": tol_hz,
        "sav_total_expected": sav_total,
        "demon_total_expected": demon_total,
    }

    payload = {
        "dataset": dataset_name,
        "synthetic": True,
        "metrics": metrics,
        "sav": sav_records,
        "demon": demon_records,
    }
    _save_json(run_dir / "sav_detection.json", payload)
    _save_json(run_dir / "metrics.json", metrics)
    np.savez(run_dir / "sav_detection_traces.npz", **traces)
    return payload


def _build_model(cfg: dict, num_classes: int):
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()
    common = {
        "num_classes": num_classes,
        "fc_dim": model_cfg.get("fc_dim", 256),
        "image_size": cfg["dataset"].get("image_size", 224),
    }
    if name == "cnn":
        return CNNClassifier(**common)
    if name == "hqcnn":
        return HQCNN(
            n_qubits=model_cfg.get("n_qubits", 10),
            n_layers=model_cfg.get("n_layers", 4),
            pqc_init=model_cfg.get("pqc_init", "uniform"),
            **common,
        )
    if name == "hqcnn_merlin":
        from lib.models_merlin import HQCNNMerLin

        return HQCNNMerLin(
            n_modes=model_cfg.get("n_modes", 10),
            n_photons=model_cfg.get("n_photons", 2),
            no_bunching=model_cfg.get("no_bunching", True),
            device=cfg.get("device", "cpu"),
            **common,
        )
    raise ValueError(f"Unknown model: {name}")


def _run_classification(cfg: dict, run_dir: Path, data_root: Path) -> dict:
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    frame_spec = FrameSpec(
        duration_s=dataset_cfg.get("frame_duration_s", 60.0),
        overlap=dataset_cfg.get("frame_overlap", 0.25),
        target_sr=dataset_cfg.get("target_sr", 8000),
    )
    if dataset_cfg.get("download", True) and dataset_name == "deepship":
        dataset_dir = data_root / "deepship"
        if not any(dataset_dir.glob("*/*.wav")):
            logger.info("No DeepShip WAV files found; downloading the public sample.")
            ensure_deepship_github_sample(data_root)
    elif dataset_cfg.get("download", False):
        raise ValueError("dataset.download is only supported for DeepShip.")

    images, labels, class_names = build_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        spectrogram_method=dataset_cfg.get("spectrogram_method", "sav"),
        frame_spec=frame_spec,
        image_size=dataset_cfg.get("image_size", 224),
        samples_per_class_synthetic=dataset_cfg.get("samples_per_class_synthetic", 8),
        duration_synthetic_s=dataset_cfg.get("duration_synthetic_s", 30.0),
        seed=cfg.get("seed", 1337),
    )
    if len(labels) == 0:
        raise RuntimeError("Empty dataset — check data_root and class subfolders.")

    x_train, y_train, x_test, y_test = split_dataset(
        images, labels, test_fraction=dataset_cfg.get("test_fraction", 0.3), seed=cfg.get("seed", 1337)
    )
    train_loader, test_loader = make_dataloaders(
        x_train, y_train, x_test, y_test, batch_size=cfg["training"].get("batch_size", 16)
    )

    device = _resolve_device(cfg.get("device", "cpu"))
    model = _build_model(cfg, num_classes=len(class_names))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s has %d trainable parameters", cfg["model"]["name"], n_params)

    history = train_loop(
        model,
        train_loader,
        test_loader,
        epochs=cfg["training"].get("epochs", 2),
        lr=cfg["training"].get("lr", 1e-4),
        device=device,
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    final_test_loss, final_test_acc, preds, ytrue = evaluate(model, test_loader, device, loss_fn)
    final_train_loss, final_train_acc, _, _ = evaluate(model, train_loader, device, loss_fn)
    cm = confusion_matrix(preds, ytrue, num_classes=len(class_names)).numpy().tolist()

    payload = {
        "dataset": cfg["dataset"]["name"],
        "model": cfg["model"]["name"],
        "n_params": n_params,
        "class_names": class_names,
        "epochs": [asdict(h) for h in history],
        "final": {
            "train_loss": final_train_loss,
            "train_acc": final_train_acc,
            "test_loss": final_test_loss,
            "test_acc": final_test_acc,
            "generalization_gap": final_train_acc - final_test_acc,
        },
        "confusion_matrix": cm,
    }
    _save_json(run_dir / "classification.json", payload)
    if cfg["training"].get("save_checkpoint", False):
        torch.save(model.state_dict(), run_dir / "model.pt")
    return payload


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Project entry point dispatched by the shared runtime."""
    task = cfg.get("task", "classification").lower()
    # The shared runtime exposes the *global* data root via cfg["data_root"];
    # paper datasets live in the per-paper subfolder.
    data_root = Path(cfg["data_root"]) / "qml_passive_sonar"
    logger.info("Task=%s  data_root=%s", task, data_root)

    if task == "sav_detection":
        result = _run_sav_detection(cfg, run_dir, data_root)
    elif task == "classification":
        result = _run_classification(cfg, run_dir, data_root)
    elif task == "merlin_classification":
        # Reuses the classification flow; _build_model dispatches to HQCNNMerLin.
        cfg["model"]["name"] = "hqcnn_merlin"
        result = _run_classification(cfg, run_dir, data_root)
    else:
        raise ValueError(f"Unknown task: {task}")

    summary_path = run_dir / "summary.json"
    summary = {"task": task, "status": "ok"}
    if isinstance(result, dict) and "final" in result:
        summary.update(result["final"])
    _save_json(summary_path, summary)
    logger.info("Run summary written to %s", summary_path)
