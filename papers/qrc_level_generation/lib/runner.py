"""Runtime entry point.

The runner performs three steps in order:
  1. Evaluate originality, broken-transition and save-point metrics for the
     Markov / uncorrelated baselines (always) and for the Moth-published QRC
     reference sequences when ``evaluation.evaluate_reference_sequences`` is
     true.
  2. Train a QRC (qubit or photonic) and generate fresh sequences at the
     requested temperatures, then compute the same metrics on the new
     sequences. Skipped when ``evaluation.reference_only`` is true.
  3. Write structured artifacts (``metrics.json``, ``summary.json``, a
     ``generated.npz`` per temperature, training history, originality figures)
     to ``run_dir``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lib import baselines, data, metrics, qrc_pipeline
from lib.qrc_qubit import QubitQRC

_LOG = logging.getLogger(__name__)


def _parse_temperatures(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(piece) for piece in value.split(",") if piece.strip()]
    if isinstance(value, (int, float)):
        return [float(value)]
    raise ValueError(f"Cannot parse temperatures: {value!r}")


def _build_reservoir(cfg: dict, num_features: int, seed: int):
    qrc_cfg = cfg["qrc"]
    backend = qrc_cfg.get("backend", "qubit")
    if backend == "qubit":
        return QubitQRC(
            n_qubits=int(qrc_cfg["n_qubits"]),
            num_features=num_features,
            n_random_gates=int(qrc_cfg.get("n_random_gates", 30)),
            embedding_layers=int(qrc_cfg.get("embedding_layers", 1)),
            input_scale=float(qrc_cfg.get("input_scale", 1.0)),
            feedback_scale=float(qrc_cfg.get("feedback_scale", 1.0)),
            depolarizing_p=float(qrc_cfg.get("depolarizing_p", 0.0)),
            shots=int(qrc_cfg.get("shots", 0)),
            seed=seed,
        )
    if backend == "photonic":
        from lib.qrc_photonic import PhotonicQRC

        return PhotonicQRC(
            num_features=num_features,
            n_modes=int(qrc_cfg.get("n_modes", 6)),
            n_photons=int(qrc_cfg.get("n_photons", 3)),
            input_scale=float(qrc_cfg.get("input_scale", 1.0)),
            feedback_scale=float(qrc_cfg.get("feedback_scale", 1.0)),
            seed=seed,
        )
    raise ValueError(f"Unknown reservoir backend: {backend}")


def _compute_metrics_for_sequences(
    label: str,
    sequences: list[list[int]],
    original: list[int],
    max_length: int,
    rules,
) -> dict:
    if not sequences:
        return {"label": label, "n_samples": 0}
    orig = metrics.originality_rate(sequences, original, max_length=max_length)
    broken = metrics.broken_rate(sequences, rules)
    save_mean, save_std = metrics.separation_stats(sequences, target=11)
    return {
        "label": label,
        "n_samples": len(sequences),
        "originality": {str(k): float(v) for k, v in orig.items()},
        "broken_rate_per_rule": {k: float(v) for k, v in broken.items()},
        "save_point_separation_mean": save_mean,
        "save_point_separation_std": save_std,
    }


def _evaluate_reference_sequences(
    cfg: dict, original: list[int], max_length: int, rules
) -> dict:
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    if not eval_cfg.get("evaluate_reference_sequences", False):
        return {}
    reference_root = data_cfg.get("reference_root")
    if reference_root is None:
        _LOG.warning(
            "evaluation.evaluate_reference_sequences is True but data.reference_root is missing"
        )
        return {}
    try:
        ref_seqs = data.load_reference_sequences(
            reference_root, level="1-2", n_qubits=6, backend="Aer"
        )
    except FileNotFoundError as exc:
        _LOG.warning("Reference data not found: %s", exc)
        return {}
    block: dict[str, dict] = {}
    for beta, seqs in sorted(ref_seqs.items()):
        block[f"Aer_T={beta}"] = _compute_metrics_for_sequences(
            f"reference Aer T={beta}",
            seqs,
            original,
            max_length,
            rules,
        )
    return block


def _save_generated_sequences(
    run_dir: Path, generated: dict[float, list[list[int]]]
) -> None:
    npz_path = run_dir / "generated_sequences.npz"
    payload = {}
    for temperature, sequences in generated.items():
        key = f"T_{temperature}"
        payload[key] = np.asarray(sequences, dtype=np.int64)
    if payload:
        np.savez_compressed(npz_path, **payload)


def _plot_originality(
    run_dir: Path,
    original: list[int],
    baseline_seqs: dict[str, list[list[int]]],
    reference_metrics: dict[str, dict],
    qrc_metrics: dict[str, dict],
    max_length: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    if baseline_seqs.get("markov"):
        rates = metrics.originality_rate(
            baseline_seqs["markov"], original, max_length=max_length
        )
        ax.plot(list(rates.keys()), list(rates.values()), "k--", label="Markov")
    if baseline_seqs.get("uncorrelated"):
        rates = metrics.originality_rate(
            baseline_seqs["uncorrelated"], original, max_length=max_length
        )
        ax.plot(list(rates.keys()), list(rates.values()), "k-.", label="Uncorr.")

    cmap = plt.colormaps.get_cmap("coolwarm")
    qrc_items = sorted(qrc_metrics.items())
    for index, (label, payload) in enumerate(qrc_items):
        if "originality" not in payload:
            continue
        lengths = sorted(int(k) for k in payload["originality"].keys())
        values = [payload["originality"][str(L)] for L in lengths]
        ax.plot(
            lengths,
            values,
            "o-",
            label=label,
            color=cmap(index / max(1, len(qrc_items) - 1)),
        )

    for label, payload in sorted(reference_metrics.items()):
        if "originality" not in payload:
            continue
        lengths = sorted(int(k) for k in payload["originality"].keys())
        values = [payload["originality"][str(L)] for L in lengths]
        ax.plot(lengths, values, "x--", label=label, alpha=0.6)

    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Originality rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Originality of generated levels")
    fig.tight_layout()
    fig.savefig(run_dir / "originality.png", dpi=150)
    plt.close(fig)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

    data_cfg = cfg.get("data", {})
    level_file = data_cfg.get(
        "level_file", "../../data/qrc_level_generation/mario_level_1-2.json"
    )
    original, num_features = data.load_original_level(level_file)
    _LOG.info(
        "Loaded Mario level 1-2: length=%d, num_features=%d",
        len(original),
        num_features,
    )

    generation_cfg = cfg.get("generation", {})
    max_length = int(generation_cfg.get("max_sequence_length_for_originality", 20))
    rules = metrics.mario_rules(level="1-2")

    baselines_cfg = cfg.get("baselines", {})
    baseline_seqs = baselines.make_baseline_sequences(
        original=original,
        num_features=num_features,
        length=len(original),
        n_samples=int(baselines_cfg.get("n_samples", 200)),
        seed=seed,
    )

    metrics_payload: dict = {
        "config_excerpt": {
            "qrc": cfg.get("qrc", {}),
            "training": cfg.get("training", {}),
            "generation": cfg.get("generation", {}),
            "baselines": cfg.get("baselines", {}),
            "evaluation": cfg.get("evaluation", {}),
        },
        "original_length": len(original),
        "num_features": num_features,
        "baselines": {
            name: _compute_metrics_for_sequences(
                name, seqs, original, max_length, rules
            )
            for name, seqs in baseline_seqs.items()
        },
    }

    reference_metrics = _evaluate_reference_sequences(cfg, original, max_length, rules)
    metrics_payload["reference"] = reference_metrics

    qrc_metrics: dict[str, dict] = {}
    generated: dict[float, list[list[int]]] = {}
    eval_cfg = cfg.get("evaluation", {})
    if not bool(eval_cfg.get("reference_only", False)):
        reservoir = _build_reservoir(cfg, num_features=num_features, seed=seed)
        temps = _parse_temperatures(generation_cfg.get("temperatures_str", "1.0"))
        training_cfg = cfg.get("training", {})
        fnn_cfg = cfg.get("fnn", {})
        result = qrc_pipeline.train_and_generate(
            reservoir=reservoir,
            original=original,
            num_features=num_features,
            hidden_dim=int(fnn_cfg.get("hidden_dim", 0)),
            epochs=int(training_cfg.get("epochs", 50)),
            lr=float(training_cfg.get("lr", 0.01)),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
            leaking_rate=float(cfg["qrc"].get("leaking_rate", 0.3)),
            temperatures=temps,
            n_samples=int(generation_cfg.get("n_samples", 10)),
            gen_length=int(generation_cfg.get("length", len(original))),
            seed=seed,
            log_progress=True,
        )
        generated = result["generated"]
        metrics_payload["training_history"] = result["training_history"]
        for temperature, seqs in generated.items():
            label = f"QRC_T={temperature}"
            qrc_metrics[label] = _compute_metrics_for_sequences(
                label,
                seqs,
                original,
                max_length,
                rules,
            )

    metrics_payload["qrc"] = qrc_metrics

    _save_generated_sequences(run_dir, generated)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    summary = {
        "label": cfg.get("description", "qrc_level_generation"),
        "seed": seed,
        "qrc_backend": cfg.get("qrc", {}).get("backend", "qubit"),
        "temperatures": list(generated.keys()) if generated else [],
        "n_baseline_samples": int(baselines_cfg.get("n_samples", 200)),
        "n_reference_groups": len(reference_metrics),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _plot_originality(
        run_dir,
        original=original,
        baseline_seqs=baseline_seqs,
        reference_metrics=reference_metrics,
        qrc_metrics=qrc_metrics,
        max_length=max_length,
    )
    _LOG.info("Wrote metrics, summary, and originality plot to %s", run_dir)
