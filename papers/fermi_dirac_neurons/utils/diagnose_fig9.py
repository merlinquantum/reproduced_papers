"""One-factor-at-a-time diagnosis of the Fig. 9 reproduction gap (exploratory).

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/diagnose_fig9.py \
        --panel tfim_2q --seeds 0-9 --epochs 3000 \
        --out papers/fermi_dirac_neurons/results/fig9_diagnosis_tfim_2q.json

Each variant overlays one change on ``configs/fig9_<panel>_original.json`` and
trains every model of that config with the same full-batch gradient descent as
``lib.training.train_models`` (without per-epoch logging). Recorded per
(variant, seed, model): the training loss at the checkpoints in
``CHECKPOINTS`` and the decay rate of the loss in decades per 1000 epochs
between the last two checkpoints. Outputs are exploratory evidence for the
gap diagnosis, not reproduction results.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.lib.data import (  # noqa: E402
    build_split,
    build_target,
    random_stream,
)
from fermi_dirac_neurons.lib.models import build_model  # noqa: E402
from runtime_lib.config import deep_update  # noqa: E402

STREAM_INITIALIZATION = 3
PAPER_STATE_SET = {"set": "paper_squared_loss", "p_values": [0.1, 0.25, 0.5, 0.75, 0.9]}


def _without_identity(cfg: dict) -> dict:
    cfg["data"]["target"]["include_identity"] = False
    for model in cfg["models"]:
        model["include_identity"] = False
    return cfg


def variants(base: dict) -> dict[str, dict]:
    """Return the variant configs, each differing from ``base`` in one factor."""
    overlays = {
        "baseline": {},
        "lr_0.03": {"training": {"learning_rate": 0.03}},
        "lr_0.3": {"training": {"learning_rate": 0.3}},
        "lr_1.0": {"training": {"learning_rate": 1.0}},
        "paper_states": {"data": {"train_states": PAPER_STATE_SET}},
        "zeta_U[-1,1]": {"data": {"target": {"low": -1.0, "high": 1.0}}},
        "T_1": {"physics": {"temperature": 1.0}},
        "T_4": {"physics": {"temperature": 4.0}},
    }
    result = {
        name: deep_update(copy.deepcopy(base), overlay)
        for name, overlay in overlays.items()
    }
    if any(model["include_identity"] for model in base["models"]):
        result["no_identity"] = _without_identity(copy.deepcopy(base))
    return result


def train(cfg: dict, epochs: int, checkpoints: list[int]) -> dict:
    """Train every model of ``cfg`` and return losses at the checkpoints."""
    temperature = float(cfg["physics"]["temperature"])
    n_qubits = int(cfg["data"]["n_qubits"])
    target = build_target(cfg)
    train_data = build_split(cfg, "train", target)
    learning_rate = float(cfg["training"]["learning_rate"])
    results = {}
    for index, model_cfg in enumerate(cfg["models"]):
        model = build_model(model_cfg, n_qubits, temperature)
        rng = random_stream(cfg["seed"], STREAM_INITIALIZATION + index)
        theta = rng.uniform(
            cfg["training"]["init_low"],
            cfg["training"]["init_high"],
            size=model.n_parameters,
        )
        losses = {}
        for epoch in range(epochs + 1):
            loss, gradient = model.loss_and_gradient(theta, train_data, "squared")
            if not math.isfinite(loss):
                losses[epoch] = float("nan")
                break
            if epoch in checkpoints:
                losses[epoch] = loss
            theta = theta - learning_rate * gradient
        last, previous = checkpoints[-1], checkpoints[-2]
        rate = None
        if losses.get(last, 0) > 0 and losses.get(previous, 0) > 0:
            rate = (
                (math.log10(losses[previous]) - math.log10(losses[last]))
                * 1000
                / (last - previous)
            )
        results[model_cfg["name"]] = {
            "losses": {str(key): value for key, value in losses.items()},
            "decades_per_1000": rate,
        }
    return results


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(value) for value in text.split(",")]


def main() -> None:
    """Run every variant for every seed and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--panel", required=True, help="tfim_2q, tfim_3q, heis_2q or heis_3q"
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(
        (PROJECT_DIR / "configs" / f"fig9_{args.panel}_original.json").read_text()
    )
    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    base = deep_update(copy.deepcopy(defaults), base)
    checkpoints = sorted({0, 100, 1000, args.epochs // 2, args.epochs})
    report = {
        "panel": args.panel,
        "epochs": args.epochs,
        "checkpoints": checkpoints,
        "runs": [],
    }
    for name, cfg in variants(base).items():
        for seed in _parse_seeds(args.seeds):
            cfg["seed"] = seed
            report["runs"].append(
                {
                    "variant": name,
                    "seed": seed,
                    "models": train(cfg, args.epochs, checkpoints),
                }
            )
        rates = {
            model: np.median(
                [
                    run["models"][model]["decades_per_1000"] or 0.0
                    for run in report["runs"]
                    if run["variant"] == name
                ]
            )
            for model in report["runs"][-1]["models"]
        }
        finals = {
            model: np.median(
                [
                    run["models"][model]["losses"].get(str(args.epochs), np.nan)
                    for run in report["runs"]
                    if run["variant"] == name
                ]
            )
            for model in report["runs"][-1]["models"]
        }
        print(
            name.ljust(14),
            " ".join(f"{m}: L={finals[m]:.1e} rate={rates[m]:.2f}" for m in finals),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
