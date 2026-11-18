"""Phase 4 diagnostic: can each Frozen Lake Q-function *represent* Q*?

Known-answer test from the gap-diagnosis guide: fit the analytic Q* (gamma =
0.8) on the 44 non-terminal state-action pairs by supervised regression
(full batch, Adam lr 0.02, 1500 steps, seeds 0-2), then report the MAE and
how often the model's greedy action equals argmax Q*. No reinforcement
learning is involved, so a failure here is a representation limit of the
circuit, not an exploration or optimisation problem of DQN.

Rows: the qubit code ansatz (5 layers), the planned photonic model (basis
phases encoded once; 5 and 15 layers), one-change variants (unbunched
readout, trainable output weights, bits as continuous inputs with trainable
scale encoded once / re-uploaded), and basis phases re-uploaded before every
layer (5 and 10 layers).

Run from the paper folder::

    python -I utils/diagnose_frozenlake_qstar_fit.py --output results/verification/frozenlake_qstar_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from lib.models import build_q_function  # noqa: E402
from lib.qfunction_merlin import MerlinQFunction  # noqa: E402
from lib.reference import optimal_q_values, terminal_states  # noqa: E402

SEEDS = [0, 1, 2]
STEPS = 1500
LEARNING_RATE = 0.02


class BitsAsContinuousInputs(MerlinQFunction):
    """Feed the 4 state bits through the continuous encoding (arctan(bit * w_d))."""

    def encoding_phases(self, observations: torch.Tensor) -> torch.Tensor:
        indices = observations[..., 0].round().long()
        bits = ((indices[..., None] >> self.bit_shifts) & 1).to(self.real_dtype)
        return super().encoding_phases(bits)


def load_config(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())


def fit(model: torch.nn.Module, states: torch.Tensor, target: torch.Tensor) -> dict:
    """Supervised fit of ``target`` (alive states x actions); returns per-seed metrics."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    batch = states.expand(len(SEEDS), -1, -1)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = ((model(batch) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        q_values = model(batch)
    mae = (q_values - target).abs().mean(dim=(1, 2))
    agreement = (q_values.argmax(-1) == target.argmax(-1)).double().mean(-1)
    return {
        "mae44": [float(v) for v in mae],
        "argmax_agreement": [float(v) for v in agreement],
        "trainable_parameters": int(model.count_trainable_parameters()),
    }


def variants() -> list[tuple[str, callable]]:
    """(label, factory) pairs; each factory builds a model for ``SEEDS``."""

    def merlin(name: str, **model_changes):
        cfg = load_config(name)
        cfg["model"].update(model_changes)
        return lambda: MerlinQFunction(cfg, SEEDS, 4, torch.float64)

    def bits(reupload: bool):
        cfg = deepcopy(load_config("e1_frozenlake_l5_merlin"))
        cfg["model"].update(
            state_encoding="continuous_phase",
            encoding_activation="arctan",
            data_reuploading=reupload,
            train_input_weights=True,
        )
        return lambda: BitsAsContinuousInputs(cfg, SEEDS, 4, torch.float64)

    qubit_cfg = load_config("e1_frozenlake_l5_codefaithful")
    return [
        (
            "qubit code ansatz, 5 layers",
            lambda: build_q_function(qubit_cfg, SEEDS, 4, torch.float64),
        ),
        (
            "photonic planned (basis phases once), 5 layers",
            merlin("e1_frozenlake_l5_merlin"),
        ),
        ("photonic planned, 15 layers", merlin("e1_frozenlake_l15_merlin")),
        (
            "+ unbunched (post-selected) readout, 5 layers",
            merlin("e1_frozenlake_l5_merlin", readout="unbunched_click"),
        ),
        (
            "+ trainable output weights, 5 layers",
            merlin("e1_frozenlake_l5_merlin", train_output_weights=True),
        ),
        ("bits with trainable scale, encoded once, 5 layers", bits(False)),
        ("bits with trainable scale, re-uploaded, 5 layers", bits(True)),
        (
            "basis phases re-uploaded before every layer, 5 layers",
            merlin("e1_frozenlake_l5_merlin_reupload"),
        ),
        (
            "basis phases re-uploaded before every layer, 10 layers",
            merlin("e1_frozenlake_l10_merlin_reupload"),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alive = [state for state in range(16) if state not in terminal_states()]
    target = torch.tensor(optimal_q_values(0.8), dtype=torch.float64)[alive]
    states = torch.tensor(alive, dtype=torch.float64).view(1, -1, 1)
    results = {
        "protocol": {
            "target": "Q* (gamma 0.8), 44 non-terminal pairs",
            "optimizer": f"Adam lr {LEARNING_RATE}, {STEPS} full-batch steps, MSE",
            "seeds": SEEDS,
        },
        "rows": {},
    }
    for label, factory in variants():
        torch.manual_seed(0)
        results["rows"][label] = fit(factory(), states, target)
        row = results["rows"][label]
        print(
            f"{label:58s} MAE44 {[round(v, 3) for v in row['mae44']]} "
            f"argmax {[round(v, 2) for v in row['argmax_agreement']]} "
            f"params {row['trainable_parameters']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
