"""Phase 4 extension: screen photonic Frozen Lake designs that avoid re-uploading.

Same known-answer test as ``utils/diagnose_frozenlake_qstar_fit.py``
(supervised fit of Q*, gamma 0.8, on the 44 non-terminal pairs; Adam lr 0.02,
1500 full-batch steps, seeds 0-2), applied to:

* phase encoding once (the paper's single-encoding design) with more modes and
  photons: 6 modes / 3 photons (planned), 8 / 4, 10 / 5;
* **dual-rail input-state encoding**: 8 modes, 4 photons; bit i of the state
  index decides whether photon i sits in mode 2i (bit 0) or 2i+1 (bit 1),
  the photonic counterpart of the qubit basis state |s>. MerLin 0.4 cannot mix
  a per-sample input ``StateVector`` with per-sample phase inputs, so the state
  is prepared exactly from the fixed input [1,0,1,0,1,0,1,0] by one MZI per pair
  (``BS, PS(x_i), BS``) with internal phase ``x_i = pi * (1 - bit_i)``: phase 0
  swaps the photon, phase pi leaves it (checked with Perceval SLOS). Phases on
  a single Fock state are global, so this equals a per-sample Fock input
  (``state_encoding="dual_rail_input"`` in ``lib/qfunction_merlin.py``).
  Trainable brickwork layers follow; the encoding happens once;
* reference: basis phases re-uploaded before every layer (6 modes / 3 photons).

Run from the paper folder::

    python -I utils/screen_photonic_frozenlake_designs.py --output results/verification/frozenlake_design_screen.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from lib.qfunction_merlin import MerlinQFunction  # noqa: E402
from lib.reference import optimal_q_values, terminal_states  # noqa: E402

SEEDS = [0, 1, 2]
STEPS = 1500
LEARNING_RATE = 0.02


def frozen_config(n_modes: int, n_photons: int, n_layers: int, reupload: bool) -> dict:
    cfg = deepcopy(
        json.loads(
            (PROJECT_DIR / "configs" / "e1_frozenlake_l5_merlin.json").read_text()
        )
    )
    centre = n_modes // 2
    cfg["model"].update(
        n_modes=n_modes,
        n_photons=n_photons,
        input_state=[1 if mode % 2 == 0 else 0 for mode in range(n_modes)][:n_modes],
        n_layers=n_layers,
        data_reuploading=reupload,
        readout_modes=list(range(centre - 2, centre + 2)),
    )
    if sum(cfg["model"]["input_state"]) != n_photons:
        raise ValueError("Input-state helper expects n_photons = n_modes / 2")
    return cfg


def designs() -> list[tuple[str, callable]]:
    def phase(n_modes, n_photons, n_layers, reupload=False):
        cfg = frozen_config(n_modes, n_photons, n_layers, reupload)
        return lambda: MerlinQFunction(cfg, SEEDS, 4, torch.float64)

    def dual(n_layers):
        cfg = frozen_config(8, 4, n_layers, False)
        cfg["model"]["state_encoding"] = "dual_rail_input"
        return lambda: MerlinQFunction(cfg, SEEDS, 4, torch.float64)

    return [
        ("phase once, 6 modes / 3 photons, 5 layers (planned)", phase(6, 3, 5)),
        ("phase once, 8 modes / 4 photons, 5 layers", phase(8, 4, 5)),
        ("phase once, 8 modes / 4 photons, 10 layers", phase(8, 4, 10)),
        ("phase once, 10 modes / 5 photons, 5 layers", phase(10, 5, 5)),
        ("dual-rail input, 8 modes / 4 photons, 5 layers", dual(5)),
        ("dual-rail input, 8 modes / 4 photons, 10 layers", dual(10)),
        (
            "phase re-uploaded, 6 modes / 3 photons, 5 layers (reference)",
            phase(6, 3, 5, True),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alive = [state for state in range(16) if state not in terminal_states()]
    target = torch.tensor(optimal_q_values(0.8), dtype=torch.float64)[alive]
    states = (
        torch.tensor(alive, dtype=torch.float64)
        .view(1, -1, 1)
        .expand(len(SEEDS), -1, -1)
    )
    results = {
        "protocol": {"steps": STEPS, "lr": LEARNING_RATE, "seeds": SEEDS},
        "rows": {},
    }
    for label, factory in designs():
        model = factory()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
        )
        start = time.perf_counter()
        for _ in range(STEPS):
            optimizer.zero_grad()
            loss = ((model(states) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        seconds = time.perf_counter() - start
        with torch.no_grad():
            q_values = model(states)
        row = {
            "mae44": [float(v) for v in (q_values - target).abs().mean(dim=(1, 2))],
            "argmax_agreement": [
                float(v)
                for v in (q_values.argmax(-1) == target.argmax(-1)).double().mean(-1)
            ],
            "trainable_parameters": int(model.count_trainable_parameters()),
            "fock_outcomes": int(model.layer.output_size),
            "ms_per_step_33_samples": 1e3 * seconds / STEPS,
        }
        results["rows"][label] = row
        print(
            f"{label:62s} MAE44 {[round(v, 3) for v in row['mae44']]} "
            f"argmax {[round(v, 2) for v in row['argmax_agreement']]} "
            f"params {row['trainable_parameters']} outcomes {row['fock_outcomes']} "
            f"{row['ms_per_step_33_samples']:.1f} ms/step",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
