"""Write the Phase 5 classical-baseline configs from the frozen matching table.

Widths come from the parameter matching in LOG.md ("Phase 5 Comparison Plan"); the
script checks every count against its target before writing. Outputs:

* ``configs/classical/tuning_base.json`` and ``configs/classical/tuning_candidates.json``
  (learning-rate selection on a validation split, E1 layout);
* after selection, with ``--learning-rates results/classical_selection.json``: one
  named config per size and layout (``configs/classical/<size>_<layout>.json``), the
  centralised references, and the candidate lists used by the final sweeps.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_DIR / "configs"
OUT = CONFIGS / "classical"

# (client hidden, wide-server hidden, narrow-server hidden, central hidden, epochs)
SIZES = {
    "gate": (195, 107, 421, 270, 30),
    "photonic_h": (7, 12, 36, 12, 100),
    "photonic_f": (7, 7, 14, 10, 100),
}
# Matched targets (quantum or photonic parameter counts) and the tolerance.
TARGETS = {
    "gate": {"client": 8000, "wide": 8000, "narrow": 8000, "central": 72000},
    "photonic_h": {"client": 280, "wide": 914, "narrow": 690, "central": 3154},
    "photonic_f": {"client": 280, "wide": 504, "narrow": 280, "central": 2744},
}
TOLERANCE = 0.075
LAYOUTS = {
    "e1_8clients": ([0, 1, 2, 3, 4, 5, 6, 7], "server_wide"),
    "e2_1client": ([0], "server_narrow"),
    "e3_8x1": ([0] * 8, "server_wide"),
    "e4_2x4": ([0, 1, 2, 3, 0, 1, 2, 3], "server_wide"),
}
LEARNING_RATES = [0.001, 0.002, 0.005, 0.01]


def params(inputs: int, hidden: int, outputs: int) -> int:
    return inputs * hidden + hidden + hidden * outputs + outputs


def check_matching() -> None:
    for size, (hc, hw, hn, hz, _) in SIZES.items():
        actual = {
            "client": params(32, hc, 8),
            "wide": params(64, hw, 10),
            "narrow": params(8, hn, 10),
            "central": params(256, hz, 10),
        }
        for part, value in actual.items():
            target = TARGETS[size][part]
            if abs(value - target) > TOLERANCE * target:
                raise ValueError(f"{size} {part}: {value} params vs target {target}")


def base_config() -> dict:
    cfg = json.loads((CONFIGS / "e1_8clients_original.json").read_text())
    cfg["model"] = {
        "name": "vqfl_classical",
        "backend": "classical",
        "architecture": "vertical",
        "size": "gate",
        "client_slices": LAYOUTS["e1_8clients"][0],
        "server_encoding": "server_wide",
        "client_hidden": SIZES["gate"][0],
        "server_hidden": SIZES["gate"][1],
        "central_hidden": SIZES["gate"][3],
    }
    cfg["training"]["epochs"] = SIZES["gate"][4]
    return cfg


def model_overrides(size: str, architecture: str, layout: str) -> dict:
    hc, hw, hn, hz, epochs = SIZES[size]
    slices, server = LAYOUTS[layout]
    return {
        "model.size": size,
        "model.architecture": architecture,
        "model.client_slices": slices,
        "model.server_encoding": server,
        "model.client_hidden": hc,
        "model.server_hidden": hn if server == "server_narrow" else hw,
        "model.central_hidden": hz,
        "training.epochs": epochs,
    }


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learning-rates", type=Path, help="Selection JSON from the tuning sweep"
    )
    args = parser.parse_args()
    check_matching()
    OUT.mkdir(exist_ok=True)

    tuning = base_config()
    tuning["description"] = (
        "Phase 5 TUNING base (classical baselines): learning-rate selection on a 20% "
        "validation split, E1 layout. NOT a result config."
    )
    tuning["experiment"] = "classical_tuning"
    tuning["label"] = "classical baseline: lr selection on validation"
    tuning["dataset"]["validation_fraction"] = 0.2
    write(OUT / "tuning_base.json", tuning)
    candidates = [
        {**model_overrides(size, architecture, "e1_8clients"), "training.lr": lr}
        for architecture in ("vertical", "central")
        for size in SIZES
        for lr in LEARNING_RATES
    ]
    write(OUT / "tuning_candidates.json", candidates)
    print(f"wrote tuning base and {len(candidates)} tuning candidates")

    if not args.learning_rates:
        return
    selection = json.loads(args.learning_rates.read_text())["selections_by_group"]
    chosen = {}
    for group, entry in selection.items():
        key = json.loads(group)
        chosen[(key["architecture"], key["size"])] = entry["selected"]["lr"]
    final = base_config()
    final["experiment"] = "classical"
    final["label"] = "classical fair baseline (Phase 5), validation-selected lr"
    for size in SIZES:
        sweep_candidates = []
        for layout in LAYOUTS:
            overrides = {
                **model_overrides(size, "vertical", layout),
                "training.lr": chosen[("vertical", size)],
            }
            sweep_candidates.append(overrides)
            cfg = copy.deepcopy(final)
            cfg["experiment"] = f"{layout}_classical_{size}"
            cfg["description"] = (
                f"Phase 5 classical vertical-FL baseline, {layout} layout, parameters "
                f"matched to the {size} model; lr selected on validation."
            )
            for dotted, value in overrides.items():
                section, leaf = dotted.split(".")
                cfg[section][leaf] = value
            write(OUT / f"{size}_{layout}.json", cfg)
        write(OUT / f"final_candidates_vertical_{size}.json", sweep_candidates)
    central_candidates = []
    for size in SIZES:
        overrides = {
            **model_overrides(size, "central", "e1_8clients"),
            "training.lr": chosen[("central", size)],
        }
        central_candidates.append(overrides)
        cfg = copy.deepcopy(final)
        cfg["experiment"] = f"central_classical_{size}"
        cfg["description"] = (
            f"Phase 5 centralised MLP reference on all 256 pixels, parameters matched to "
            f"the {size} model's E1 total; lr selected on validation."
        )
        for dotted, value in overrides.items():
            section, leaf = dotted.split(".")
            cfg[section][leaf] = value
        write(OUT / f"{size}_central.json", cfg)
    write(OUT / "final_candidates_central.json", central_candidates)
    write(OUT / "final_base.json", final)
    print("wrote final configs and candidate lists:", chosen)


if __name__ == "__main__":
    main()
