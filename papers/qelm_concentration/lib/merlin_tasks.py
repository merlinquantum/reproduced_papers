"""Phase 4 MerLin tasks: the photonic reservoir of `lib.qelm_merlin` on the Phase 3 tasks.

Each task reuses the Phase 3 runner (splits, sequences, seeds, shot sampling, sweeps, ridge
readouts, baselines) and swaps only the reservoir and its readout features:

- ``landsat_readouts_merlin``: `lib.landsat_task.run_landsat_readouts` with photonic pair
  click probabilities and the click readouts;
- ``narma_scaling_merlin``: `lib.narma_task.run_narma_scaling` with neighbour click
  features;
- ``concentration_merlin``: output variability over random inputs (E1b analogue).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .landsat_task import run_landsat_readouts
from .narma_task import ReservoirFeatureSource, run_narma_scaling
from .qelm import real_dtype_from_config
from .qelm_merlin import (
    PhotonicArchitecture,
    PhotonicReservoir,
    click_readout_features,
    neighbour_click_features,
    reservoir_from_config,
    sampled_neighbour_click_features,
)

logger = logging.getLogger(__name__)


def _hardware_fields(arch: PhotonicArchitecture) -> dict:
    """Hardware-aware fields of every MerLin run (photonic-translation policy)."""
    return {
        "computation_space": "FOCK",
        "detector_model": "threshold (ideal, every mode)",
        "n_photons": arch.n_photons,
        "n_modes": arch.n_modes,
        "n_layers": arch.n_layers,
        "input_state": arch.input_state,
        "encoding": "beam-splitter angle theta = angle_scale * input_strength * u",
        "measurement_strategy": "MeasurementStrategy.probs(FOCK) with threshold detectors",
        "postselection": "none",
        "simulator": "MerLin analytic simulation (CPU)",
    }


def _simulate_pair_clicks(cfg: dict, data, reservoir_seed: int):
    arch = PhotonicArchitecture.from_config(cfg["merlin_reservoir"])
    reservoir = reservoir_from_config(cfg, reservoir_seed, data.train_x.shape[1])
    chunk = int(cfg["merlin_reservoir"]["chunk_size"])
    pairs = arch.disjoint_pairs()
    return (
        reservoir.pair_click_probabilities(data.train_x, pairs, chunk),
        reservoir.pair_click_probabilities(data.test_x, pairs, chunk),
    )


def run_landsat_readouts_merlin(cfg: dict, run_dir: Path) -> dict:
    """Landsat readout study (E5, E3, E4, E7, E8 analogues) with the photonic reservoir."""
    arch = PhotonicArchitecture.from_config(cfg["merlin_reservoir"])
    results = run_landsat_readouts(
        cfg,
        run_dir,
        simulate=_simulate_pair_clicks,
        features=click_readout_features,
    )
    return {"hardware": _hardware_fields(arch), **results}


def photonic_feature_source(cfg: dict) -> ReservoirFeatureSource:
    """Neighbour click features of the photonic reservoir for NARMA windows."""
    arch = PhotonicArchitecture.from_config(cfg["merlin_reservoir"])
    chunk = int(cfg["merlin_reservoir"]["chunk_size"])

    def compute(reservoir_seed: int, windows: np.ndarray):
        reservoir = reservoir_from_config(cfg, reservoir_seed, windows.shape[1])
        pair_probs = reservoir.pair_click_probabilities(
            windows, arch.neighbour_pairs(), chunk
        )
        features = neighbour_click_features(pair_probs)

        def noisy(condition: dict, rng: np.random.Generator) -> np.ndarray:
            kind = condition["kind"]
            if kind == "none":
                return features
            if kind == "gaussian":
                return features + rng.normal(
                    0.0, float(condition["std"]), size=features.shape
                )
            if kind == "shots":
                return sampled_neighbour_click_features(
                    pair_probs, int(condition["shots"]), rng
                )
            raise ValueError(f"Unknown photonic noise kind {kind!r}")

        return features, noisy

    return ReservoirFeatureSource(
        window=arch.window,
        metadata={
            "n_modes": arch.n_modes,
            "n_photons": arch.n_photons,
            "n_layers": arch.n_layers,
        },
        compute=compute,
    )


def run_narma_scaling_merlin(cfg: dict, run_dir: Path) -> dict:
    """NARMA-n versus order (E2 analogue) with the photonic reservoir."""
    arch = PhotonicArchitecture.from_config(cfg["merlin_reservoir"])
    results = run_narma_scaling(cfg, run_dir, source=photonic_feature_source(cfg))
    return {"hardware": _hardware_fields(arch), **results}


def run_concentration_merlin(cfg: dict, run_dir: Path) -> dict:
    """Variability of click statistics over random inputs (E1b analogue).

    For every setting, size ``(n_modes, n_photons)``, depth and reservoir seed: the
    variance over ``u ~ Unif(-1, 1)^(n_modes / 2)`` of each per-mode click probability (w1)
    and of each neighbour coincidence probability (w2), averaged over features.
    """
    exp = cfg["concentration_merlin"]
    dtype = real_dtype_from_config(cfg)
    chunk = int(cfg["merlin_reservoir"]["chunk_size"])
    records = []
    logger.info(
        "DATASET_READY | dataset=uniform_inputs | n_samples=%s | distribution=Unif(-1,1)^(n_modes/2)",
        exp["n_samples"],
    )
    for setting_name, setting in exp["parameter_settings"].items():
        for n_modes, n_photons in exp["sizes"]:
            for n_layers in exp["n_layers"]:
                arch = PhotonicArchitecture.from_config(
                    cfg["merlin_reservoir"],
                    n_modes=int(n_modes),
                    n_photons=int(n_photons),
                    n_layers=int(n_layers),
                )
                for seed in exp["reservoir_seeds"]:
                    inputs = np.random.default_rng(
                        [int(seed), int(n_modes), 5]
                    ).uniform(-1.0, 1.0, size=(int(exp["n_samples"]), arch.window))
                    reservoir = PhotonicReservoir(
                        arch, setting, int(seed), arch.window, dtype
                    )
                    features = neighbour_click_features(
                        reservoir.pair_click_probabilities(
                            inputs, arch.neighbour_pairs(), chunk
                        )
                    )
                    variance = features.var(axis=0)
                    w1, w2 = variance[:n_modes], variance[n_modes:]
                    record = {
                        "setting": setting_name,
                        "n_modes": int(n_modes),
                        "n_photons": int(n_photons),
                        "n_layers": int(n_layers),
                        "reservoir_seed": int(seed),
                        "w1_variability_mean": float(w1.mean()),
                        "w1_variability_median": float(np.median(w1)),
                        "w2_variability_mean": float(w2.mean()),
                        "w2_variability_median": float(np.median(w2)),
                    }
                    records.append(record)
                    logger.info(
                        "EVALUATION_COMPLETED | setting=%s | modes=%d | photons=%d | layers=%d | seed=%s | w1_var=%.3e | w2_var=%.3e",
                        setting_name,
                        n_modes,
                        n_photons,
                        n_layers,
                        seed,
                        record["w1_variability_mean"],
                        record["w2_variability_mean"],
                    )
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = (
            f"{record['setting']}|modes={record['n_modes']}|photons={record['n_photons']}"
            f"|n_layers={record['n_layers']}"
        )
        groups.setdefault(key, []).append(record)
    summary = {}
    for key, items in groups.items():
        summary[key] = {"n": len(items)}
        for metric in ("w1_variability_mean", "w2_variability_mean"):
            values = np.array([item[metric] for item in items])
            summary[key][f"{metric}_avg"] = float(values.mean())
            summary[key][f"{metric}_std"] = float(values.std())
    hardware = _hardware_fields(
        PhotonicArchitecture.from_config(cfg["merlin_reservoir"])
    )
    hardware.update(n_modes="per record", n_photons="per record", n_layers="per record")
    hardware["input_state"] = "one photon every n_modes / n_photons modes from mode 0"
    return {"hardware": hardware, "records": records, "summary": summary}
