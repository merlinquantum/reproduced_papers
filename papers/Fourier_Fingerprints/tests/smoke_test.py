from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.fourier import (  # noqa: E402
    AVAILABLE_CIRCUITS,
    N_PHOTONS,
    SCALE_FACTORS,
    PhotonicSpectralModel,
    compute_fingerprint,
)
from lib.learning import (  # noqa: E402
    SpectralRegressor,
    random_fourier_target,
    train_regressor,
)


def test_configs_have_required_keys() -> None:
    """Every config carries the schema the runner expects."""
    config_paths = sorted((PROJECT_ROOT / "configs").glob("*.json"))
    assert config_paths, "No configuration files found."

    for config_path in config_paths:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["description"]
        assert cfg["outdir"]
        assert cfg["graph_name"]
        assert cfg["circuits"]
        assert cfg["dimension"] in SCALE_FACTORS
        assert cfg["encoding"] in SCALE_FACTORS[cfg["dimension"]]
        for circuit_name in cfg["circuits"]:
            assert circuit_name in AVAILABLE_CIRCUITS


def test_defaults_config_is_discoverable() -> None:
    """The root runner requires configs/defaults.json, cli.json and lib/runner.py."""
    for marker in (
        Path("configs") / "defaults.json",
        Path("cli.json"),
        Path("lib") / "runner.py",
    ):
        assert (PROJECT_ROOT / marker).is_file(), f"Missing project marker: {marker}"


@pytest.mark.parametrize("dimension", [1, 2])
def test_mode0_mask_matches_fock_basis(dimension: int) -> None:
    """The measured observable covers every state with a photon in mode 0.

    This guards the bug where a fixed ``[:, :5]`` slice silently measured
    P(n0 >= 2) for the 2D model, whose Fock space has 35 states rather than 15.
    """
    model = PhotonicSpectralModel(dimension=dimension, encoding="linear")
    keys = model.quantum_layer.output_keys
    expected = [index for index, state in enumerate(keys) if state[0] >= 1]

    assert model.mode0_mask.shape[0] == len(keys)
    assert model.mode0_mask.nonzero().flatten().tolist() == expected
    assert model.n_photons == N_PHOTONS[dimension]


@pytest.mark.parametrize("dimension", [1, 2])
def test_mode0_occupancy_is_a_probability(dimension: int) -> None:
    """The measured signal is a probability, so it must lie in [0, 1]."""
    torch.manual_seed(0)
    model = PhotonicSpectralModel(dimension=dimension, encoding="linear")
    batch = torch.rand(8, dimension) * 2 * np.pi

    probabilities = model(batch)
    signal = model.mode0_occupancy(probabilities)

    assert signal.shape == (8,)
    assert torch.all(signal >= -1e-6)
    assert torch.all(signal <= 1 + 1e-6)


@pytest.mark.parametrize("dimension", [1, 2])
def test_compute_fingerprint_end_to_end(dimension: int) -> None:
    """A short run produces a square, symmetric fingerprint and a valid FCC."""
    torch.manual_seed(0)
    model = PhotonicSpectralModel(dimension=dimension, encoding="exponential")

    fingerprint, fcc_score, labels, coefficients = compute_fingerprint(
        model, n_samples=4
    )

    assert fingerprint.ndim == 2
    assert fingerprint.shape[0] == fingerprint.shape[1] == len(labels)
    assert coefficients.shape[0] == 4
    assert 0.0 <= fcc_score <= 1.0
    if fingerprint.size:
        assert np.allclose(fingerprint, fingerprint.T, atol=1e-8)
        assert np.all(np.abs(fingerprint) <= 1.0 + 1e-8)


def test_seeding_makes_a_run_reproducible() -> None:
    """Two runs under the same seed agree; a different seed disagrees."""

    def run(seed: int) -> float:
        torch.manual_seed(seed)
        model = PhotonicSpectralModel(dimension=1, encoding="linear")
        return compute_fingerprint(model, n_samples=3)[1]

    assert run(0) == pytest.approx(run(0))
    assert run(0) != pytest.approx(run(123))


def test_runner_dispatches_config_to_the_implementation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner forwards config fields to the shared entry point."""
    from lib import runner

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "run_fingerprints", lambda **kw: calls.append(kw))

    cfg = {
        "circuits": ["circuit_0"],
        "encoding": "linear",
        "graph_name": "smoke",
        "dimension": 2,
    }
    runner.train_and_evaluate(cfg, tmp_path / "run")

    assert len(calls) == 1
    assert calls[0]["dimension"] == 2
    assert calls[0]["encoding"] == "linear"
    assert calls[0]["name"] == "smoke"
    assert calls[0]["rundir"] == tmp_path / "run"


def test_invalid_inputs_are_rejected() -> None:
    """Bad dimension, encoding and circuit index raise rather than pass silently."""
    with pytest.raises(ValueError):
        PhotonicSpectralModel(dimension=3)
    with pytest.raises(ValueError):
        PhotonicSpectralModel(dimension=1, encoding="not_an_encoding")
    with pytest.raises(ValueError):
        PhotonicSpectralModel(dimension=1, circuit_index=99)


def test_random_fourier_target_is_real_and_standardised() -> None:
    """Targets are real, standardised, and carry only the requested frequencies."""
    target = random_fourier_target(max_frequency=6, n_points=64, seed=0)

    assert target.y.shape == (64,)
    assert torch.isfinite(target.y).all()
    assert float(target.y.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(target.y.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)

    spectrum = np.abs(np.fft.rfft(target.y.numpy()))
    assert spectrum[:7].sum() > 0
    # Nothing above the requested maximum frequency.
    assert spectrum[7:].max() < 1e-6 * max(spectrum.max(), 1.0)


def test_random_fourier_target_rejects_undersampling() -> None:
    """A grid below the Nyquist rate for the requested spectrum is refused."""
    with pytest.raises(ValueError, match="Nyquist"):
        random_fourier_target(max_frequency=40, n_points=64, seed=0)


def test_random_fourier_target_is_seed_reproducible() -> None:
    a = random_fourier_target(max_frequency=4, n_points=32, seed=3)
    b = random_fourier_target(max_frequency=4, n_points=32, seed=3)
    c = random_fourier_target(max_frequency=4, n_points=32, seed=4)
    assert torch.allclose(a.y, b.y)
    assert not torch.allclose(a.y, c.y)


def test_regressor_head_is_affine_and_trains() -> None:
    """The head adds exactly two parameters and training reduces the loss."""
    torch.manual_seed(0)
    model = SpectralRegressor(encoding="linear", circuit_index=3)
    core_params = sum(p.numel() for p in model.core.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == core_params + 2

    target = random_fourier_target(max_frequency=3, n_points=32, seed=0)
    best, history = train_regressor(model, target, epochs=25, lr=0.1)
    assert len(history) == 25
    assert best <= history[0]
    assert np.isfinite(history).all()
