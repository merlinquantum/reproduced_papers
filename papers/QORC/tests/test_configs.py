from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib.config import load_config

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"


def test_defaults_has_description_and_required_keys() -> None:
    cfg = load_config(CONFIG_DIR / "defaults.json")
    assert cfg["description"], "defaults.json must document the experiment"
    for key in (
        "xp_type",
        "outdir",
        "n_photons",
        "n_modes",
        "n_epochs",
        "dataset_name",
        "noise_enabled",
        "use_qpu",
        "save_weights",
    ):
        assert key in cfg, f"Missing '{key}' in defaults.json"
    assert cfg["n_photons"] == 3
    assert cfg["n_modes"] == 20
    assert cfg["n_epochs"] == 200
    assert cfg["dataset_name"] == "mnist"
    assert cfg["noise_enabled"] is False
    assert cfg["use_qpu"] is False
    assert cfg["save_weights"] is False


def test_cli_schema_matches_defaults_path() -> None:
    defaults_path = PROJECT_DIR / "configs" / "defaults.json"
    cli_schema_path = PROJECT_DIR / "cli.json"
    if not cli_schema_path.exists():
        cli_schema_path = PROJECT_DIR / "configs" / "cli.json"

    assert defaults_path.exists(), "defaults.json missing"
    assert cli_schema_path.exists(), "cli.json missing"

    runner_module = importlib.import_module("lib.runner")
    assert hasattr(runner_module, "train_and_evaluate"), (
        "Runner must expose train_and_evaluate()"
    )


def test_qorc_mlr_comparison_config_has_requested_settings() -> None:
    comparison = load_config(CONFIG_DIR / "comparison_QORC_MLR_mnist.json")

    assert comparison["xp_type"] == "comparison_qorc_lsvc"
    assert comparison["dataset_name"] == "mnist"
    assert comparison["n_photons"] == 3
    assert comparison["n_modes"] == 20
    assert comparison["n_epochs"] == 200
    assert comparison["noise_enabled"] is False
    assert comparison["use_qpu"] is False


def test_noisy_indistinguishability_config_has_figure_2b_defaults() -> None:
    config = load_config(CONFIG_DIR / "noisy_QORC_indistinguishability.json")

    assert config["xp_type"] == "noisy_qorc_indistinguishability"
    assert config["n_photons"] == 3
    assert config["n_epochs"] == 200
    assert config["indistinguishability_m12"] == [0, 25, 50, 75, 100]
    assert config["indistinguishability_m20"] == [0, 20, 35, 50, 70, 85, 100]


def test_medmnist_fig3_config_has_requested_settings() -> None:
    config = load_config(CONFIG_DIR / "QORC_medmnist.json")

    assert config["xp_type"] == "fig3_qorc_mlr_medmnist"
    assert config["datasets"] == ["OCT", "OrganS", "OrganA", "Derma"]
    assert config["seeds"] == [42, 43, 44]
    assert config["n_photons"] == 3
    assert config["n_modes"] == 20
    assert config["n_epochs"] == 200


def test_fig4_dataset_size_config_has_requested_settings() -> None:
    config = load_config(CONFIG_DIR / "fig4_dataset_size_comparison.json")

    assert config["xp_type"] == "fig4_dataset_size_comparison"
    assert config["n_photons"] == 3
    assert config["n_modes"] == 12
    assert config["n_epochs"] == 100
    assert config["n_subsets"] > 0
    assert config["training_sizes"][0] == 100
    assert config["training_sizes"][-1] == 50000
    assert config["enable_qpu"] is False


def test_fig6_architecture_config_has_requested_settings() -> None:
    config = load_config(CONFIG_DIR / "fig6_MNIST_different_architectures.json")

    assert config["xp_type"] == "fig6_mnist_different_architectures"
    assert config["n_photons"] == 3
    assert config["n_modes"] == 20
    assert config["b_no_bunching"] is True
    assert config["epochs"] == 30
    assert config["n_runs"] > 0
    assert config["models"]["Linear"]["accelerated"]["batch_size"] == 32
    assert config["models"]["Deep"]["accelerated"]["learning_rate"] == 2.2e-4


def test_fig5_distribution_config_has_requested_settings() -> None:
    config = load_config(CONFIG_DIR / "fig5_distribution.json")

    assert config["xp_type"] == "fig5_distribution"
    assert config["n_photons"] == 3
    assert config["n_modes"] == 12
    assert config["distribution_start"] == 170
    assert config["distribution_end"] == 210
    assert config["n_simulation_runs"] == 500
    assert config["shots"] == 30000
    assert config["noise_g2"] == 0.0195
    assert config["noise_indistinguishability"] == 0.8636
    assert config["use_qpu"] is True


def test_fig4_mlr_only_config_requests_only_mlr() -> None:
    config = load_config(CONFIG_DIR / "fig4_MLR_only.json")

    assert config["xp_type"] == "fig4_dataset_size_comparison"
    assert config["conditions"] == ["MLR"]
    assert config["n_subsets"] == 50
