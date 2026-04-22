from __future__ import annotations

import pytest

from lib.config import validate_run_config


def test_validate_run_config_accepts_generic_full_defaults() -> None:
    cfg = validate_run_config({"model_type": "B"})
    assert cfg["model_type"] == "B"
    assert cfg["circuit_family"] == "generic"
    assert cfg["profile"] == "full"


def test_validate_run_config_accepts_paper_baselines() -> None:
    vit_cfg = validate_run_config({"model_type": "VisionTransformer"})
    ortho_cfg = validate_run_config(
        {"model_type": "OrthoFNN", "circuit_family": "butterfly", "image_embed_grayscale": True}
    )
    assert vit_cfg["model_type"] == "VisionTransformer"
    assert ortho_cfg["model_type"] == "OrthoFNN"
    assert ortho_cfg["image_embed_grayscale"] is True


def test_validate_run_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_run_config({"model_type": "Z"})


def test_validate_run_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        validate_run_config({"model_type": "A", "profile": "debug"})


def test_validate_run_config_rejects_invalid_full_sector_model() -> None:
    with pytest.raises(ValueError, match="compound_readout='full_sector'"):
        validate_run_config({"model_type": "E", "compound_readout": "full_sector"})


def test_validate_run_config_rejects_non_power_of_two_butterfly_embed_dim() -> None:
    with pytest.raises(ValueError, match="embed_dim to be a power of two"):
        validate_run_config(
            {"model_type": "A", "circuit_family": "butterfly", "embed_dim": 12}
        )


def test_validate_run_config_rejects_invalid_butterfly_total_modes_for_model_d() -> None:
    with pytest.raises(ValueError, match="total_modes"):
        validate_run_config(
            {
                "model_type": "D",
                "circuit_family": "butterfly",
                "img_size": 28,
                "patch_size": 7,
                "embed_dim": 16,
                "use_cls_token": True,
            }
        )


def test_validate_run_config_accepts_valid_butterfly_model_d_without_cls() -> None:
    cfg = validate_run_config(
        {
            "model_type": "D",
            "circuit_family": "butterfly",
            "img_size": 28,
            "patch_size": 7,
            "embed_dim": 16,
            "use_cls_token": False,
        }
    )
    assert cfg["use_cls_token"] is False


def test_validate_run_config_accepts_retina_sized_subset_settings() -> None:
    cfg = validate_run_config(
        {
            "model_type": "B",
            "train_subset_size": 1080,
            "train_subset_seed": 5,
            "train_subset_mode": "stratified",
        }
    )
    assert cfg["train_subset_size"] == 1080
    assert cfg["train_subset_seed"] == 5
    assert cfg["train_subset_mode"] == "stratified"
    assert cfg["data_regime"] == "retina_sized_train"


def test_validate_run_config_defaults_non_retina_subset_regime_from_size() -> None:
    cfg = validate_run_config(
        {
            "model_type": "B",
            "train_subset_size": 5000,
            "train_subset_seed": 5,
            "train_subset_mode": "stratified",
        }
    )
    assert cfg["data_regime"] == "train_subset_5000"


def test_validate_run_config_rejects_unknown_train_subset_mode() -> None:
    with pytest.raises(ValueError, match="Unknown train_subset_mode"):
        validate_run_config({"model_type": "A", "train_subset_size": 1080, "train_subset_mode": "weird"})


def test_validate_run_config_rejects_image_embed_grayscale_for_non_orthofnn() -> None:
    with pytest.raises(ValueError, match="only valid for model_type='OrthoFNN'"):
        validate_run_config({"model_type": "A", "image_embed_grayscale": True})
