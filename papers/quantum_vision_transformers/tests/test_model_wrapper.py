from __future__ import annotations

import lib.models as models
import pytest
import torch
import torch.nn as nn


class DummyTensorLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DummyTupleLayer(nn.Module):
    def __init__(self, d: int, sector_key: str):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.sector_key = sector_key

    def forward(self, x: torch.Tensor):
        return self.proj(x), {"sector_masses": {self.sector_key: 1.0}}


def _patch_lightweight_attention_layers(monkeypatch) -> None:
    monkeypatch.setattr(models, "ModelA", lambda d, **kwargs: DummyTensorLayer(d))
    monkeypatch.setattr(models, "ModelB", lambda d, **kwargs: DummyTensorLayer(d))
    monkeypatch.setattr(models, "ModelC", lambda d, **kwargs: DummyTensorLayer(d))
    monkeypatch.setattr(
        models,
        "CompoundTransformerLayer",
        lambda n_patches, d, *args, **kwargs: DummyTupleLayer(d, "cross"),
    )
    monkeypatch.setattr(
        models,
        "MultiSectorLayer",
        lambda n_patches, d, *args, **kwargs: DummyTupleLayer(d, "pp"),
    )
    monkeypatch.setattr(
        models,
        "HierarchicalCompoundLayer",
        lambda n_regions, n_patches_per_region, d, *args, **kwargs: DummyTupleLayer(
            d, "triple_cross"
        ),
    )


@pytest.mark.parametrize(
    ("model_type", "circuit_family", "use_cls_token", "use_pos_embed", "embed_dim"),
    [
        ("A", "generic", True, True, 8),
        ("A", "butterfly", True, True, 8),
        ("B", "generic", True, True, 8),
        ("B", "butterfly", True, True, 8),
        ("C", "generic", True, True, 8),
        ("C", "butterfly", True, True, 8),
        ("D", "generic", True, True, 8),
        ("D", "butterfly", False, True, 16),
        ("E", "generic", True, True, 8),
        ("E", "butterfly", False, True, 16),
        ("F", "generic", False, True, 8),
        ("F", "butterfly", False, True, 8),
        ("VisionTransformer", "generic", True, True, 8),
        ("OrthoFNN", "generic", False, False, 8),
        ("OrthoFNN", "butterfly", False, False, 8),
    ],
)
def test_qvt_wrapper_covers_all_model_types(
    monkeypatch,
    model_type: str,
    circuit_family: str,
    use_cls_token: bool,
    use_pos_embed: bool,
    embed_dim: int,
) -> None:
    _patch_lightweight_attention_layers(monkeypatch)

    imgs = torch.randn(2, 3, 28, 28)
    model = models.QVTModel(
        model_type=model_type,
        img_size=28,
        in_channels=3,
        patch_size=7,
        embed_dim=embed_dim,
        n_layers=1,
        n_classes=5,
        use_cls_token=use_cls_token,
        use_pos_embed=use_pos_embed,
        circuit_family=circuit_family,
        device="cpu",
    )
    logits = model(imgs)

    assert logits.shape == (2, 5)
    param_counts = model.count_trainable_params()
    assert param_counts["patch_embed"] > 0
    assert param_counts["attention"] > 0
    assert param_counts["head"] > 0
    assert param_counts["total"] > 0
    if model_type == "OrthoFNN":
        assert param_counts["mlp"] == 0
    else:
        assert param_counts["mlp"] > 0

    if model_type in {"D", "E", "F"}:
        assert model.sector_masses, f"{model_type} should record sector masses"
    else:
        assert model.sector_masses == []

    if model_type in {"F", "OrthoFNN"}:
        assert model.use_cls_token is False

    if model_type == "OrthoFNN":
        assert model.use_pos_embed is False


def test_orthofnn_grayscale_image_embed_matches_784xd_projection(monkeypatch) -> None:
    _patch_lightweight_attention_layers(monkeypatch)

    model = models.QVTModel(
        model_type="OrthoFNN",
        img_size=28,
        in_channels=3,
        embed_dim=16,
        n_layers=1,
        n_classes=5,
        use_cls_token=False,
        use_pos_embed=False,
        image_embed_grayscale=True,
        circuit_family="butterfly",
        device="cpu",
    )

    param_counts = model.count_trainable_params()
    assert param_counts["patch_embed"] == (28 * 28 * 16) + 16
