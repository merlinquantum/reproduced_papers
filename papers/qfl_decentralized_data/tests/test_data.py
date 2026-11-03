from __future__ import annotations

import numpy as np
import pytest
import torch
from common import MNIST_DIR, load_named_config
from lib.data import (
    build_dataloader,
    build_mnist_splits,
    dirichlet_partition,
    downsample_area,
    iid_partition,
)

needs_mnist = pytest.mark.skipif(not MNIST_DIR.is_dir(), reason="MNIST not downloaded")


def test_downsample_area_is_the_exact_block_mean():
    image = np.zeros((1, 28, 28), dtype=np.uint8)
    image[0, :7, :7] = 255  # the whole top-left block
    image[0, 7:14, 7:14] = 51  # a block at one fifth intensity
    reduced = downsample_area(image).reshape(4, 4)
    assert reduced[0, 0] == pytest.approx(1.0)
    assert reduced[1, 1] == pytest.approx(0.2)
    assert reduced.sum() == pytest.approx(1.2)


def test_dirichlet_partition_covers_all_samples_once_and_is_unbalanced():
    labels = torch.randint(0, 2, (4000,), generator=torch.Generator().manual_seed(0))
    parts = dirichlet_partition(labels, 4, 0.5, torch.Generator().manual_seed(1), 50)
    merged = torch.cat(parts).sort().values
    assert torch.equal(merged, torch.arange(4000))
    sizes = [len(part) for part in parts]
    assert max(sizes) > 1.3 * min(sizes), sizes
    positive_share = [labels[part].float().mean().item() for part in parts]
    assert max(positive_share) - min(positive_share) > 0.2, positive_share


def test_partition_is_reproducible_from_the_seed():
    labels = torch.randint(0, 4, (1000,), generator=torch.Generator().manual_seed(0))
    first = dirichlet_partition(labels, 3, 0.5, torch.Generator().manual_seed(7), 10)
    second = dirichlet_partition(labels, 3, 0.5, torch.Generator().manual_seed(7), 10)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_partition_fails_loudly_when_no_split_meets_the_minimum():
    labels = torch.zeros(30, dtype=torch.long)
    with pytest.raises(RuntimeError, match="min_client_size|at least"):
        dirichlet_partition(labels, 8, 0.05, torch.Generator().manual_seed(0), 25)


def test_iid_partition_is_balanced():
    parts = iid_partition(1001, 4, torch.Generator().manual_seed(0))
    assert sorted(len(p) for p in parts) == [250, 250, 250, 251]


@needs_mnist
def test_loader_returns_one_batch_of_16_pixels_with_binary_labels():
    cfg = load_named_config("e1_reduced")
    splits = build_mnist_splits(cfg, torch.float32)
    loader = build_dataloader(cfg, "train", splits)
    pixels, labels = next(iter(loader))
    assert pixels.shape == (20, 16), pixels.shape
    assert labels.shape == (20,)
    assert 0.0 <= pixels.min() and pixels.max() <= 1.0
    assert set(labels.tolist()) <= {0, 1}


@needs_mnist
def test_client_loader_serves_only_that_clients_samples():
    cfg = load_named_config("e1_reduced", federation={"n_clients": 3})
    splits = build_mnist_splits(cfg, torch.float32)
    sizes = splits["summary"]["client_sizes"]
    assert sum(sizes) == len(splits["train"][1])
    for client, size in enumerate(sizes):
        loader = build_dataloader(cfg, "train", splits, client=client)
        assert sum(len(batch[1]) for batch in loader) == size
    with pytest.raises(ValueError, match="train split"):
        build_dataloader(cfg, "test", splits, client=0)


@needs_mnist
def test_four_class_task_relabels_classes_in_the_given_order():
    cfg = load_named_config("e2_reduced")
    splits = build_mnist_splits(cfg, torch.float32)
    assert splits["summary"]["classes"] == [2, 3, 6, 7]
    assert set(splits["train"][1].tolist()) == {0, 1, 2, 3}


@needs_mnist
def test_full_mnist_sizes_match_the_original_files():
    cfg = load_named_config("e1_original", dataset={"classes": list(range(10))})
    splits = build_mnist_splits(cfg, torch.float32)
    assert splits["summary"]["split_sizes"] == {"train": 60000, "test": 10000}


@needs_mnist
def test_per_feature_minmax_is_fitted_on_train_and_applied_to_every_split():
    cfg = load_named_config(
        "e1_reduced",
        dataset={"feature_scaling": "per_feature_minmax", "validation_fraction": 0.1},
    )
    splits = build_mnist_splits(cfg, torch.float32)
    train = splits["train"][0]
    assert torch.allclose(train.min(0).values, torch.zeros(16), atol=1e-6)
    assert torch.allclose(train.max(0).values, torch.ones(16), atol=1e-6)
    for name in ("validation", "test"):
        assert 0.0 <= splits[name][0].min() and splits[name][0].max() <= 1.0


@needs_mnist
def test_unknown_feature_scaling_fails_clearly():
    cfg = load_named_config("e1_reduced", dataset={"feature_scaling": "whiten"})
    with pytest.raises(ValueError, match="feature_scaling"):
        build_mnist_splits(cfg, torch.float32)
