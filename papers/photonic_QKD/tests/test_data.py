from __future__ import annotations

import common  # noqa: F401
import torch
from lib.data import TensorBatchLoader, generate_synthetic_blobs
from torch.utils.data import TensorDataset


def _dataset(n: int = 10) -> TensorDataset:
    return TensorDataset(torch.arange(n).float().view(n, 1), torch.arange(n))


def test_shuffled_epoch_visits_every_sample_once_with_partial_last_batch():
    loader = TensorBatchLoader(_dataset(10), 4, shuffle=True, seed=0)
    batches = list(loader)
    assert len(loader) == 3
    assert [batch[1].numel() for batch in batches] == [4, 4, 2]
    seen = torch.cat([batch[1] for batch in batches])
    assert sorted(seen.tolist()) == list(range(10))
    for images, labels in batches:
        assert torch.equal(images.view(-1).long(), labels)


def test_shuffle_changes_between_epochs_and_is_seed_reproducible():
    first_loader = TensorBatchLoader(_dataset(50), 50, shuffle=True, seed=3)
    epoch_one = next(iter(first_loader))[1]
    epoch_two = next(iter(first_loader))[1]
    assert not torch.equal(epoch_one, epoch_two)
    replay = TensorBatchLoader(_dataset(50), 50, shuffle=True, seed=3)
    assert torch.equal(next(iter(replay))[1], epoch_one)


def test_unshuffled_loader_keeps_order():
    loader = TensorBatchLoader(_dataset(7), 3, shuffle=False)
    assert torch.cat([batch[1] for batch in loader]).tolist() == list(range(7))


def test_synthetic_blobs_are_balanced_and_bounded():
    images, labels = generate_synthetic_blobs(
        200, seed=1, anchor_seed=0, blob_sigma=1.6, jitter_sigma=0.8, pixel_sigma=0.05
    )
    assert images.shape == (200, 1, 28, 28)
    assert images.min() >= 0 and images.max() <= 1
    assert labels.bincount().tolist() == [20] * 10
