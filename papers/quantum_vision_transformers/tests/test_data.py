from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from lib.data import (
    get_medmnist_loaders,
    make_data_loader_generator,
    select_train_subset,
)


def test_make_data_loader_generator_is_deterministic() -> None:
    gen_a = make_data_loader_generator(123)
    gen_b = make_data_loader_generator(123)
    assert gen_a is not None
    assert gen_b is not None
    assert torch.equal(gen_a.get_state(), gen_b.get_state())


def test_get_medmnist_loaders_uses_seeded_train_shuffle(monkeypatch) -> None:
    medmnist = ModuleType("medmnist")
    medmnist.INFO = {
        "fake": {"label": {"0": "zero", "1": "one"}, "python_class": "FakeDataset"}
    }

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, split, transform=None, download=True, root=None):
            self.transform = transform
            self.samples = []
            base = {"train": 0, "val": 100, "test": 200}[split]
            for i in range(6):
                image = torch.full((1, 2, 2), float(base + i))
                label = torch.tensor([i % 2], dtype=torch.long)
                self.samples.append((image, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            image, label = self.samples[idx]
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    medmnist.FakeDataset = FakeDataset

    torchvision = ModuleType("torchvision")
    transforms = ModuleType("torchvision.transforms")

    class Compose:
        def __init__(self, funcs):
            self.funcs = funcs

        def __call__(self, value):
            for fn in self.funcs:
                value = fn(value)
            return value

    class ToTensor:
        def __call__(self, value):
            return value if isinstance(value, torch.Tensor) else torch.tensor(value)

    transforms.Compose = Compose
    transforms.ToTensor = ToTensor
    torchvision.transforms = transforms

    monkeypatch.setitem(sys.modules, "medmnist", medmnist)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)

    train_a, val_a, test_a, n_classes_a = get_medmnist_loaders(
        "fake", batch_size=2, num_workers=0, download=False, seed=123
    )
    train_b, val_b, test_b, n_classes_b = get_medmnist_loaders(
        "fake", batch_size=2, num_workers=0, download=False, seed=123
    )

    def flatten_batches(loader):
        order = []
        for images, _ in loader:
            order.extend(images[:, 0, 0, 0].tolist())
        return order

    assert n_classes_a == 2
    assert n_classes_b == 2
    assert flatten_batches(train_a) == flatten_batches(train_b)
    assert flatten_batches(val_a) == flatten_batches(val_b)
    assert flatten_batches(test_a) == flatten_batches(test_b)


def test_select_train_subset_is_stratified_and_deterministic() -> None:
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.labels = np.array([[0]] * 6 + [[1]] * 4 + [[2]] * 2)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.tensor(float(idx)), torch.tensor(
                self.labels[idx, 0], dtype=torch.long
            )

    ds = ToyDataset()
    subset_a = select_train_subset(ds, 6, 11, "stratified")
    subset_b = select_train_subset(ds, 6, 11, "stratified")

    assert subset_a.indices == subset_b.indices
    picked = ds.labels[subset_a.indices, 0]
    counts = {
        int(k): int(v)
        for k, v in zip(*np.unique(picked, return_counts=True), strict=True)
    }
    assert counts == {0: 3, 1: 2, 2: 1}


def test_select_train_subset_rejects_multi_label_stratification() -> None:
    class MultiLabelDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.tensor(float(idx)), torch.tensor(
                self.labels[idx], dtype=torch.long
            )

    with pytest.raises(ValueError, match="single-label datasets"):
        select_train_subset(MultiLabelDataset(), 2, 0, "stratified")
